import math
import random
import torch
from torch import Tensor
from torch.utils.data import Dataset
import torch.distributed as dist
import numpy as np
from typing import Optional, Sized, Iterable, Tuple, Union

#############################################################################################
### Memory Classes
#############################################################################################

class Memory:
    """
    Stores samples to memory buffer
    These are stored as indices for the data source with the corresponding sample label

    Note: As we do not use it, the storage of image data was commented out to reduce object size
        - Need to keep an eye on whether this is needed for any other benchmark methods
    """
    def __init__(self, args, data_source: Dataset=None) -> None:
        
        self.data_source = data_source
        self.args = args

        ### Moved memory size and # samples seen into the memory class to enable multiple buffers being tracked independently
        self.memory_size = 0
        #!# We are allowing the memory buffer to expand for each task so that we can have longer task sequences while maintaining reasonable memory_per_class
        self.memory_per_task = args.memory_size

        ### If we do not specify a value then it is dynamically determined by memory_size and the number of known classes
        if args.memory_per_class == 0:
            self.memory_per_class = -1
        else: 
            self.memory_per_class = args.memory_per_class
        self.seen = {}

        self.current_task = -1
        self.classes_by_task = {}
        self.class_names_by_task = {}

        self.memory = torch.empty(0)
        self.labels = torch.empty(0)
        self.label_names = np.empty(0) # Need to use numpy since torch doesn't support strings and we still need indexing options
        
        self.cls_list = []
        # if self.args.use_memory_class_names:
        #     self.cls_list = np.empty(0)
        # else:
        #     self.cls_list = torch.empty(0)
        self.cls_count = {}
        self.cls_train_cnt = {}
        self.previous_idx = torch.empty(0)




    def add_task(self, task_id, class_names = None):
        ### When a new task is encountered, add it to the memory and adjust memory size if allocating memory by task
        if self.args.memory_per_class == 0:
            self.memory_size += self.memory_per_task
            print("New memory size: ", self.memory_size)

        if task_id not in self.classes_by_task.keys():
            self.classes_by_task[task_id] = []

        if class_names is not None:
            self.class_names_by_task[task_id] = class_names


    def set_task(self, task_id):
        self.current_task = task_id










    def remove_samples(self, cls, excess_count) -> None:
        if self.args.use_memory_class_names:
            class_indices = torch.from_numpy(self.label_names == cls).nonzero(as_tuple=True)[0]
        else:
            class_indices = (self.labels == cls).nonzero(as_tuple=True)[0]
        
        random_removal_indices = torch.randperm(len(class_indices))
        ### Shuffle the indices corresponding to given class, then take the first ones to be removed
        random_removal_indices = class_indices[random_removal_indices][:excess_count]

        kept_indices = torch.arange(len(self.labels))
        kept_indices = kept_indices[~torch.isin(kept_indices, random_removal_indices)]
        self.memory = self.memory[kept_indices]
        self.labels = self.labels[kept_indices]
        self.label_names = self.label_names[kept_indices]
        self.cls_count[cls] -= excess_count


    ### Note: Samples would only be removed if the number of known tasks increases or the memory size decreases
    def resize_class_memory(self):
        print("Resizing memory buffer!")
        for c in self.cls_list:
            # excess_count = self.cls_count[c.item()] - self.memory_per_class
            excess_count = self.cls_count[c] - self.memory_per_class
            if excess_count > 0:
                self.remove_samples(c, excess_count)





    ### Note: cls_list is not necessarily ordered as increasing or contiguous integers
    def add_new_class(self, cls_list: Union[Iterable[int], Iterable[str]]) -> None:
        _memory_size = self.memory_size

        # if self.args.use_memory_class_names:
        #     self.cls_list = np.array(cls_list)
        # else:
        #     self.cls_list = torch.tensor(cls_list)
        self.cls_list = cls_list

        for c in cls_list:
            if c not in self.cls_count.keys():
                self.classes_by_task[self.current_task].append(c)
                self.cls_count[c] = 0

                self.cls_train_cnt[c] = 0
                self.seen[c] = 0

                #!# This is a bit of an awkward naming issue that I need to deal with, but its a fitting name for both the arg and the variable
                self.memory_size += self.args.memory_per_class

        _memory_per_class = self.memory_per_class
        if self.args.memory_per_class == 0:
            self.memory_per_class = math.floor(self.memory_size/len(self.cls_list))


        if self.memory_per_class != _memory_per_class:
            print("New available memory per class: ", self.memory_per_class)
            ### Check if any previous classes need to have samples removed to allow new classes to be stored
            self.resize_class_memory()


        if self.memory_size != _memory_size:
            print("New available memory size: ", self.memory_size)


    def replace_data(self, data: Tuple[Tensor, Tensor, str], idx: int=None) -> None:
        index, label, label_name = data
        # if self.data_source is not None:
        #     _, label = self.data_source.__getitem__(index)

        ### When class capacity is not yet filled
        if idx is None:
            self.memory = torch.cat([self.memory, torch.tensor([index])])
            self.labels = torch.cat([self.labels, torch.tensor([label])])
            self.label_names = np.concatenate([self.label_names, np.array([label_name])])
            if self.args.use_memory_class_names:
                self.cls_count[label_name] += 1
            else:
                self.cls_count[label] += 1

        else:
            if self.args.use_memory_class_names:
                # replaced_label = self.label_names[idx]
                self.memory[idx] = index
                self.labels[idx] = label
                self.label_names[idx] = label_name
                # ### Should always be the same label, but left this in for flexibility of use (e.g. in case we want to allow replacement of any class)
                # self.cls_count[replaced_label.item()] -= 1
                # self.cls_count[label_name] += 1
    
            else:
                # replaced_label = self.labels[idx]
                self.memory[idx] = index
                self.labels[idx] = label
                self.label_names[idx] = label_name
                # ### Should always be the same label, but left this in for flexibility of use (e.g. in case we want to allow replacement of any class)
                # self.cls_count[replaced_label.item()] -= 1
                # self.cls_count[label] += 1


    def __len__(self) -> int:
        return len(self.labels)



class DummyMemory(Memory):
    def __init__(self, data_source: Dataset=None, shape: Tuple[int, int, int]=(3, 32, 32), datasize: int=100) -> None:
        super(DummyMemory, self).__init__(data_source)
        self.shape = shape
        self.datasize = datasize
        self.images = torch.rand(self.datasize, *self.shape)
        self.labels = torch.randint(0, 10, (self.datasize,))
        self.cls_list = torch.unique(self.labels)
        self.cls_count = {}
        self.cls_train_cnt = {}
        self.others_loss_decrease = torch.zeros(self.datasize)














#############################################################################################
### Sampler Classes
#############################################################################################



class MemorySubnetSampler(torch.utils.data.Sampler):
    def __init__(self, memory: Memory, tasks: list = [], shuffle: bool = False) -> None:
        self.memory = memory
        self.shuffle = shuffle


        if memory.args.use_memory_class_names:
            valid_class_names = []

            #!# This is a little hacky but I'm going directly to the data source to get the class names
            ### This should be fine when using the MultiDataset class, since any of these names that were not encountered yet
            ###   will simply not exist in the memory to begin with so they won't lead to erroneous matches
            for task in tasks:
                valid_class_names.extend(self.memory.class_names_by_task[task])

            ### Get only the labels corresponding to the given subnetwork, if one is given
            if len(valid_class_names) > 0:
                valid_indices = np.isin(self.memory.label_names, np.array(valid_class_names)).nonzero()[0]
            #!# This feels like a weird design choice? Why not just raise an error?
            else:
                valid_indices = torch.tensor(len(self.memory), dtype=torch.int64)

        else:
            valid_classes = []
            for task in tasks:
                valid_classes.extend(self.memory.classes_by_task[task])


            ### Get only the labels corresponding to the given subnetwork, if one is given
            if len(valid_classes) > 0:
                valid_indices = torch.isin(self.memory.labels, torch.tensor(valid_classes))
                valid_indices = valid_indices.nonzero(as_tuple=True)[0]
            else:
                valid_indices = torch.tensor(len(self.memory), dtype=torch.int64)



        self.indices = valid_indices.tolist()
        ### Mapping the memory indices to the data_source sample indices
        for i, idx in enumerate(self.indices):
            self.indices[i] = int(self.memory.memory[idx])
    
    def __iter__(self):
        shuffled_indices = self.indices[:]
        if self.shuffle == True:
            random.shuffle(shuffled_indices)

        return iter(shuffled_indices)


    def __len__(self):
        return len(self.indices)
    




class MemorySubnetBatchSampler(torch.utils.data.Sampler):
    def __init__(self, memory: Memory, batch_size: int, iterations: int = 1, tasks: list = []) -> None:
        self.memory = memory
        self.batch_size = batch_size
        self.iterations = int(iterations)

        if memory.args.use_memory_class_names:
            valid_class_names = []
            for task in tasks:
                valid_class_names.extend(self.memory.class_names_by_task[task])

            ### Get only the labels corresponding to the given subnetwork, if one is given
            if len(valid_class_names) > 0:
                valid_indices = np.isin(self.memory.label_names, np.array(valid_class_names)).nonzero()[0]
            else:
                valid_indices = torch.tensor(len(self.memory), dtype=torch.int64)

            ### Shuffle the indices for each iteration that the sampler is used for
            concat_indices = []
            for _ in range(self.iterations):
                shuffled_indices = valid_indices[torch.randperm(len(valid_indices), dtype=torch.int64)]
                concat_indices.append(shuffled_indices[:min(self.batch_size, len(self.memory))])


        else:
            valid_classes = []
            for task in tasks:
                valid_classes.extend(self.memory.classes_by_task[task])

            ### Get only the labels corresponding to the given subnetwork, if one is given
            if len(valid_classes) > 0:
                valid_indices = torch.isin(self.memory.labels, torch.tensor(valid_classes))
                valid_indices = valid_indices.nonzero(as_tuple=True)[0]
            else:
                valid_indices = torch.tensor(len(self.memory), dtype=torch.int64)

            ### Shuffle the indices for each iteration that the sampler is used for
            concat_indices = []
            for _ in range(self.iterations):
                shuffled_indices = valid_indices[torch.randperm(len(valid_indices), dtype=torch.int64)]
                concat_indices.append(shuffled_indices[:min(self.batch_size, len(self.memory))])


        self.indices = torch.cat(concat_indices).tolist()
        for i, idx in enumerate(self.indices):
            self.indices[i] = int(self.memory.memory[idx])
    
    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
    






class MemoryBatchSampler(torch.utils.data.Sampler):
    def __init__(self, memory: Memory, batch_size: int, iterations: int = 1) -> None:
        self.memory = memory
        self.batch_size = batch_size
        self.iterations = int(iterations)
        self.indices = torch.cat([torch.randperm(len(self.memory), dtype=torch.int64)[:min(self.batch_size, len(self.memory))] for _ in range(self.iterations)]).tolist()
        for i, idx in enumerate(self.indices):
            self.indices[i] = int(self.memory.memory[idx])
    
    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
    

class BatchSampler(torch.utils.data.Sampler):
    def __init__(self, samples_idx: int, batch_size: int, iterations: int = 1) -> None:
        self.samples_idx = samples_idx
        self.batch_size = batch_size
        self.iterations = int(iterations)
        self.indices = torch.cat([torch.randperm(len(self.samples_idx), dtype=torch.int64)[:min(self.batch_size, len(self.samples_idx))] for _ in range(self.iterations)]).tolist()
        for i, idx in enumerate(self.indices):
            self.indices[i] = int(self.samples_idx[idx])
    
    def __iter__(self) -> Iterable[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)

class MemoryOrderedSampler(torch.utils.data.Sampler):
    def __init__(self, memory: Memory, batch_size: int, iterations: int = 1) -> None:
        self.memory = memory
        self.batch_size = batch_size
        self.iterations = int(iterations)
        self.indices = torch.cat([torch.arange(len(self.memory), dtype=torch.int64) for _ in range(self.iterations)]).tolist()
        for i, idx in enumerate(self.indices):
            self.indices[i] =  int(self.memory.memory[idx])
    
    def __iter__(self) -> Iterable[int]:
        if dist.is_initialized():
            return iter(self.indices[dist.get_rank()::dist.get_world_size()])
        else:
            return iter(self.indices)
    def __len__(self) -> int:
        if dist.is_initialized():
            return len(self.indices[dist.get_rank()::dist.get_world_size()])
        else:
            return len(self.indices)