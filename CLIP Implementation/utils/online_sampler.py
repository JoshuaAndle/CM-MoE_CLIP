import torch
import torch.distributed as dist
from torch.utils.data.sampler import Sampler
from typing import Optional, Sized, Iterable, Tuple
import logging


class OnlineSampler(Sampler):

    def __init__(self,
                 data_source: Optional[Sized],
                 num_tasks: int,
                 cur_iter: int = 0) -> None:


        print("\n\n\nStarting OnlineSampler preparation")
        self.data_source = data_source
        self.classes = self.data_source.classes
        self.targets = self.data_source.targets
        self.generator = torch.Generator().manual_seed(rnd_seed)

        self.task = cur_iter



        self.disjoint_num = len(self.classes)
        self.blurry_num = len(self.classes) - self.disjoint_num


        print("Number of classes: {}, # disjoint: {}, # blurry: {}".format(len(self.classes), self.disjoint_num, self.blurry_num))




        class_order = torch.randperm(len(self.classes), generator=self.generator)
        self.disjoint_classes = class_order[:]
        print("Shape of disjoint classes: ", self.disjoint_classes.shape)
        self.disjoint_classes = self.disjoint_classes.reshape(num_tasks, -1).tolist()
        print("Shape of reshaped disjoint classes: ", self.disjoint_classes)

        logging.info("disjoint classes: {}".format(self.disjoint_classes))
        # Get indices of disjoint and blurry classes
        self.disjoint_indices = [[] for _ in range(num_tasks)]
        for i in range(len(self.targets)):
            for j in range(num_tasks):
                if self.targets[i] in self.disjoint_classes[j]:
                    self.disjoint_indices[j].append(i)
                    break


        self.indices = [[] for _ in range(num_tasks)]
        for i in range(num_tasks):
            self.indices[i] = self.disjoint_indices[i]
            self.indices[i] = torch.tensor(self.indices[i])[torch.randperm(len(self.indices[i]), generator=self.generator)].tolist()
  
        self.num_samples = int(len(self.indices[self.task]))
        self.total_size = self.num_samples
        self.num_selected_samples = int(len(self.indices[self.task]))

    def __iter__(self) -> Iterable[int]:
        return iter(self.indices[self.task])

    def __len__(self) -> int:
        return self.num_selected_samples

    def set_task(self, cur_iter: int) -> None:
        if cur_iter >= len(self.indices) or cur_iter < 0:
            raise ValueError("task out of range")
        self.task = cur_iter

        self.num_samples = int(len(self.indices[self.task]))
        self.total_size = self.num_samples
        self.num_selected_samples = int(len(self.indices[self.task]))

    def get_task(self, cur_iter: int) -> Iterable[int]:
        indices = self.indices[cur_iter][:self.total_size:1]
        assert len(indices) == self.num_samples
        return indices[:self.num_selected_samples]



### A sampler for online training with the provided multiDataset class in /datasets/multiDataset.py
class OnlineMultiDatasetSampler(Sampler):

    def __init__(self,
                 data_source: Optional[Sized],
                 num_tasks: int,
                 rnd_seed: int = None,                 
                 cur_iter: int = 0) -> None:

        self.data_source = data_source
        self.classes = self.data_source.classes
        self.targets = self.data_source.targets
        self.generator = torch.Generator()

        if rnd_seed is not None:
            self.generator = self.generator.manual_seed(rnd_seed)
        
        ### Note: This assumes the data_source is of the IndexedDataset wrapper class, accessing the multiDataset object as .dataset
        self.sample_task_ids = data_source.dataset.sample_task_ids

        self.task = cur_iter


        self.indices = [[] for _ in range(num_tasks)]
        for t in range(num_tasks):
            for i in range(len(self.targets)):
                if self.sample_task_ids[i] == t:
                    self.indices[t].append(i)

        for i in range(num_tasks):
            self.indices[i] = torch.tensor(self.indices[i])[torch.randperm(len(self.indices[i]), generator=self.generator)].tolist()



        self.num_samples = int(len(self.indices[self.task]))
        self.total_size = self.num_samples
        self.num_selected_samples = int(len(self.indices[self.task]))

    def __iter__(self) -> Iterable[int]:
        return iter(self.indices[self.task])

    def __len__(self) -> int:
        return self.num_selected_samples

    def set_task(self, cur_iter: int) -> None:
        if cur_iter >= len(self.indices) or cur_iter < 0:
            raise ValueError("task out of range")
        self.task = cur_iter

        self.num_samples = int(len(self.indices[self.task]))
        self.total_size = self.num_samples
        self.num_selected_samples = int(len(self.indices[self.task]))

    def get_task(self, cur_iter: int) -> Iterable[int]:
        indices = self.indices[cur_iter][:self.total_size:1]
        assert len(indices) == self.num_samples
        return indices[:self.num_selected_samples]











# class OnlineBatchSampler(Sampler):

#     def __init__(self,
#                  data_source: Optional[Sized],
#                  num_tasks: int,
#                  m: int,
#                  n: int,
#                  rnd_seed: int,
#                  batchsize: int = 16,
#                  online_iter: int = 1,
#                  cur_iter: int = 0,
#                  varing_NM: bool = False) -> None:
#         super().__init__(data_source)
#         self.data_source = data_source
#         self.classes = self.data_source.classes
#         self.targets = self.data_source.targets
#         self.num_tasks = num_tasks
#         self.m = m
#         self.n = n
#         self.rnd_seed = rnd_seed
#         self.batchsize = batchsize
#         self.online_iter = online_iter
#         self.cur_iter = cur_iter
#         self.varing_NM = varing_NM

#         self.disjoint_num = len(self.classes) * self.n // 100
#         self.disjoint_num = int(self.disjoint_num // num_tasks) * num_tasks
#         self.blurry_num = len(self.classes) - self.disjoint_num
#         self.blurry_num = int(self.blurry_num // num_tasks) * num_tasks

#         if not self.varing_NM:
#             # Divide classes into N% of disjoint and (100 - N)% of blurry
#             class_order = torch.randperm(len(self.classes),
#                                          generator=self.generator)
#             self.disjoint_classes = class_order[:self.disjoint_num]
#             self.disjoint_classes = self.disjoint_classes.reshape(num_tasks, -1).tolist()
#             self.blurry_classes = class_order[self.disjoint_num:self.disjoint_num +
#                                               self.blurry_num]
#             self.blurry_classes = self.blurry_classes.reshape(num_tasks,
#                                                               -1).tolist()

#             logging.info("disjoint classes: {}".format(self.disjoint_classes))
#             logging.info("blurry classes: {}".format(self.blurry_classes))
#             # Get indices of disjoint and blurry classes
#             self.disjoint_indices = [[] for _ in range(num_tasks)]
#             self.blurry_indices = [[] for _ in range(num_tasks)]
#             for i in range(len(self.targets)):
#                 for j in range(num_tasks):
#                     if self.targets[i] in self.disjoint_classes[j]:
#                         self.disjoint_indices[j].append(i)
#                         break
#                     elif self.targets[i] in self.blurry_classes[j]:
#                         self.blurry_indices[j].append(i)
#                         break

#             # Randomly shuffle M% of blurry indices
#             blurred = []
#             for i in range(num_tasks):
#                 blurred += self.blurry_indices[i][:len(self.blurry_indices[i]) * m // 100]
#                 self.blurry_indices[i] = self.blurry_indices[i][len(self.blurry_indices[i]) * m // 100:]
#             blurred = torch.tensor(blurred)
#             blurred = blurred[torch.randperm(len(blurred), generator=self.generator)].tolist()
#             logging.info("blurry indices: {}".format(len(blurred)))
#             num_blurred = len(blurred) // num_tasks
#             for i in range(num_tasks):
#                 self.blurry_indices[i] += blurred[:num_blurred]
#                 blurred = blurred[num_blurred:]

#             self.indices = [[] for _ in range(num_tasks)]
#             for i in range(num_tasks):
#                 logging.info("task %d: disjoint %d, blurry %d" %
#                              (i, len(self.disjoint_indices[i]),
#                               len(self.blurry_indices[i])))
#                 self.indices[i] = self.disjoint_indices[i] + self.blurry_indices[i]
#                 self.indices[i] = torch.tensor(self.indices[i])[torch.randperm(len(self.indices[i]), generator=self.generator)]
#                 num_batches = int(self.indices[i].size(0) // self.batchsize)
#                 rest = self.indices[i].size(0) % self.batchsize
#                 self.indices[i] = self.indices[i][:num_batches * self.batchsize].reshape(-1, self.batchsize).repeat(self.online_iter, 1).flatten().tolist() + self.indices[i][-rest:].tolist()
#         else:
#             # Divide classes into N% of disjoint and (100 - N)% of blurry
#             class_order = torch.randperm(len(self.classes),
#                                          generator=self.generator)
#             self.disjoint_classes = class_order[:self.disjoint_num].tolist()
#             if self.disjoint_num > 0:
#                 self.disjoint_slice = [0] + torch.randint(0,
#                     self.disjoint_num, (num_tasks - 1, ),
#                     generator=self.generator).sort().values.tolist() + [self.disjoint_num]
#                 self.disjoint_classes = [self.disjoint_classes[self.disjoint_slice[i]:self.disjoint_slice[i + 1]]
#                     for i in range(num_tasks)]
#             else:
#                 self.disjoint_classes = [[] for _ in range(num_tasks)]

#             self.blurry_classes = class_order[self.disjoint_num:self.disjoint_num +
#                                               self.blurry_num]
#             self.blurry_classes = self.blurry_classes.reshape(num_tasks,
#                                                               -1).tolist()

#             logging.info("disjoint classes: {}".format(self.disjoint_classes))
#             logging.info("blurry classes: {}".format(self.blurry_classes))

#             # Get indices of disjoint and blurry classes
#             self.disjoint_indices = [[] for _ in range(num_tasks)]
#             self.blurry_indices = [[] for _ in range(num_tasks)]
#             num_blurred = 0
#             for i in range(len(self.targets)):
#                 for j in range(num_tasks):
#                     if self.targets[i] in self.disjoint_classes[j]:
#                         self.disjoint_indices[j].append(i)
#                         break
#                     elif self.targets[i] in self.blurry_classes[j]:
#                         self.blurry_indices[j].append(i)
#                         num_blurred += 1
#                         break

#             # Randomly shuffle M% of blurry indices
#             blurred = []
#             num_blurred = num_blurred * m // 100
#             num_blurred = [0] + torch.randint(0, num_blurred, (num_tasks - 1, ), generator=self.generator).sort().values.tolist() + [num_blurred]

#             for i in range(num_tasks):
#                 blurred += self.blurry_indices[i][:num_blurred[i + 1] - num_blurred[i]]
#                 self.blurry_indices[i] = self.blurry_indices[i][num_blurred[i + 1] - num_blurred[i]:]
#             blurred = torch.tensor(blurred)
#             blurred = blurred[torch.randperm(len(blurred), generator=self.generator)].tolist()
#             logging.info("blurry indices: {}".format(len(blurred)))
#             # num_blurred = len(blurred) // num_tasks
#             for i in range(num_tasks):
#                 self.blurry_indices[i] += blurred[:num_blurred[i + 1] -
#                                                   num_blurred[i]]
#                 blurred = blurred[num_blurred[i + 1] - num_blurred[i]:]

#             self.indices = [[] for _ in range(num_tasks)]
#             for i in range(num_tasks):
#                 logging.info("task %d: disjoint %d, blurry %d" %
#                              (i, len(self.disjoint_indices[i]),
#                               len(self.blurry_indices[i])))
#                 self.indices[i] = self.disjoint_indices[i] + self.blurry_indices[i]
#                 self.indices[i] = torch.tensor(self.indices[i])[torch.randperm(len(self.indices[i]), generator=self.generator)].tolist()
#                 num_batches = int(self.indices[i].size(0) // self.batchsize)
#                 rest = self.indices[i].size(0) % self.batchsize
#                 self.indices[i] = self.indices[i][:num_batches * self.batchsize].reshape(-1, self.batchsize).repeat(self.online_iter, 1).flatten().tolist() + self.indices[i][-rest:].tolist()

#     def __iter__(self) -> Iterable[int]:
#         return iter(self.indices[self.task])

#     def __len__(self) -> int:
#         return self.num_selected_samples

#     def set_task(self, cur_iter: int) -> None:

#         if cur_iter >= len(self.indices) or cur_iter < 0:
#             raise ValueError("task out of range")
#         self.task = cur_iter

#         self.num_samples = int(len(self.indices[self.task]))
#         self.total_size = self.num_samples
#         self.num_selected_samples = int(len(self.indices[self.task]))

#     def get_task(self, cur_iter: int) -> Iterable[int]:
#         indices = self.indices[cur_iter][0:self.total_size:1]
#         assert len(indices) == self.num_samples
#         return indices[:self.num_selected_samples]

#     def get_task_classes(self, cur_iter: int) -> Iterable[int]:
#         return list(set(self.classes[self.indices[cur_iter]]))










class OnlineTestSampler(Sampler):

    def __init__(self,
                 data_source: Optional[Sized],
                 exposed_class: Iterable[int]) -> None:
        self.data_source = data_source
        self.classes = self.data_source.classes
        self.targets = self.data_source.targets
        self.exposed_class = exposed_class
        # print("Exposed classes for Online Test Sampler: ", self.exposed_class)
        print("# Exposed classes for Online Test Sampler: ", len(self.exposed_class))

        # print("Targets for online test sampler: ", sorted(set(self.targets)))
        # print("Length of data source length in test sampler: ", self.data_source.__len__())

        #!# Seems to be set up for class incremental if it covers all seen classes, unless you limit exposed classes to a single task at a time in _trainer.py
        self.indices = [i for i in range(self.data_source.__len__())
            if self.targets[i] in self.exposed_class]
        # matched_targets = torch.tensor([self.targets[idx] for idx in self.indices])
        # print("Unique test targets: ", matched_targets.unique(), " ", matched_targets.shape)

        #!# Note: Setting this to shuffle for activation collection since we only want to save n batches.
        ###       This should be changed back afterwards as test data is typically not shuffled
        self.indices = torch.tensor(self.indices)[torch.randperm(len(self.indices))].tolist()


        self.num_samples = int(len(self.indices))
        self.total_size = self.num_samples
        self.num_selected_samples = int(len(self.indices))

    def __iter__(self) -> Iterable[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return self.num_selected_samples



