"""Contains utility functions for calculating activations and connectivity. Adapted code is acknowledged in comments"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from AuxiliaryScripts import DataGenerator as DG
from AuxiliaryScripts import cifarsubsets
from AuxiliaryScripts import network as net

import time
import copy
import math
import sklearn
import random 

import scipy.spatial     as ss

from math                 import log, sqrt
from scipy                import stats
from sklearn              import manifold
from scipy.special        import *
from sklearn.neighbors    import NearestNeighbors


from torch.autograd import Variable, grad


acts = {}

"""
hook_fn(), activations(), and get_all_layers() 
adapted from: https://blog.paperspace.com/pytorch-hooks-gradient-clipping-debugging/ 
    and https://web.stanford.edu/~nanbhas/blog/forward-hooks-pytorch/
    Need to cite the latter as:
    @article{ nanbhas2020forwardhook,
      title   = "Intermediate Activations — the forward hook",
      author  = "Bhaskhar, Nandita",
      journal = "Blog: Roots of my Equation (web.stanford.edu/~nanbhas/blog/)",
      year    = "2020",
      url     = "https://web.stanford.edu/~nanbhas/blog/forward-hooks-pytorch/"
    }
"""
### Returns a hook function directed to store activations in a given dictionary key "name"
def getActivation(name):
    # the hook signature
    def hook(model, input, output):
        acts[name] = output.detach().cpu()
    return hook

### Collected from ReLu layers when possible, but not all resnet18 trainable layers have coupled relu layers
def get_all_layers(net, hook_handles, relu_idxs):
    for module_idx, module in enumerate(net.shared.modules()):
        if module_idx in relu_idxs:
            hook_handles.append(module.register_forward_hook(getActivation(module_idx)))


### Process and record all of the activations for the given pair of layers
def activations(data_loader, model, cuda, relu_idxs, act_idxs, using_dataloader = True):
    temp_op       = None
    temp_label_op = None

    parents_op  = None
    labels_op   = None

    handles     = []

    ### Set hooks in all tunable layers
    get_all_layers(model, handles, relu_idxs)

    ### A dictionary for storing the activations
    actsdict = {}
    labels = None
    
    for i in act_idxs: 
        actsdict[i] = None
    
    ### Note: The stacking will cause an error if the dataloader only has one batch, seemingly. Not currently an issue, but leaving this just in case
    with torch.no_grad():
        ### If we're using a dataloader then we need to iterate over batches, otherwise we need to manually split batches off a larger tensor
        if using_dataloader:
            for step, data in enumerate(data_loader):
                x_input, y_label = data
                model(x_input.cuda())

    
                if step == 0:
                    labels = y_label.detach().cpu()
                    for key in acts.keys():
                        ### We need to convert from relu idxs to trainable layer idxs for future masking purposes
                        acts_idx = act_idxs[relu_idxs.index(key)]
                        # print("")
                        if len(acts[key].shape) > 2:
                            actsdict[acts_idx] = acts[key].mean(dim=3).mean(dim=2)
                        else:
                            actsdict[acts_idx] = acts[key]

                else: 
                    labels = torch.cat((labels, y_label.detach().cpu()),dim=0)
                    for key in acts.keys():
                        acts_idx = act_idxs[relu_idxs.index(key)]
                        if len(acts[key].shape) > 2:
                            actsdict[acts_idx] = torch.cat((actsdict[acts_idx], acts[key].mean(dim=3).mean(dim=2)), dim=0)
                        else:
                            actsdict[acts_idx] = torch.cat((actsdict[acts_idx], acts[key]), dim=0)
                            
        else:
            ### Given a set of data from the buffer, we need to still split it by batchsize to ensure it fits in memory
            finalstep = math.ceil(data_loader['x'].size()[0]/512)-1
            for step in range(0,math.ceil(data_loader['x'].size()[0]/512)):
                if step < finalstep:
                    x_input = data_loader['x'][step*512:(step+1)*512]
                    y_label = data_loader['y'][step*512:(step+1)*512]
                else:
                    x_input = data_loader['x'][step*512:]
                    y_label = data_loader['y'][step*512:]
                model(x_input.cuda())
    
                if step == 0:
                    labels = y_label.detach().cpu()
                    for key in acts.keys():
                        ### We need to convert from relu idxs to trainable layer idxs for future masking purposes
                        acts_idx = act_idxs[relu_idxs.index(key)]
                        if len(acts[key].shape) > 2:
                            actsdict[acts_idx] = acts[key].mean(dim=3).mean(dim=2)
                        else:
                            actsdict[acts_idx] = acts[key]
                else: 
                    labels = torch.cat((labels, y_label.detach().cpu()),dim=0)
                    for key in acts.keys():
                        acts_idx = act_idxs[relu_idxs.index(key)]
                        if len(acts[key].shape) > 2:
                            actsdict[acts_idx] = torch.cat((actsdict[acts_idx], acts[key].mean(dim=3).mean(dim=2)), dim=0)
                        else:
                            actsdict[acts_idx] = torch.cat((actsdict[acts_idx], acts[key]), dim=0)
            
    # Remove all hook handles
    for handle in handles:
        handle.remove()    
    
    return actsdict, labels















### Get the prediction accuracy for all classifiers on new set's data
def get_all_accs(data_loader, model, classifiers, arch, numclasses):

    accs = {}
    for task in classifiers:
        accs[task] = 0

    
    with torch.no_grad():
        ### Returns after the first batch since we only want to look 1 batch ahead
        for step, data in enumerate(data_loader):
            x_input, y_label = data
            x_input, y_label = x_input.cuda(), y_label.cuda()
            ### Run through every layer except the classifier and collect the outputs
            outputs = model(x_input, use_classifier = False)

            ### For every layer that has recorded activations
            for task in classifiers:
                # print("Task Classifier: ", task, " with weights: ", classifiers[task].weight.data[0,0:10])
                preds = classifiers[task](outputs)
                # print("Shape of predictions: ", preds.size())
                preds = torch.argmax(preds, dim=1)
                # print("Shape of predictions: ", preds.size())
    
                correct = (y_label == preds).long().sum()
                total = y_label.numel()
                
                accs[task] = torch.tensor(100*(correct/total))
                
        

            ### Returning after collecting only the first batch. We're assuming some number of batches in a row belong to the same task, so we only want to evaluate the 
            ###    first batch to make our decision to avoid looking ahead for data that has yet to "arrive"
            return accs            
    





### Add current training batch samples to the growing task buffer, up to N per class
def add_samples_to_buffer(x,y, buffer, numclasses, task, N):
 
    counts = {}
            
    TaskTensor = {}
    if len(list(buffer.keys())) > 0:
        TaskTensor['x'] = buffer['x']
        TaskTensor['y'] = buffer['y']
        TaskTensor['z'] = buffer['z']
        bins = torch.bincount(TaskTensor['y'])
        for i in range(numclasses):
            if i < len(bins):
                counts[i] = bins[i]
            else:
                counts[i] = 0
    else:
        TaskTensor['x'] = None
        TaskTensor['y'] = None
        TaskTensor['z'] = None
        for i in range(numclasses):
            counts[i] = 0
        
        
   
   
    for i, img in enumerate(x):
        label = int(y[i])
        if counts[label] < N:
            if TaskTensor['x'] == None:
                ### We store the images, labels, and task ID for buffered samples so we can track which tasks each sample belongs to when retraining subnetworks
                TaskTensor['x'] = torch.unsqueeze(img, dim=0)
                TaskTensor['y'] = torch.unsqueeze(y[i],dim=0)
                TaskTensor['z'] = torch.unsqueeze(torch.tensor(task),dim=0)
            else:
                TaskTensor['x'] = torch.cat((TaskTensor['x'],torch.unsqueeze(img, dim=0)))
                TaskTensor['y'] = torch.cat((TaskTensor['y'],torch.unsqueeze(y[i],dim=0)))
                TaskTensor['z'] = torch.cat((TaskTensor['z'],torch.unsqueeze(torch.tensor(task),dim=0)))
            
            counts[label] += 1

    printout=False
    for key in list(counts.keys()):
        if counts[key] < N:
            printout=True
    if printout == True:
        print("Label Counts: " , counts)

    return TaskTensor







### Create a buffer for the current task from all batches
def get_samples_to_buffer(dataset, numclasses, task, N):
 
    TaskTensor = {}
    TaskTensor['x'] = None
    TaskTensor['y'] = None
    TaskTensor['z'] = None
   
    counts = {}
    for i in range(numclasses):
        counts[i] = 0
   
    for i, img in enumerate(dataset['x']):
        label = int(dataset['y'][i])
        if counts[label] < N:
            if TaskTensor['x'] == None:
                ### We store the images, labels, and task ID for buffered samples so we can track which tasks each sample belongs to when retraining subnetworks
                TaskTensor['x'] = torch.unsqueeze(img, dim=0)
                TaskTensor['y'] = torch.unsqueeze(dataset['y'][i],dim=0)
                TaskTensor['z'] = torch.unsqueeze(torch.tensor(task),dim=0)
            else:
                TaskTensor['x'] = torch.cat((TaskTensor['x'],torch.unsqueeze(img, dim=0)))
                TaskTensor['y'] = torch.cat((TaskTensor['y'],torch.unsqueeze(dataset['y'][i],dim=0)))
                TaskTensor['z'] = torch.cat((TaskTensor['z'],torch.unsqueeze(torch.tensor(task),dim=0)))
            
            counts[label] += 1

    for key in list(counts.keys()):
        if counts[key] < N:
            print("Label Count for: ", key, " is ", counts[key])

    return TaskTensor



def get_numclasses(dataset):
    if dataset == 'online_mixed_cifar_pmnist' or dataset == 'online_simplemixed_cifar_pmnist':
        numclasses = [10] * 28            
    elif dataset == 'online_cifar_rotmnist':
        numclasses = [10] * 23         
    elif dataset == 'online_cifar_jump20rotmnist':
        numclasses = [10] * 20                  
    elif dataset == 'online_cifar_jump30rotmnist':
        numclasses = [10] * 19         
    return numclasses
    
### Returns a dictionary of "train", "valid", and "test" data+labels for the appropriate cifar subset
def get_dataset(dataset, batch_size,setsize, num_workers=4, pin_memory=False, normalize=None, set_num=0, set="train", printpath=False):
    print("get dataset: ", dataset, " ", batch_size, " ", setsize, " ", set_num)

    ### Experiments    
    if dataset == "online_mixed_cifar_pmnist":
        dataset = cifarsubsets.get_online_mixed_cifar_pmnist(set_num = set_num, setsize=setsize, printpath=printpath)    
    elif dataset == "online_simplemixed_cifar_pmnist":
        dataset = cifarsubsets.get_online_simplemixed_cifar_pmnist(set_num = set_num, setsize=setsize, printpath=printpath)       
    elif dataset == "online_cifar_rotmnist":
        dataset = cifarsubsets.get_online_cifar_rotmnist(set_num = set_num, setsize=setsize, printpath=printpath)            
    elif dataset == "online_cifar_jump20rotmnist":
        dataset = cifarsubsets.get_online_cifar_jump20rotmnist(set_num = set_num, setsize=setsize, printpath=printpath)  
    elif dataset == "online_cifar_jump30rotmnist":
        dataset = cifarsubsets.get_online_cifar_jump30rotmnist(set_num = set_num, setsize=setsize, printpath=printpath)  
        
    else: 
        print("Incorrect dataset for get_dataloader()")
        return -1
        

    return dataset


### Returns a dictionary of "train", "valid", and "test" data+labels for the appropriate cifar subset
def get_dataloader(dataset, batch_size,setsize, num_workers=4, pin_memory=False, normalize=None, set_num=0, set="train"):
    print("get dataloader: ", dataset, " ", batch_size, " ", setsize, " ", set_num)
    ### Get the dataset from the helper function then process it into a dataloader
    dataset = get_dataset(dataset, batch_size,setsize, num_workers, pin_memory, normalize, set_num, set, printpath=True)
    
    # print(dataset, flush=True)
    ### Makes a custom dataset for CIFAR through torch
    generator = DG.CifarDataGenerator(dataset['x'],dataset['y'])

    ### Loads the custom data into the dataloader
    if set == "train":
        return data.DataLoader(generator, batch_size = batch_size, shuffle = True, num_workers = num_workers, pin_memory=pin_memory)
    else:
        return data.DataLoader(generator, batch_size = batch_size, shuffle = False, num_workers = num_workers, pin_memory=pin_memory)



### Saves a checkpoint of the model
def save_ckpt(manager, savename):
    """Saves model to file."""

    # Prepare the ckpt.
    ckpt = {
        'args': manager.args,
        'all_task_masks': manager.all_task_masks,
        # 'network': manager.network,
        'manager': manager,
    }

    # Save to file.
    torch.save(ckpt, savename)







#####################################################
###    Masking Functions
#####################################################


### Get a binary mask where all previously frozen weights are indicated by a value of 1
### After pruning on the current task, this will still return the same masks, as the new weights aren't frozen until the task ends
def get_frozen_mask(weights, module_idx, all_task_masks, subnetID):
    mask = torch.zeros(weights.shape)
    ### Include all weights used in past tasks (which would have been subsequently frozen)
    for i in range(0, subnetID):
        if i == 0:
            mask = all_task_masks[i][module_idx].clone().detach()
        else:
            mask = torch.maximum(all_task_masks[i][module_idx], mask)
    return mask
        
    
### Get a binary mask where all unpruned, unfrozen weights are indicated by a value of 1
### Unlike get_frozen_mask(), this mask will change after pruning since the pruned weights are no longer trainable for the current task
def get_trainable_mask(module_idx, all_task_masks, subnetID):
    ### Get all of the indices included in the given subnetwork
    mask = all_task_masks[subnetID][module_idx]
    

    return mask
        
        
        
        
### Doesnt depend on the state of the network, just needs up-to-date alltaskmask masks
def get_omitted_outgoing_mask(all_task_masks, task_num, network, arch):
    omit_mask = {}
    parent_mask = torch.zeros(0,0,0,0)
    child_mask = torch.zeros(0,0,0,0)
    iteration = 0
    
    """ 
    Note: A limitation I had to make here is that the 1x1 Conv skip layers in downsample aren't considered when omitting outgoing weights (their outgoing values aren't omitted if
    they are). This may lead to slightly unpredictable behavior if that omitted filter is later shared alongside the current task, as the weight is allowed to be nonzero this way, but
    this mostly affects interpretability and isn't expected to affect performance significantly, and won't matter unless multiple past tasks are shared with the current task.
    """
    for module_idx, module in enumerate(network.model.shared.modules()):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            ### Need to explicitly handle skip connections since they dont correspond to the immediate parent
            ### This will rely on the outputs of downsample skip layers being pruned identically to the outputs theyre added back into. We prune layer 29 based on layer 23,
            ###   assuming that 23 and 26 are pruned identically prior to the addition of their outputs. This is true at least for our pruning function
            if module_idx not in [13,23,34,44,55,65,76,86] or arch != "modresnet18":
                # print("Iteration is:", iteration)
                parent_mask = child_mask
                child_mask = all_task_masks[task_num][module_idx].clone().detach()
                omit_layer = torch.ones(child_mask.size())
            
                if iteration > 0:
                    ### This will hold true where all weights in the parent mask at dim 0 are zero 
                    ### For task mask this means the filter wasn't used in the task. Its pruned or omitted
                    if len(parent_mask.shape) > 2:
                        filtered_parents = parent_mask.eq(0).all(dim=3).all(dim=2).all(dim=1).clone().detach()
                    else:
                        filtered_parents = parent_mask.eq(0).all(dim=1).clone().detach()
                        
                    # filtered_indices = torch.nonzero(filtered_parents)
                    # Omit_layer represents the indices of child_mask. I want to set all elements in omit_mask that correspond to weights incoming from the filtered_parents to be 0.
                    if len(omit_layer.shape) > 2:
                        omit_layer[:, filtered_parents, :, :] = 0
                    else:
                        omit_layer[:, filtered_parents] = 0
                    # omit_layer.index_fill_(1, filtered_indices, 0)
                    # Assign the resulting mask to the dictionary
                omit_mask[module_idx] = omit_layer    
            else:
                if module_idx == 13:
                    preskip_mask = all_task_masks[task_num][1].clone().detach()
                elif module_idx == 23:
                    preskip_mask = all_task_masks[task_num][10].clone().detach()
                elif module_idx == 34:
                    preskip_mask = all_task_masks[task_num][20].clone().detach()                    
                elif module_idx == 44:
                    preskip_mask = all_task_masks[task_num][31].clone().detach()                    
                elif module_idx == 55:
                    preskip_mask = all_task_masks[task_num][41].clone().detach()                    
                elif module_idx == 65:
                    preskip_mask = all_task_masks[task_num][52].clone().detach()                    
                elif module_idx == 76:
                    preskip_mask = all_task_masks[task_num][62].clone().detach()                    
                elif module_idx == 86:
                    preskip_mask = all_task_masks[task_num][73].clone().detach()                    

                ### Dont want to overwrite the child, which would break the chain of parent/children relationships in the main network connections
                omit_layer = torch.ones(all_task_masks[task_num][module_idx].clone().detach().size())

                filtered_parents = preskip_mask.eq(0).all(dim=3).all(dim=2).all(dim=1).clone().detach()
                omit_layer[:, filtered_parents, :, :] = 0
                omit_mask[module_idx] = omit_layer                  
                
            iteration += 1

    return omit_mask
    
    
    
    
    
    
    





    
"""
All of the following functions update the various dictionaries to reflect merging of subnetworks
"""
  
def update_tasks_by_subnet(tasks_by_subnet, cluster):

    tasksMerged = []
    for i in range(len(cluster)):
        tasksMerged = tasksMerged + tasks_by_subnet[cluster[i]]
    
    tasks_by_subnet[cluster[-1]] = tasksMerged
    
    ### Use the builtin reverse function to ensure we remove the largest elements first
    for i in reversed(cluster[:-1]):
        finalkey = max(list(tasks_by_subnet.keys()))

        for key in tasks_by_subnet.keys():
            if key >= i and key != finalkey:
                ### Offsetting by 1 assumes subsequent continuous subnet IDs, which we maintain throughout merging
                tasks_by_subnet[key] = tasks_by_subnet[key+1].copy()
        
        del tasks_by_subnet[finalkey]   
        
    return tasks_by_subnet





def update_ID_ACCS(ID_ACCS, cluster, tasks_by_subnet):
    for i in cluster:
        for t in tasks_by_subnet[i]:
            ID_ACCS[t] = []
    return ID_ACCS
        
        
        



def update_ACC_thresholds(ACC_thresholds, cluster, tasks_by_subnet):
    for i in cluster:
        for t in tasks_by_subnet[i]:
            ACC_thresholds[t] = -1
    return ACC_thresholds
        
        
        
        
        
def update_all_task_masks(all_task_masks, cluster):

    ### Get an initial taskmask from the task that will be merged into
    taskmaskMerged = all_task_masks[cluster[-1]].copy()
    
    ### Loop over all subnetworks in the given cluster in order to merge their masks into taskmaskMerge
    for i in cluster[:-1]:
        taskmask_i = all_task_masks[i].copy()
        for key in taskmaskMerged:
            # print("Merging masks for layer: ", key)
            maskA = taskmaskMerged[key]
            maskB = taskmask_i[key]
            
            ### Since we may now be sharing more tasks than either subnetA or B had access to alone, we should reinitialize incoming weights and then omit outgoing weights at the end
    
            if len(maskA.shape) > 2:
                maskA[maskA.eq(1).any(dim=3).any(dim=2).any(dim=1)] = 1
                maskB[maskB.eq(1).any(dim=3).any(dim=2).any(dim=1)] = 1
            else:
                maskA[maskA.eq(1).any(dim=1)] = 1
                maskB[maskB.eq(1).any(dim=1)] = 1
                
            taskmaskMerged[key] = torch.max(maskA, maskB)
            

    all_task_masks[cluster[-1]] = taskmaskMerged.copy()

    ### Loop over all subnetworks in the given cluster in order to merge their masks into taskmaskMerge
    for i in reversed(cluster[:-1]):            
        ### Same as for share_history, we remove subnetworkA from the masking by shifting all subsequent task IDs down by 1 and remove the redundant final task
        finaltask = max(list(all_task_masks.keys()))

        for task in all_task_masks.keys():
            if task >= i and task != finaltask:
                all_task_masks[task] = all_task_masks[task+1].copy()
        
        del all_task_masks[finaltask]
        
    return all_task_masks
    
    
    
    
    
    
    
    
### Removing subnetID from task masks and tasks_by_subnet dictionary
def remove_subnetwork(subnetID, tasks_by_subnet, all_task_masks):

    ### shift all subnet IDs after the one being removed, then remove the final redundant key
    finalkey = max(list(all_task_masks.keys()))

    ### Taking advantage of the fact that tasks_by_subnet and all_task_masks have identical keys since all subnets are used for at least one task
    ### Additionally keys are consecutive integers so we can offset by 1 to get the next key
    for key in tasks_by_subnet.keys():
        if key >= subnetID and key != finalkey:
            all_task_masks[key]  = all_task_masks[key+1].copy()
            tasks_by_subnet[key] = tasks_by_subnet[key+1].copy()
    
    del all_task_masks[finalkey]   
    del tasks_by_subnet[finalkey]   
        
    return tasks_by_subnet, all_task_masks

    
    
    
    
    
    
    







    