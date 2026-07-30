"""
Handles all the pruning and connectivity. Pruning steps are adapted from: https://github.com/arunmallya/packnet/blob/master/src/prune.py
Connectivity steps and implementation of connectivity into the pruning steps are part of our contribution
"""
from __future__ import print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tqdm import tqdm

import collections
import time
import copy
import random
import multiprocessing
import json
import copy
from math import floor
import bisect
import math 

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable
from torch.optim.lr_scheduler  import MultiStepLR
import torchnet as tnt

# Custom imports
from AuxiliaryScripts import network as net
from AuxiliaryScripts import utils
from AuxiliaryScripts import clmodels
from AuxiliaryScripts.utils import activations
from AuxiliaryScripts import DataGenerator as DG

from sklearn.cluster import KMeans


class Manager(object):
    """Performs pruning on the given model."""
    ### Relavent arguments are moved to the manager to explicitly show which arguments are used by it
    def __init__(self, args, checkpoint, num_classes_by_task = None, first_task_classnum=10):
        self.args = args
        
        """
        Note on task IDs: The tasks and buffer are stored sequentially. So if we have an online stream of mini-tasks, then theyre numbered 0:T and this is referred to with set_num
        The subnetworks merge over time, and so for organization we maintain them from 0:S, where S is the number of subnetworks at any given time. Each subnetwork is responsible
            for handling potentially multiple tasks due to the merging and retraining. 
        This necessitates having a 2nd tasknum-equivalent in manager.py that tracks the current subnet 
            for masking while tasknum is used to track the most recent batch that was trained on.
        """        
        ### Which sequential task's data the network is using
        self.set_num = -1
        self.taskid = -1

        ### which subnetwork is being used for the current task data
        self.subnetID = -1
        
        self.retrain_path = None
        self.train_loader = None 
        self.test_loader = None 
        self.trainset_raw = None  
        

        ### For storing activations temporarily when calculating connectivity
        self.acts_dict = {}
        self.labels = None
        self.numclasses = 10
        self.numclasses_by_task = num_classes_by_task
        
        
        """
        Training and Network Dictionaries:
        Tasks-by-Subnetwork: Stores lists of which buffered tasks each subnetwork was used on
        In-Distribution ACC: The Accuracy for ID training data for each task to compare against when assessing ID/OOD distance
        ACC Thresholds: The Accuracy threshold distance for new training data to compare against when assessing ID/OOD distance
        Task Recency: Tracks how many tasks it's been since a given task was encountered. Tasks above a given threshold are thrown out
        Buffered Tasks: The task ID needs to be appended to the data structure in some way when we make a dataloader. Stored as Task ID --> X,Y as tensors
        """

        self.tasks_by_subnet = {}
        self.task_recency = {}
        self.ID_ACCS = {}
        self.ACCthresholds = {}
        self.buffered_tasks = {}
        
        
        """
        Eval Dictionary Descriptions:
        Merge History: For each task where clustering happened, store the resulting clusters
        Online Accs: Store the accuracy on each task during online runtime
        Identified Tasks: For each task, store the task classifier that was identified and used to determine how well the network reidentified old tasks
        New Distributions: What tasks were deemed new tasks (slightly redundant with above, but easier to evaluate with both)
        In-Distribution Accuracy History: A convenient stored history of the ID accuracy of each task over the course of the online training
        Accuracy Threshold History: The threshold of accuracy within which new data is considered ID for the given task, updated each time the task is trained on
        Overall Capacity: At the end of every task store the remaining free capacity per layer by %filters unallocated
        Training Times: Tracks when each set Z_i begins training
        """
        ### Bundle together all of the metrics which will be used to analyze experiment behavior and results
        self.eval_dicts = {}
        self.eval_dicts['mergehistory'] = {}
        self.eval_dicts['onlineaccs'] = {}
        self.eval_dicts['identifiedtasks'] = {}
        self.eval_dicts['newdistributions'] = {}
        self.eval_dicts['ID_ACCS_history'] = {}
        self.eval_dicts['ACCthresholds_history'] = {}
        self.eval_dicts['overall_capacity'] = {}
        self.eval_dicts['training_times'] = {}
        self.eval_dicts['tasks_by_set'] = {}


        ### Dictionary for storing FLOP estimates during runtime
        self.FLOPcounts = {}
        self.FLOPcounts["inference"] = 0
        self.FLOPcounts["training"] = 0
        self.FLOPcounts["retraining"] = 0

        if args.arch == "modresnet18":
            ### The parent and child idxs for the updated relu layers with in_place=False
            self.parent_idxs = [1,1, 7, 10,10,13,17,20,20,23,28,31,31,34,38,41,41,44,49,52,52,55,59,62,62,65,70,73,73,76,80]
            self.child_idxs =  [7,13,10,17,23,17,20,28,34,28,31,38,44,38,41,49,55,49,52,59,65,59,62,70,76,70,73,80,86,80,83]

            ### Relu idxs are either set to the conv layers or the relu layers. Due to skip layers the shared relu after each re-adding of residuals is duplicated
            if self.args.userelu == True:
                self.relu_idxs = [ 3, 9,15,13,19,25,23,30,36,34,40,46,44,51,57,55,61,67,65,72,78,76,82,88,86]
            else:
                self.relu_idxs = [ 1, 7,10,13,17,20,23,28,31,34,38,41,44,49,52,55,59,62,65,70,73,76,80,83,86]

            self.act_idxs =  [ 1, 7,10,13,17,20,23,28,31,34,38,41,44,49,52,55,59,62,65,70,73,76,80,83,86]

            ### Output sizes of activations from each conv2d and linear layer for use in computing FLOPS
            self.outputsizes = {1:1024, 7:256, 10:256, 13:256, 17:256, 20:256, 23:256, 
                                        28:64, 31:64, 34:64, 38:64, 41:64, 44:64, 
                                        49:16, 52:16, 55:16, 59:16, 62:16, 65:16, 
                                        70:4, 73:4, 76:4, 80:4, 83:4, 86:4}
            self.kernelsizes = {1:9, 7:9, 10:9, 13:1, 17:9, 20:9, 23:1, 
                                    28:9, 31:9, 34:1, 38:9, 41:9, 44:1, 
                                    49:9, 52:9, 55:1, 59:9, 62:9, 65:1, 
                                    70:9, 73:9, 76:1, 80:9, 83:9, 86:1}


        elif args.arch == "vgg16":
            self.parent_idxs = [1,4,8,11,15,18,21,25,28,31,35,38,41,47]
            self.child_idxs =  [4,8,11,15,18,21,25,28,31,35,38,41,47,49]
            
            if self.args.userelu == True:
                self.relu_idxs = [3,6,10,13,17,20,23,27,30,33,37,40,43,48,49]
            else:
                self.relu_idxs = [1,4,8,11,15,18,21,25,28,31,35,38,41,47,49]

            self.act_idxs = [1,4,8,11,15,18,21,25,28,31,35,38,41,47,49]
            
            ### FLOPS not calculated for VGG16 in current code
            self.outputsizes = {}
            self.kernelsizes = {}


        if checkpoint != None:
            self.network = checkpoint['network']
            self.all_task_masks = checkpoint['all_task_masks']
        else:
            ### This is for producing and setting the classifier layer for a given task's # classes
            self.network = net.Network(args)
            self.network.add_dataset(0, first_task_classnum)
            self.network.set_dataset(0)
            
            self.all_task_masks = {}
            task_mask = {}

            for module_idx, module in enumerate(self.network.model.shared.modules()):
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    mask = torch.ByteTensor(module.weight.data.size()).fill_(1)
                    mask = mask.cuda()
                    task_mask[module_idx] = mask

            ### Assign a mask for the classifier as well. This is so we can mask away weights pointing to pruned filters

            ### Starts off with a task mask of all weights, so we just need to never prune this or delete it and the full network will be used.
            self.all_task_masks[0] = task_mask

        
     
    """
    ###########################################################################################
    #####
    #####  Connectivity Functions
    #####
    #####  Use: Gets the connectivity between each pair of convolutional or linear layers. 
    #####       The primary original code for our published connectivity-based freezing method
    #####
    ###########################################################################################
    """


    def calc_activations(self, data = None):
        self.network.model.eval()
        ### If no data is specified, get activations on the validation data loader
        if data == None:
            self.acts_dict, self.labels = activations(self.train_loader, self.network.model, self.args.cuda, self.relu_idxs, self.act_idxs)
        ### Otherwise get activations on the data, which is of format data['x':[N,C,W,H]; 'y':[N]]
        else:
            self.acts_dict, self.labels = activations(data, self.network.model, self.args.cuda, self.relu_idxs, self.act_idxs, using_dataloader=False)

        self.network.model.train()


    ### Calculate connectivities for the full network without masking
    def calc_conns(self, data = None):
        all_conns = {}
        self.calc_activations(data = data)
    
        ### The keys for all_conns are labeled as a range because both parent and child idxs have duplicate entries. These keys match to indices in those dictionaries, not module idxs 
        for key_id in range(0,len(self.parent_idxs)): 
            all_conns[key_id] = self.calc_conn(self.parent_idxs[key_id], self.child_idxs[key_id], key_id)
        return all_conns

    ### Calculate the connectivity between a given pair of layers
    def calc_conn(self, parent_key, child_key, key_id):
        p1_op = {}
        c1_op = {}

        p1_op = copy.deepcopy(self.acts_dict[parent_key]) 
        c1_op = copy.deepcopy(self.acts_dict[child_key])

        parent_aves = []
        p1_op = p1_op.numpy()
        c1_op = c1_op.numpy()
        
        for label in list(np.unique(self.labels.numpy())):
            parent_mask = np.ones(p1_op.shape,dtype=bool)
            child_mask = np.ones(c1_op.shape,dtype=bool)

            parent_mask[self.labels != label] = False
            parent_mask[:,np.all(np.abs(p1_op) < 0.0001, axis=0)] = False
            child_mask[self.labels != label] = False
            child_mask[:,np.all(np.abs(c1_op) < 0.0001, axis=0)] = False
            
            p1_op[parent_mask] -= np.mean(p1_op[parent_mask])
            p1_op[parent_mask] /= np.std(p1_op[parent_mask])

            c1_op[child_mask] -= np.mean(c1_op[child_mask])
            c1_op[child_mask] /= np.std(c1_op[child_mask])


        """
        Code for averaging conns by parent prior by layer
        """
        parent_class_aves = []
        parents_by_class = []
        parents_aves = []
        conn_aves = []
        parents = []
        for cl in list(np.unique(self.labels.numpy())):
            p1_class = p1_op[self.labels == cl]
            c1_class = c1_op[self.labels == cl]

            ### Parents is a 2D list of all of the connectivities of parents and children for a single class
            coefs = np.corrcoef(p1_class, c1_class, rowvar=False).astype(np.float32)
            parents = []
            ### Loop over the cross correlation matrix for the rows corresponding to the parent layer's filters
            for i in range(0, len(p1_class[0])):
                ### Append the correlations to all children layer filters for the parent filter i. We're indexing the upper-right quadrant of the correlation matrix between x and y
                #!# Nans: If a parent is omitted, this entire set will be NaN, if a child is omitted, then only the corresponding correlation is nan
                ###    Note: These NaNs are expected and not an issue since they dont appear in the indexed values for the current subnetwork/task
                parents.append(coefs[i, len(p1_class[0]):])
            ### We take the absolute value because we only care about the STRENGTH of the correlation, not the 
            parents = np.abs(np.asarray(parents))

            ### This is a growing list of each p-c connectivity for all activations of a given class
            ###     The dimensions are (class, parent, child)
            parents_by_class.append(parents)
        
        conn_aves = np.mean(np.asarray(parents_by_class), axis=0)
        
        return conn_aves
        

        

       
       
       
               
        
    """
    ##########################################################################################################################################
    Train and Evaluate Functions
    ##########################################################################################################################################
    """
    ### Offline evaluation
    def eval(self):
        """Performs evaluation."""
        # print("Task number in Eval: ", self.subnetID)

        ### Apply the current subnetwork's mask
        self.network.model.eval()

        error_meter = None
        accs = []

        ### This is used for getting ID ACCs after retraining merged networks on the buffered data, so we split the buffer and eval on a subset other than the training data
        ### Otherwise we'll get unrealistically high overfitted accuracies for ACCthresholds due to evaluating on the training data after training on it
        for batch, label in self.test_loader:
            if self.args.cuda:
                batch = batch.cuda()
                label = label.cuda()

            preds = self.network.model(batch)
            preds = torch.argmax(preds, dim=1)


            correct = (label == preds).long().sum()
            total = label.numel()
            
            accs.append(100*(correct/total))  

        accs = torch.tensor(accs)
        print("Eval Accuracy: ", torch.mean(accs))
        return torch.mean(accs), accs
       









    ### Evaluate on the incoming data before training on it for online training
    def online_eval(self, x, y):
        """Performs evaluation."""
        # print("Task number in Eval: ", self.subnetID)

        self.network.model.eval()

        preds = self.network.model(x)
        preds = torch.argmax(preds, dim=1)

        correct = (y == preds).long().sum()
        total = y.numel()

        acc = 100*(correct/total)

        acc = torch.tensor(acc)

        self.network.model.train()
        return acc
       


    ### Train the model online
    def online_train(self, epochs, task, save=True, savename=''):

        subnettasks = self.tasks_by_subnet[self.subnetID]
        test_accuracy_history = []
        merged = False
        if len(subnettasks) > 1:
            merged = True
        print("Online train")
        print("SubnetID: ", self.subnetID)

        ### Replay-only method simply trains the full network on set data mixed with all available buffered tasks in equal amounts
        self.FLOPcounts["training"] += self.get_FLOPS(self.subnetID, epochs=self.args.online_epochs, 
                                                    numsamples=(self.args.setsize*len(subnettasks)), step="train")

        if self.args.cuda:
            self.network.model = self.network.model.cuda()

        for i, batch in enumerate(self.train_loader):
            start_time_batch = time.time()
            xset, yset = batch
            if self.args.cuda:
                xset = xset.cuda()
                yset = yset.cuda()

            # Get optimizer with correct params.
            params_to_optimize = self.network.model.parameters()
            optimizer = optim.SGD(params_to_optimize, lr=self.args.lr, momentum=0.9, weight_decay=0.0, nesterov=True)
            scheduler = MultiStepLR(optimizer, milestones=[floor(epochs/2), floor(2*epochs/3)], gamma=self.args.Gamma)    
            loss = nn.CrossEntropyLoss()
    
            acc = self.online_eval(xset,yset)
            print("Training on batch: ", i, "                 test accuracy: ", acc, flush=True)

            test_accuracy_history.append(acc)
            ### We'll update the ACCthresholds after training since we dont use it until the next set of batches in the subsequent task
            self.ID_ACCS[task].append(acc)

            self.network.model.train()
            
            ### If training a merged subnet we want to do joint training with the necessary past tasks to avoid forgetting. We also switch in a shared classifier head
            if merged == True:
                ### Haven't made a merged classifier yet, need to initialize it and zero the appropriate connections in its mask
                if self.network.classifiermerged == None:
                    self.network.merge_task_classifiers(subnettasks)
                    if self.args.cuda:
                        self.network.model = self.network.model.cuda()
                else:
                    self.network.model.classifier = self.network.classifiermerged
                    if self.args.cuda:
                        self.network.model = self.network.model.cuda()
                        
                ### Prepares a batch with the current task batch mixed together with buffered samples from other tasks belonging to the current subnetwork
                xset, yset = self.prepare_buffered_batch(copy.deepcopy(xset), copy.deepcopy(yset), task)
                xset, yset = xset.cuda(), yset.cuda()
                
            finalstep = math.ceil(xset.size()[0]/512)-1
            print("Final step is:" , finalstep, " and x size: " ,xset.size()[0])
            for step in range(0,math.ceil(xset.size()[0]/512)):
                # print("Step: ", step)
                if step < finalstep:
                    x = xset[step*512:(step+1)*512]
                    y = yset[step*512:(step+1)*512]
                    # print("X from: ", step*512," to ", (step+1)*512)
                else:
                    x = xset[step*512:]
                    y = yset[step*512:]
                    # print("X from: ", step*512," to end")
                # print("x subset size: " ,x.size()[0])
                for idx in range(epochs):
                    epoch_idx = idx + 1
                    # Set grads to 0.
                    self.network.model.zero_grad()
            
                    # Do forward-backward.
                    output = self.network.model(x)
                    loss(output, y).backward()
    
                    # # Set frozen param grads to 0 and update
                    optimizer.step()
                    scheduler.step()


                ### Commit the changes to the weights to the backup composite model
                self.network.update_backup(self.subnetID, self.all_task_masks[self.subnetID])

                
            ### Update all individual classifier weights from training and re-assign the task classifier for evaluating next batch
            if merged == True:
                self.network.split_task_classifiers(subnettasks, deletemerged=False)
                self.network.model.classifier = self.network.classifiers[task]
            

        ### To clean up now that training is finished we reset the merged classifier
        if merged == True:
            self.network.classifiermerged = None
            self.network.merged_mask = None          
            self.store_ID_ACCS(self.subnetID, omittask=task)
            
        ### Recalculate the current task's ID threshold accuracy
        if len(self.ID_ACCS[task]) > self.args.thresholdwindow:
            self.ACCthresholds[task] = torch.min(torch.tensor(self.ID_ACCS[task][-self.args.thresholdwindow:]))
        elif len(self.ID_ACCS[task]) > 2:
            self.ACCthresholds[task] = torch.amin(torch.tensor(self.ID_ACCS[task][1:]))            
        else:
            ### Given insufficient stored values we just set threshold as 95% of the observed accuracy as an initial threshold
            self.ACCthresholds[task] = torch.mean(torch.tensor(self.ID_ACCS[task])) - (0.05 * torch.mean(torch.tensor(self.ID_ACCS[task]))    )

        self.ACCthresholds[task] = torch.max(torch.tensor(15), self.ACCthresholds[task])
        self.eval_dicts['onlineaccs'][self.set_num] = test_accuracy_history    








    """
    ##########################################################################################################################################
    Functions for Determining if Data is In/Out of Distribution of Known Tasks
    ##########################################################################################################################################
    """




    ### Compare Acc for all subnetworks on the current data to determine which subnetwork is most appropriate to use    
    ### Because each classifier relies on a entirely on a single subnetwork, this also identifies the appropriate subnetwork as well as classifier
    def get_acc_difference(self):
    
        ### Collect the final layers activations and then pass them to each classifier one at a time to record the resulting accuracies
        accs = utils.get_all_accs(self.train_loader, self.network.model, self.network.classifiers, self.args.arch, self.numclasses)
        self.FLOPcounts["inference"] += self.get_FLOPS(self.subnetID, epochs=1, numsamples=self.args.batch_size, step="forwardonly")

        maxacc = 0
        maxtask = None

        print("Target thresholds: ", self.ACCthresholds)
        ### Identify which task is predicted for the new set data. Max task is t'
        for task in self.ID_ACCS:
            print("Accuracy Task ", task, ": ", accs[task])
            if accs[task] > maxacc:
                maxacc = accs[task]
                maxtask = task

        ### There is only one subnet
        maxsubnet = 0


        return maxsubnet, maxtask, maxacc
    
    
         
    



    ### Check if the current set data is ID for the predicted maximum-accuracy task t'
    def check_if_existing_dist(self, task):
        if task == 0:
            ### There is no distribution to match with for the first task
            return False, -1, -1

        if self.args.IDscore == 'acc':
            subnetID, maxtask, maxacc = self.get_acc_difference()       
            print("Closest tasks all: ", maxtask, " with accuracy: ", maxacc)
             
            if maxacc >= self.ACCthresholds[maxtask]:
                print("ACC of subnetwork: ", subnetID, " on task ", maxtask, " of ", maxacc," is within the threshold: ", self.ACCthresholds[maxtask])
                return True, subnetID, maxtask
            else:
                print("ACC of subnetwork: ", subnetID, " on task ", maxtask, " of ", maxacc," is outside the threshold: ", self.ACCthresholds[maxtask])
                return False, -1, -1
                                
                
        else:
            return False, -1, -1
       








    """
    ##########################################################################################################################################
    Functions for Updating Dictionaries and Managing Subnetworks
    ##########################################################################################################################################
    """


    ### Set the given task to 0 and increment the number of tasks since last use by 1 for all other tasks
    def update_recency(self, task):
        found = False
        for t in self.task_recency.keys():
            if t == task:
                self.task_recency[t] = 0
                found = True
            else:
                self.task_recency[t] += 1
        
        ### For new tasks, initialize their recency as 0
        if found == False:
            self.task_recency[task] = 0
            
            
            

    ### We look at which tasks haven't been used recently and remove them from the network and buffer
    def remove_unused_tasks(self):
        print("Beginning to remove unused tasks, recency dict: ", self.task_recency)
        removed_tasks = {}
        removing = False
        for s in self.tasks_by_subnet.keys():
            removed_tasks[s] = []
            for t in self.tasks_by_subnet[s]:
                if self.task_recency[t] > self.args.unusedthreshold:
                    removing = True
                    removed_tasks[s].append(t)
        

        ### remove unused tasks from classifiers, classifier masks, ID and threshold dictionaries, tasks_by_subnet, and buffer
        for i in removed_tasks.keys():
            for j in removed_tasks[i]:
                del self.ID_ACCS[j]
                del self.ACCthresholds[j]
                del self.task_recency[j]
                del self.network.classifiers[j]
                del self.network.classifier_masks[j]
                del self.buffered_tasks[j]
                self.tasks_by_subnet[i].remove(j)
        
        ### If any subnet no longer has any allocated tasks, delete it and free its weights for later use
        removed_subnets = []
        for k in self.tasks_by_subnet.keys():
            if len(self.tasks_by_subnet[k]) == 0:
                removed_subnets.append(k)
        
        if removing == True:
            print("\n Removed tasks: ", removed_tasks, " resulting in:")
            # print("ID_ACCS: ", self.ID_ACCS)
            print("ACCthresholds: ", self.ACCthresholds)
            print("Classifiers: ", self.network.classifiers.keys())
            print("Classifier masks: ", self.network.classifier_masks.keys())
            print("buffered tasks: ", self.buffered_tasks.keys())
            print("tasks_by_subnet: ", self.tasks_by_subnet, "\n")
            
        
        
        if len(removed_subnets) > 0:
            print("Removing subnets: ", removed_subnets)
            print("Initial tasksbysubnet ", self.tasks_by_subnet)
            print("Initial all_task_masks ", self.all_task_masks.keys())
        ### Remove any subnetworks that have no tasks. Removing them from largest index to smallest to avoid compounding offsets from shifting subnet IDs
        for s in reversed(removed_subnets):
            for module_idx, module in enumerate(self.network.model.shared.modules()):
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    ### Dont omit anything that isn't in the current task. Mostly needed to add this since it was overwriting other tasks in the backupmodel
                    for module_idx2, module2 in enumerate(self.network.backupmodel.shared.modules()):
                        if module_idx == module_idx2:
                            module.weight.data[self.all_task_masks[s][module_idx].eq(1)] = 0.0                     
                            module2.weight.data[self.all_task_masks[s][module_idx].eq(1)] = 0.0  

            self.tasks_by_subnet, self.all_task_masks = utils.remove_subnetwork(s, self.tasks_by_subnet.copy(), self.all_task_masks.copy())
            print("Intermediate tasksbysubnet ", self.tasks_by_subnet)
            print("Intermediate all_task_masks ", self.all_task_masks.keys())
        
        if removing == True:
            print("Final tasksbysubnet ", self.tasks_by_subnet)
            print("Final all_task_masks ", self.all_task_masks.keys())
        
        

    
    ### Get Acc of current subnetwork on current dataset and score the score for it
    def store_ID_ACCS(self, subnet, omittask=-1):
        print("Storing ID ACCS for the subnet: ", subnet, " and its tasks: ", self.tasks_by_subnet[subnet] )
        print("Buffered tasks: ", self.buffered_tasks.keys())

        ### The same subnetwork is used each time, but we need to do forward passes on all related tasks' samples so effectively we just increase # samples
        self.FLOPcounts["inference"] += self.get_FLOPS(self.subnetID, epochs=1, numsamples=(self.args.buffernum*10*len(self.tasks_by_subnet[subnet])), 
                                                        step="forwardonly")
        
        for task in self.tasks_by_subnet[subnet]:
            if task != omittask:
                ### Sets the testloader to the current task prior to eval. 
                ### Eval will use test_loader which is hardcoded to use batchsize of 32 to produce more sample accs to threshold among
                self.prepare_buffered_dataset([task])
                self.network.set_dataset(task)
                ### The self.subnetID should already have been set to the correct subnetwork from retraining
                meanacc, accs = self.eval()
                print("Post-Retrain ID Accs for task: ", task, " are: ", accs)
                ### Because the last batch may often be composed of only a few samples it can lead to highly variable accuracies, so we just omit it for stability
                self.ID_ACCS[task] = [accs[:-1]]
                self.ACCthresholds[task] = torch.amin(accs[:-1])
                self.ACCthresholds[task] = torch.max(torch.tensor(15), self.ACCthresholds[task])

    
    
    """
    ##########################################################################################################################################
    Functions for Clustering and Merging Subnetworks
    ##########################################################################################################################################
    """
     


    ### Either prepare all tasks for the current subnet, or a given set of tasks if a list of IDs is passed
    def prepare_buffered_batch(self,  x, y, task):
        ### Get the tasks assigned for subnetB
        tasks = self.tasks_by_subnet[self.subnetID]
        
        print("Preparing dataset from buffered task data for tasks: ", tasks)
        
        batchsize = y.size()[0]
        
        allx = None
        ally = None
        
        cumulative_class_total = 0
        ### We wont offset the first task, but offset each subsequent task to have non-overlapping labels for joint training
        for t in tasks:
            if t != task:
                taskx = copy.deepcopy(self.buffered_tasks[t]['x'])
                tasky = copy.deepcopy(self.buffered_tasks[t]['y'])
                taskz = copy.deepcopy(self.buffered_tasks[t]['z'])
                numclasses = self.numclasses_by_task[taskz[0]]
                # print("numclasses for task: ", t," is ", numclasses)
                ### For this simple setting we allocate a number of samples from each buffer equal to the training batch size
                ### For more unbalanced tasks it might be necessary to instead allocate # per class, such as for tiny imagenet or cifar100
                perm = torch.randperm(tasky.size()[0])
                taskx = taskx[perm][:batchsize]
                tasky = tasky[perm][:batchsize]
            else:
                taskx = x
                tasky = y
                numclasses = self.numclasses_by_task[task]
            ### Offset labels for all tasks after the first, before updating the cumulative count
            tasky += cumulative_class_total

            
            ### Get the number of classes and add it to the total for offsetting future task labels
            cumulative_class_total += numclasses
            
            if allx == None:
                allx = taskx
                ally = tasky
            else:
                allx = torch.cat((allx.cuda(),taskx.cuda()),dim=0)
                ally = torch.cat((ally.cuda(),tasky.cuda()),dim=0)


        ###Manually shuffle the data just in case
        indices = torch.randperm(ally.size()[0])
        allx = allx[indices]
        ally = ally[indices]
        
        ### Since we're setting new dataloaders for consistency we should also update the number of classes for the current "task"
        self.numclasses = cumulative_class_total
        
        return allx,ally







    ### Either prepare all tasks for the current subnet, or a given set of tasks if a list of IDs is passed
    def prepare_buffered_dataset(self, tasks=None, batchsize = None):
        ### Get the tasks assigned for subnetB
        if tasks == None:
            tasks = self.tasks_by_subnet[self.subnetID]
        
        print("Preparing dataset from buffered task data for tasks: ", tasks)
        for s in ['train','test']:
            dataset = {}
            dataset['x'] = None
            dataset['y'] = None
            
            cumulative_class_total = 0
            for t in tasks:
                taskx = copy.deepcopy(self.buffered_tasks[t]['x'])
                tasky = copy.deepcopy(self.buffered_tasks[t]['y'])
                taskz = copy.deepcopy(self.buffered_tasks[t]['z'])   

                ### We assume that the data is not significantly ordered by class here, otherwise this split isn't random
                ### If we shuffle prior to splitting however then the test set is inconsistent between retraining and recalculating ID Accs

                split = int(math.floor(0.7 * list(taskx.size())[0]))
                # print("Splitting for buffered dataset: ", split, " of ", taskx.size()[0], " samples")
                if s == 'train':
                    taskx = taskx[:split]
                    tasky = tasky[:split]
                    taskz = taskz[:split]
                else:
                    taskx = taskx[split:]
                    tasky = tasky[split:]
                    taskz = taskz[split:]
                    
                ### Offset labels for all tasks after the first, before updating the cumulative count
                if t > 0:
                    tasky += cumulative_class_total

                ### Get the number of classes and add it to the total for offsetting future task labels
                numclasses = self.numclasses_by_task[taskz[0]]
                cumulative_class_total += numclasses
                
                if dataset['x'] == None:
                    dataset['x'] = taskx
                    dataset['y'] = tasky
                else:
                    dataset['x'] = torch.cat((dataset['x'],taskx),dim=0)
                    dataset['y'] = torch.cat((dataset['y'],tasky),dim=0)
            
            ###Manually shuffle the data just in case
            indices = torch.randperm(dataset['y'].size()[0])
            dataset['x']=dataset['x'][indices]
            dataset['y']=dataset['y'][indices]
            
            ### Since we're setting new dataloaders for consistency we should also update the number of classes for the current "task"
            self.numclasses = cumulative_class_total
            
            ### Makes a custom dataset for CIFAR through torch
            generator = DG.CifarDataGenerator(dataset['x'],dataset['y'])
    
            if s == "train":
                if batchsize == None:
                    self.train_loader = data.DataLoader(generator, batch_size = self.args.offline_batch_size, shuffle = True, num_workers = 4, pin_memory=self.args.cuda)
                else:
                    self.train_loader = data.DataLoader(generator, batch_size = batchsize, shuffle = True, num_workers = 4, pin_memory=self.args.cuda)
            else:
                if batchsize == None:
                    self.test_loader =  data.DataLoader(generator, batch_size = 32, shuffle = False, num_workers = 4, pin_memory=self.args.cuda)
                else:
                    self.test_loader =  data.DataLoader(generator, batch_size = 32, shuffle = False, num_workers = 4, pin_memory=self.args.cuda)
                    
        
        
        
        




    # """
    # The following functions are for approximating FLOPS for a given experiment
    # """
    ### Linear: 2 * I * O
    """
    subnetID: Subnetwork being used, for getting number of channels when computing FLOPS and total network size
    epochs: How many epochs training is being performed for
    numsamples: The number of samples being used for the operation. The number of samples needs to be calculated prior to calling this function
                    based on the number of buffered tasks, the batch size, the number of batches being calculated for, etc.
    step: Which step is being performed, among "train", "forwardonly" 
    Note: For backprop operations we use the approximate ratio of 2:1 back:forward FLOPS per layer
            Also for simplicity we omit the FLOPs from the classifiers. All classifiers are 10-label for experiments, with the final layer being 512 neurons
                as such these costs are expected to be fairly negligible overall
    """
    def get_FLOPS(self, subnetID, epochs=1, numsamples=5000, step="train"):
        
        ### Get the current task's subnetID. This gives information about both the size of the network and the size of the subnetwork
        taskmask = self.all_task_masks[subnetID]
        FLOPS = 0

        ### Computing on ResNet18 only, with all layers except the Classifier being Conv2d
        ### Conv2D FLOPS: output_image_size * kernel shape * output_channels

        ### All computations occur over the full network, so to be concise we loop over the modules in the outermost loop
        for key in taskmask.keys():
            ### Get the number of included filters and the number of total filters in the current subnetwork and overall layer
            numfilters = taskmask[key].eq(1).all(dim=3).all(dim=2).any(dim=1).long().sum()
            totalfilters = taskmask[key].eq(1).all(dim=3).all(dim=2).any(dim=1).long().numel()
            outputsize = self.outputsizes[key]
            kernelsize = self.kernelsizes[key]

            ### Just do forward task
            if step == "forwardonly":
                ### Forward pass on all samples with the current subnetwork only
                FLOPS += (outputsize * kernelsize * totalfilters * numsamples)

            ### Train the network for some number of batches and epochs        
            elif step == "train":
                ### For a 2:1 back:forward ratio of FLOPs training takes 3 times as many flops as a forward pass, and is done for some number of epochs on all samples
                FLOPS += 3 * (outputsize * kernelsize * totalfilters * numsamples * epochs)
            
        return FLOPS