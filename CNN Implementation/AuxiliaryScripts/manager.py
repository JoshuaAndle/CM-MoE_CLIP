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
from torch.optim.lr_scheduler  import CosineAnnealingLR
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
        ### set_num reflects the current set number while taskid reflects which task distribution it belongs to
        self.set_num = -1
        self.taskid = -1

        ### which subnetwork is being used for the current task data
        self.subnetID = -1
        
        self.retrain_path = None
        
        self.train_loader = None 
        self.test_loader = None 

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
        self.eval_dicts = {
        'mergehistory': {},
        'onlineaccs': {},
        'identifiedtasks': {},
        'newdistributions': {},
        'ID_ACCS_history': {},
        'ACCthresholds_history': {},
        'overall_capacity': {},
        'training_times': {},
        'tasks_by_set': {}
        }

        ### Dictionary for storing FLOP estimates during runtime
        self.FLOPcounts = {
        "inference": 0,
        "training": 0,
        "retraining": 0
        }
        
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




            # 1, 7, 10, 13, 17, 20, 23, 28, 31, 34, 38, 41, 44, 49, 52, 55, 59, 62, 65, 70, 73, 76, 80, 83, 86
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


            ### FLOPS only calculated for resnet18 in current code
            self.outputsizes = {}
            self.kernelsizes = {}
        
        

        ### Either load from a checkpoint or initialize the necessary masks and network
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
        
        
        ### Connectivity is standardized by class mean and stdev
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
        
   

    ### Omits from the task mask any weights which were used in other tasks
    def zero_new_mask(self):
        print("Current tasknum: ", self.subnetID)

        for module_idx, module in enumerate(self.network.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                new_weights = utils.get_trainable_mask(module_idx, self.all_task_masks, self.subnetID)
                
                ### Identify frozen filters and prevent new weights being connected to them
                frozen_filters = utils.get_frozen_mask(module.weight.data, module_idx, self.all_task_masks, self.subnetID)

                ### Any filter with a weight allocated to an existing subnetwork/task needs to be omitted from the current subnetwork to maintain structured sparsity
                if len(frozen_filters.shape) > 2:
                    frozen_filters = frozen_filters.eq(1).any(dim=3).any(dim=2).any(dim=1)
                else:
                    frozen_filters = frozen_filters.eq(1).any(dim=1)
                    
                new_weights[frozen_filters] = 0
                
                self.all_task_masks[self.subnetID][module_idx] = new_weights
                
        ### Account for any outgoing weights from omitted filters
        ### Only affects the current subnetwork, delisting these hanging weights and updating the taskmask and backup model
        self.zero_outgoing_omitted()


        
    """
    ##########################################################################################################################################
    Pruning and Initialization Functions
    ##########################################################################################################################################
    """
    ### Goes through and calls prune_mask for each layer and stores the results
    ### Then applies the masks to the weights
    def prune(self, merged = False):
        print('Pruning for dataset idx: ',self.task_num, ' for each layer by removing ', (100 * self.args.prune_perc_per_layer),' of values')
        one_dim_prune_mask = 0
        four_dim_prune_mask = 0
        
        for module_idx, module in enumerate(self.network.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                ### If this isn't a skip layer, prune as normal
                if module_idx not in [13,23,34,44,55,65,76,86] or self.args.arch != "modresnet18":

                    ### Get the pruned mask for the current layer
                    pruned_mask = self.pruning_mask(module.weight.data.clone().detach(), module_idx, reuse_frozen=False, merged = merged)

                    for module_idx2, module2 in enumerate(self.network.backupmodel.shared.modules()):
                        if module_idx == module_idx2:
                            module2.weight.data[pruned_mask.eq(1)] = 0.0
                        
                    ### Set pruned weights to 0.
                    module.weight.data[pruned_mask.eq(1)] = 0.0
                    self.all_task_masks[self.subnetID][module_idx][pruned_mask.eq(1)] = 0

                    ### Store the prune mask to make sure its reused for the appropriate skip junction which will be readded to the network, such that pruned filters match
                    if module_idx in [10,20,31,41,52,62,73,83] and self.args.arch == "modresnet18":
                        one_dim_prune_mask =  torch.amax(pruned_mask, dim=(1,2,3))
                    
        
                ### For skip layers, reuse the prune mask from the layer that they'll be added together with, expanded to the appropriate weight shape
                ### This is done to avoid re-adding residuals into frozen feature maps, which would cause feature drift
                else:
                    four_dim_prune_mask = one_dim_prune_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(module.weight.data.shape)
                    module.weight.data[four_dim_prune_mask.eq(1)] = 0.0
                    self.all_task_masks[self.subnetID][module_idx][four_dim_prune_mask.eq(1)] = 0
                    for module_idx2, module2 in enumerate(self.network.backupmodel.shared.modules()):
                        if module_idx == module_idx2:
                            module2.weight.data[four_dim_prune_mask.eq(1)] = 0.0
                                    
        ### Once all pruning is done, we need to make sure that the weights going out of newly pruned filters are zeroed to avoid future issues when sharing subnetworks
        self.zero_outgoing_omitted()  

 
    ### Note that this pruning is seemingly overly complex, and many of the steps are done in order to ensure that the weight magnitudes that we
    ###   consider when pruning are invarient to the number of filters in the previous layer which were omitted. Since the newer task will always have more incoming
    ###   non-omitted weights than any shared past tasks, we want to avoid this biasing the l2 norm of the filters, and so we only consider the non-omitted incoming weight values 
    def pruning_mask(self, weights, layer_idx, reuse_frozen=False, merged=False):
        """
            Ranks prunable filters by magnitude. Sets all below kth to 0.
            Returns pruned mask.

            weight_magnitudes: 2D weight magnitudes
            trainable_mask: 2D boolean mask, True for trainable weights
            task_mask: 2D boolean mask, 1 for all weights included in current task (not omitted incoming weights)
        """

        if len(weights.size()) > 2 and self.args.prune_method == "structured":
            weight_magnitudes = torch.mean(weights.abs(), axis=(2,3))
            task_mask = self.all_task_masks[self.subnetID][layer_idx].eq(1).any(dim=3).any(dim=2)
        else:
            weight_magnitudes = weights.abs()
            task_mask = self.all_task_masks[self.subnetID][layer_idx].eq(1)

        ### Sum all incoming weights for each filter. We can't directly average them as that unfairly biases filters with different number of omitted incoming weights
        weights_sum = weight_magnitudes.sum(dim=1)

        ### Calculate the number of incoming weights that haven't been omitted for each filter prior to averaging
        if reuse_frozen == False:
            weights_num = task_mask.long().sum(dim=1)
            ### This is the average weight values for ALL filters in current layer (not counting omitted incoming weights)
            current_task_weight_averages = torch.where(weights_num.gt(0), weights_sum/weights_num, weights_sum)
            # current_task_weight_averages = weights_simple_mean
            
            ### This is done to further mask out any frozen filters, since we want to set a pruning threshold based on new features
            included_weights = current_task_weight_averages[task_mask.any(dim=1)]

        ### Same idea, but we want to incldue shared, frozen filters in the threshold calculation
        ###    This way if all of the frozen weights are of higher magnitude, then we may prune a larger portion of the newly trained features
        else:
            weights_num = task_mask.long().sum(dim=1)
            current_task_weight_averages = torch.where(weights_num.gt(0), weights_sum/weights_num, weights_sum)
            ### This will just exlude all omitted features from the threshhold calculation
            included_weights = current_task_weight_averages[task_mask.any(dim=1)]

        if merged == False:
            if self.args.sparsity_type == "unfrozen_only":
                ### Now we use our masked set of averaged 1D feature weights to get a pruning threshold
                prune_rank = round(self.args.prune_perc_per_layer * included_weights.size(dim=0))
            else:
                ### Determine number of filters to prune based on overall layer width, uniform regardless of task number/order
                prune_rank = round(self.args.prune_perc_per_layer * task_mask.size(dim=0))
        else:
            if self.args.sparsity_type == "unfrozen_only":
                ### Now we use our masked set of averaged 1D feature weights to get a pruning threshold
                prune_rank = round(self.args.merged_prune_perc_per_layer * included_weights.size(dim=0))
            else:
                ### Determine number of filters to prune based on overall layer width, uniform regardless of task number/order
                prune_rank = round(self.args.merged_prune_perc_per_layer * task_mask.size(dim=0))
            

        prune_value = included_weights.view(-1).cpu().kthvalue(prune_rank)[0]

        ### Now that we have the pruning threshold, we need to get a mask of all filters who's average incoming weights fall below it        
        weights_to_prune = current_task_weight_averages.le(prune_value)

        prune_mask = torch.zeros(weights.shape)
        ### The frozen mask has 1's indicating frozen weight indices
        if len(weights.size()) > 2 and self.args.prune_method == "structured":
            expanded_prune_mask = weights_to_prune.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(prune_mask.shape)
            prune_mask[expanded_prune_mask]=1        
        else:
            expanded_prune_mask = weights_to_prune.unsqueeze(-1).expand(prune_mask.shape)
            prune_mask[expanded_prune_mask]=1        
                        
        ### Prevent pruning of any non-trainable weights (frozen or omitted)
        prune_mask[task_mask.eq(0)]=0


        return prune_mask
        
        
        
               
        
    """
    ##########################################################################################################################################
    Train and Evaluate Functions
    ##########################################################################################################################################
    """
    ### Offline evaluation
    def eval(self):
        """Performs evaluation."""

        ### Apply the current subnetwork's mask
        self.network.apply_mask(self.all_task_masks, self.subnetID)

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
       


    ### Train the current subnetwork for the current task
    def train_offline(self, epochs, task, save=True, savename='', best_accuracy=0, mergedclassifier=False):
        best_model_acc = best_accuracy
        test_accuracy_history = []

        if self.args.cuda:
            self.network.model = self.network.model.cuda()

        # Get optimizer with correct params.
        params_to_optimize = self.network.model.parameters()
        optimizer = optim.SGD(params_to_optimize, lr=self.args.lr, momentum=0.9, weight_decay=0.0, nesterov=True)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)    

        loss = nn.CrossEntropyLoss()

        self.network.model.train()


        for idx in range(epochs):
            epoch_idx = idx + 1
            for x, y in self.buffer_loader:
                if self.args.cuda:
                    x = x.cuda()
                    y = y.cuda()
                    
                    
                # Set grads to 0.
                self.network.model.zero_grad()
        
                # Do forward-backward.
                output = self.network.model(x)
                loss(output, y).backward()

                # Set frozen param grads to 0.
                self.network.make_grads_zero(self.all_task_masks, self.subnetID, task, mergedclassifier)

                # Update params.
                optimizer.step()

            scheduler.step()
            


        ### Commit the changes to the weights to the backup composite model
        self.network.update_backup(self.subnetID, self.all_task_masks[self.subnetID])
        















    ### Evaluate on the incoming data before training on it for online training
    def online_eval(self, x, y):

        print("Classifier size eval: ", self.network.model.classifier.weight.data.size())
        self.network.model.eval()

        preds = self.network.model(x)
        preds = torch.argmax(preds, dim=1)

        correct = (y == preds).long().sum()
        total = y.numel()

        acc = 100*(correct/total)

        acc = torch.tensor(acc)
        # print("Eval Accuracy: ", acc)
        self.network.model.train()
        return acc
       


    ### Do online training for model on data stream
    def online_train(self, epochs, task, save=True, savename='', new_subnet=False, frozen=False, storebuffer=False):
        """
        epochs:   How many epochs to train each batch for.
        task:   The task being trained on.
        save, savename:   Whether to save best models, and where to save to. Currently unimplemented.
        new_subnet:   Denotes if this is a new subnetwork. Used as a flag to trigger pruning after some batches.
        frozen:   If the subnetwork is frozen, we just do inference on incoming data and then update in-distribution accuracies.
        storebuffer:   Dictates whether or not to store buffer data for newly encountered tasks.
        """


        subnettasks = self.tasks_by_subnet[self.subnetID]
        test_accuracy_history = []
        merged = False

        print("Online train - SubnetID: {} - Subnet Tasks: {}".format(self.subnetID, self.tasks_by_subnet))

        self.network.apply_mask(self.all_task_masks, self.subnetID)
        
        ### If current subnetwork is responsible for multiple tasks, it's been merged and those tasks should be jointly trained to mitigate forgetting
        if len(subnettasks) > 1:
            merged = True
            ### We merge the individual task classifiers temporarily during joint training, then split them up at the end
            self.prepare_buffered_dataset()
            buffer_iter = iter(self.buffer_loader)
            self.network.merge_task_classifiers(subnettasks)
            self.zero_outgoing_omitted()

            if frozen == False:
                self.FLOPcounts["training"] += self.get_FLOPS(self.subnetID, epochs=self.args.online_epochs, numsamples=(self.args.setsize*len(subnettasks)), step="train")

        else:
            self.network.set_dataset(task)
            
            if frozen == False:
                ### FLOPs for the training prior to pruning the network
                if new_subnet == True:
                    self.FLOPcounts["training"] += self.get_FLOPS(self.subnetID, epochs=self.args.online_epochs, numsamples=(self.args.batch_size * (self.args.prune_batch+1)), step="train")
                else:
                    self.FLOPcounts["training"] += self.get_FLOPS(self.subnetID, epochs=self.args.online_epochs, numsamples=self.args.setsize, step="train")
                    
    
        self.FLOPcounts["inference"] += self.get_FLOPS(self.subnetID, epochs=1, numsamples=self.args.setsize, step="forwardonly", unmasked=False)




        ##############################################
        ###  Begin training on batches in data stream
        ##############################################

        # Get optimizer with correct params.
        params_to_optimize = self.network.model.parameters()
        optimizer = optim.SGD(params_to_optimize, lr=self.args.lr, momentum=0.9, weight_decay=0.0, nesterov=True)
        loss = nn.CrossEntropyLoss()

        for i, batch in enumerate(self.train_loader):
            x, y = batch
            if self.args.cuda:
                x, y = x.cuda(), y.cuda()
                
            ### We set to the single-task classifier to evaluate on the samples, then reload the merged classifier if needed
            #!# Would be better to just mask the output logits, will change it when I can find the time
            self.network.set_dataset(task)

            for param_group in optimizer.param_groups:
                param_group['lr'] = initial_lr
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs)    

            acc = self.online_eval(x,y)
            print("Training on batch: ", i, " ----------------- test accuracy: ", acc, flush=True)

            ### We'll update the ACCthresholds after training since we dont use it until the next set of batches in the subsequent task
            test_accuracy_history.append(acc)
            self.ID_ACCS[task].append(acc)


            ############################################
            ###  Train current batch for some few epochs
            ############################################

            ### Skip training if the subnetwork is frozen. Used in prune-only experiments
            if frozen == False:
                for idx in range(epochs):
                    epoch_idx = idx + 1
                    # Set grads to 0.
                    self.network.model.zero_grad()
            
                    # Do forward-backward.
                    output = self.network.model(x)
                    loss(output, y).backward()
    

                    ### If training a merged subnet we accumulate gradients with a batch of replay data before updating weights 
                    if merged == True:
                        self.network.set_dataset("merged")
                        try:
                            mem_batch = next(buffer_iter)
                        except StopIteration:
                            buffer_iter = iter(self.buffer_loader)
                            mem_batch = next(buffer_iter)
                                
                        ### Prepares a batch with the current task batch mixed together with buffered samples from other tasks belonging to the current subnetwork
                        x_mem, y_mem = mem_batch
                        if self.args.cuda:
                            x_mem, y_mem = x_mem.cuda(), y_mem.cuda()

                        # Set grads to 0.
                        self.network.model.zero_grad()
                
                        # Do forward-backward.
                        output = self.network.model(xbatch)
                        loss(output, ybatch).backward()
            
                    self.network.make_grads_zero(self.all_task_masks, self.subnetID, task, merged)
                    optimizer.step()
                    scheduler.step()


                #########################################
                ###  After epochs for current batch
                #########################################

                ### Store samples for unmerged subnetworks up to the per-class limit of args.buffernum
                if storebuffer == True:
                    self.buffered_tasks[task] = utils.add_samples_to_buffer(copy.deepcopy(x).cpu(),copy.deepcopy(y).cpu(),self.buffered_tasks[task],
                                                                            self.numclasses, task, self.args.buffernum)
                        
                ### Commit the changes to the weights to the backup composite model
                self.network.update_backup(self.subnetID, self.all_task_masks[self.subnetID])
    
                
                ### After a certain number of initial training batches we prune the network before continuing and deactivate the pruning flag  
                if new_subnet == True and i == self.args.prune_batch:
                    self.prune(merged = False)
                    
                    ### FLOPs for the retraining after pruning, only using the current tasks' buffered data
                    self.FLOPcounts["retraining"] += self.get_FLOPS(self.subnetID, epochs=self.args.finetune_epochs, 
                                                                        numsamples=(self.buffered_tasks[task]['x'].size()[0]), step="train")

                    ### FLOPs for the the remainder of online training after pruning to reflect the new sparsity of the subnetwork
                    self.FLOPcounts["training"] += self.get_FLOPS(self.subnetID, epochs=self.args.online_epochs, 
                                                                    numsamples=(self.args.setsize - (self.args.batch_size * (self.args.prune_batch+1))), step="train")

                    if self.args.retrainpruned == True:
                        ### Finetune on buffered samples from only the current task
                        self.prepare_buffered_dataset(tasks=[task])
                        self.train_offline(self.args.finetune_epochs, task)
                    new_subnet = False
                    
                ### Update all individual classifier weights from training and re-assign the task classifier for evaluating next batch
                if merged == True:
                    self.network.split_task_classifiers(subnettasks, deletemerged=False)
        
        
        ###########################
        ###  After training
        ###########################

        self.network.set_dataset(task)

        ### To clean up now that training is finished we reset the merged classifier
        if merged == True:
            self.network.classifiermerged = None
            self.network.merged_mask = None  
            
        ### If the number of batches in the task was insufficient to trigger pruning we prune at the end of training. 
        #!# We include this for clarity, but this doesn't happen in the experiment settings and should always be followed by additional training steps  
        if new_subnet == True:
            self.prune(merged = False)
        if merged == True:
            self.store_ID_ACCS(self.subnetID, omittask=task)
            self.network.model.classifier = self.network.classifiers[task]

        ### Recalculate the current task's ID threshold accuracy
        if len(self.ID_ACCS[task]) > self.args.thresholdwindow:
            self.ACCthresholds[task] = torch.min(torch.tensor(self.ID_ACCS[task][-self.args.thresholdwindow:]))
        elif len(self.ID_ACCS[task]) > 2:
            self.ACCthresholds[task] = torch.amin(torch.tensor(self.ID_ACCS[task][1:]))            
        else:
            ### Given insufficient stored values we just set threshold as 95% of the observed accuracy as an initial threshold
            self.ACCthresholds[task] = torch.mean(torch.tensor(self.ID_ACCS[task])) - (0.05 * torch.mean(torch.tensor(self.ID_ACCS[task])))

        self.ACCthresholds[task] = torch.max(torch.tensor(15), self.ACCthresholds[task])
        self.eval_dicts['onlineaccs'][self.set_num] = test_accuracy_history    



















        
        

    ### Network needs to be unmasked prior to this being called, or have the current subnetworks mask applied. 
    ### Correct classifier also needs to be applied to network
    ### Zero weights outgoing from omitted filters in order to avoid drift of frozen filters in the next layer
    def zero_outgoing_omitted(self, zeroshared = True):
        if zeroshared == True:
            omitmask = utils.get_omitted_outgoing_mask(self.all_task_masks, self.subnetID, self.network, self.args.arch)
    
            for module_idx, module in enumerate(self.network.model.shared.modules()):
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    ### Dont omit anything that isn't in the current task. Mostly needed to add this since it was overwriting other tasks in the backupmodel
                    omitmask[module_idx][self.all_task_masks[self.subnetID][module_idx].eq(0)] = 1
                    
                    module.weight.data[omitmask[module_idx].eq(0)] = 0.0
                    self.all_task_masks[self.subnetID][module_idx][omitmask[module_idx].eq(0)] = 0
                        
                    ### Reflect the changes in the backup model before removing weights from the task mask
                    for module_idx2, module2 in enumerate(self.network.backupmodel.shared.modules()):
                        if module_idx == module_idx2:
                            module2.weight.data[omitmask[module_idx].eq(0)] = 0.0                
                
        
        ### Manually mask the classifier layer accordingly to ensure it doesn't rely on any pruned filters
        if self.args.arch == "modresnet18":
            final_omitted = self.all_task_masks[self.subnetID][83]
            final_omitted_flattened = final_omitted.eq(0).all(dim=3).all(dim=2).all(dim=1).flatten()
        elif self.args.arch == "vgg16":
            final_omitted = self.all_task_masks[self.subnetID][49]
            final_omitted_flattened = final_omitted.eq(0).all(dim=1).flatten()

        ### Get all filters that are omitted for the current task and flatten the mask for use with the subsequent classifier layer
        self.network.model.classifier.weight.data[:,final_omitted_flattened] = 0.0
        
        if self.network.merged_mask != None:
            self.network.merged_mask[:,final_omitted_flattened] = 0
        if self.network.classifiermerged != None:
            self.network.classifiermerged.weight.data[:,final_omitted_flattened] = 0.0

        for task in self.tasks_by_subnet[self.subnetID]:
            # print("Zeroing outgoing for classifier of task: ", task)
            self.network.classifiers[task].weight.data[:,final_omitted_flattened] = 0.0
            self.network.classifier_masks[task][:,final_omitted_flattened] = 0


        
        
        
        


    ### Reset the weights of the initialization model and reinitialized the previously pruned weights
    def reinit_statedict(self):
        if self.args.arch == "modresnet18":
            self.network.initmodel = clmodels.modifiedresnet18()
        for module_idx, module in enumerate(self.network.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                new_weights = utils.get_trainable_mask(module_idx, self.all_task_masks, self.subnetID)
                ### Get the frozen filters and use to mask out any previously omitted weights which appear as trainable                
                frozen_filters = utils.get_frozen_mask(module.weight.data, module_idx, self.all_task_masks, self.subnetID)
                if len(frozen_filters.shape) > 2:
                    frozen_filters = frozen_filters.eq(1).any(dim=3).any(dim=2).any(dim=1)
                else:
                    frozen_filters = frozen_filters.eq(1).any(dim=1)
                new_weights[frozen_filters] = 0
                for module_idx2, module2 in enumerate(self.network.initialmodel.shared.modules()):
                    if module_idx2 == module_idx:
                        module.weight.data[new_weights.eq(1)] = module2.weight.data.clone()[new_weights.eq(1)]


       
       
       

    ### Make a new task mask for a created subnetwork
    def make_taskmask(self):
        ### Creates the task-specific mask during the initial weight allocation
        task_mask = {}
        for module_idx, module in enumerate(self.network.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                task = torch.ByteTensor(module.weight.data.size()).fill_(1)
                task = task.cuda()
                task_mask[module_idx] = task

        ### Initialize the new tasks' inclusion map with all 1's
        self.all_task_masks[self.subnetID] = task_mask

       
       





















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
        self.FLOPcounts["inference"] += self.get_FLOPS(self.subnetID, epochs=1, numsamples=self.args.batch_size, step="forwardonly", unmasked=True)


        maxacc = -1
        maxtask = None

        print("Target thresholds: ", self.ACCthresholds)
        
        ### Identify which task is predicted for the new set data. Max task is t'
        for task in self.ID_ACCS:
            print("Accuracy Task ", task, ": ", accs[task])
            if accs[task] > maxacc:
                maxacc = accs[task]
                maxtask = task

    
        maxsubnet = None
        for key in self.tasks_by_subnet.keys():
            if maxtask in self.tasks_by_subnet[key]:
                maxsubnet = key



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
        
        
        
        
    ### Get remaining free capacity in network
    def get_overall_capacity(self):
        overall_capacity = {}
        tasks = list(self.all_task_masks.keys())
        
        for layer in self.all_task_masks[0]:
            ### Get the composite mask of all tasks
            maskl = copy.deepcopy(self.all_task_masks[0][layer])
            for t in tasks[1:]:
                maskl = torch.max(maskl, self.all_task_masks[t][layer])
            
            if len(maskl.size()) > 2:
                ### For filters get any filters with allocated weights
                maskl = maskl.amax(dim=(1,2,3))
            else:
                ### For linear layers just get neurons with any included weights
                maskl = maskl.amax(dim=1)
            
            capacity = maskl.sum()/maskl.numel()
            overall_capacity[layer] = capacity
        self.eval_dicts['overall_capacity'][self.set_num] = overall_capacity
            
            
    
        

    ### Get Acc of current subnetwork on current dataset and score the score for it
    def store_ID_ACCS(self, subnet, omittask=-1):
        print("Storing ID ACCS for the subnet: ", subnet, " and its tasks: ", self.tasks_by_subnet[subnet] )
        print("Buffered tasks: ", self.buffered_tasks.keys())

        ### The same subnetwork is used each time, but we need to do forward passes on all related tasks' samples so effectively we just increase # samples
        self.FLOPcounts["inference"] += self.get_FLOPS(self.subnetID, epochs=1, numsamples=(self.args.buffernum*10*len(self.tasks_by_subnet[subnet])), 
                                                        step="forwardonly", unmasked=False)
        
        for task in self.tasks_by_subnet[subnet]:
            if task != omittask:
                ### Sets the testloader to the given task prior to eval. 
                ### Eval will use test_loader which is hardcoded to use batchsize of 32 to produce more sample accs to threshold among
                self.prepare_test_dataset(task)
                self.test_loader = get_dataloader(self.args.dataset, batch_size=32, set_num=task, set="test"):

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
     

    ### Prepare and set a new self.test_dataloader for a specified task
    def prepare_test_dataset(self, task, batchsize = None):
        
        
        generator = DG.CifarDataGenerator(dataset['x'],dataset['y'])

        self.test_loader =  data.DataLoader(generator, batch_size = 32, shuffle = False, num_workers = 4, pin_memory=self.args.cuda)
            


    ### Prepare and set a new self.train_dataloader using buffer data for all tasks belonging to the current subnetwork
    def prepare_buffered_dataset(self, tasks=None, batchsize = None):
        ### Get the tasks assigned for subnetB
        if tasks == None:
            tasks = self.tasks_by_subnet[self.subnetID]
        
        print("Preparing dataset from buffered task data for tasks: ", tasks)
        dataset = {'x': None, 'y': None}
        
        cumulative_class_total = 0
        for t in tasks:
            taskx = copy.deepcopy(self.buffered_tasks[t]['x'])
            tasky = copy.deepcopy(self.buffered_tasks[t]['y'])
            taskz = copy.deepcopy(self.buffered_tasks[t]['z'])   

                
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
        indices = torch.randperm(dataset['y'].size(0))
        dataset['x']=dataset['x'][indices]
        dataset['y']=dataset['y'][indices]
        
        ### Since we're setting new dataloaders for consistency we should also update the number of classes for the current "task"
        self.numclasses = cumulative_class_total
        
        generator = DG.CifarDataGenerator(dataset['x'],dataset['y'])

        if batchsize == None:
            batch_size = self.args.offline_batch_size

        self.buffer_loader = data.DataLoader(generator, batch_size = batchsize, shuffle = True, num_workers = 4, pin_memory=self.args.cuda)


    
    ### Assume that the network has been unmasked before this is called
    def get_task_conns_buffers(self):
        """
        1. Loop over all tasks in the buffer
        2. For each buffered task, get the network conns
        3. For each subnetwork, store or append the conns to their corresponding dictionary vectors
        4. Get the clustered results from the prepared vectors
        """
        taskconnsdict = {}

        ### Using the full network to do forward passes on all buffered samples, hardcoding 10 as # classes for current experiments
        self.FLOPcounts["inference"] += self.get_FLOPS(self.subnetID, epochs=1, numsamples=(self.args.buffernum * 10 * len(list(self.buffered_tasks.keys()))), 
                                                        step="forwardonly", unmasked=True)

        for task in list(self.buffered_tasks.keys()):
            ### Pass the buffered task data to be used for getting conns
            connsdict = self.calc_conns(data = self.buffered_tasks[task])
            
            taskconnsdict[task] = {}
            
            ### Loop over all subnetworks
            for subnet in self.all_task_masks.keys():
                taskmask = self.all_task_masks[subnet]
                task_conn_ave = 0
    
                for i in range(0, len(self.child_idxs)):
                    ### Skip the first N convolutional layers when calculating the value. 
                    ###   This lets us focus only on the later layers which are expected to have the more specialized features
                    parent_idx = self.parent_idxs[i]
                    child_idx = self.child_idxs[i]
    
                    ### Use the task mask to mask for weights in the chosen task
                    bool_mask = taskmask[child_idx].eq(1)
                    
                    ### Need to convert to a shape of 2 to match the the connectivity dictionary shape            
                    if bool_mask.dim() == 4:
                        ### True for all filters in the parent layer which have at least one outgoing, non-zero weight in the current task
                        bool_mask = bool_mask.any(dim=3).any(dim=2)
        
                    bool_mask = bool_mask.transpose(0,1)
        
                    ### Index into connectivity dictionary based on current child layer index. Works for layers that only appear once in child_idxs
                    task_conns = torch.from_numpy(connsdict[i])[bool_mask.cpu()]
                    ### For now I just want to use the average conns over the layer
                    if task_conns.numel() > 0:
                        task_conn_ave += torch.nanmean(torch.square(task_conns))

                taskconnsdict[task][subnet] = task_conn_ave/len(self.child_idxs)

        ### Returns task-specific network connectivity averages. This is the average connectivity over all weights used during a given task
        return taskconnsdict

    
    
    ### Apply KNN clustering on subnetworks based on their connectivity for all buffered tasks
    def cluster_subnetworks_by_conns(self):  
        self.network.unmask_network()
        subnetconns = self.get_task_conns_buffers()
        print("Subnets collected: ")
        for key in subnetconns.keys():
            print(subnetconns[key])
        vectors = {}
        ### For each subnetwork, get all task conns
        for subnet in self.all_task_masks.keys():
          vectors[subnet] = []
          for task in subnetconns.keys():
            # vectors[subnet] = vectors[subnet] + list(subnetconns[task][subnet].tolist()
            vectors[subnet].append(subnetconns[task][subnet])

        
        ### Get T-dimensional KNN clustering where T is the number of tasks or subnetworks. In this way we're seeing how correlated each pair is for each other subnetwork.
        ### A subnetworks correlation with itself is 1, acting as a basis vector for one of the T dimensionsimport numpy as np
        
        corrs = {}
        ### Compute Pearson correlations between all pairs of subnetworks
            ### Each subnetwork is represented here by a vector of all task-mean conns for that subnetwork, so this is the correlation among 6 such vectors
        for a in vectors.keys():
          corrs[a] = []
          for b in vectors.keys():
            vectora = np.asarray(vectors[a])
            vectorb = np.asarray(vectors[b])
            corrs[a].append(np.corrcoef(vectora,vectorb)[0,1])
            
        ### Convert the dictionary into a feature matrix
        feature_matrix = np.array([corrs[i] for i in range(len(corrs))])

        ### Perform KNN clustering
        kmeans = KMeans(n_clusters=self.args.clusterNum, random_state=0)
        clusters = kmeans.fit_predict(feature_matrix)
        
        ### The 'clusters' variable now contains cluster assignments for each network.
        ### Each network (0 to clusterNum) is assigned to one of the K clusters.
        print("\n\n\nCluster assignments:", clusters)
        print("Tasks by subnet: ", self.tasks_by_subnet)
        
        return clusters
    
    
    





    ### Retrain the merged subnetwork and any dependent subnetworks (since we don't share weights, there are no dependent subnetworks for this paper)
    def retrain_subnetworks(self, subnet):
        
        ###Before retraining we need to go through all necessary buffered tasks, pool them together, make a trainloader out of them, and then assign them
        ### The dataset takes advantage of our self-defined task numbers (from when buffered data was stored) to offset each tasks labels for the stacked classifier
        ### This offsetting is only done during retraining. During inference of online data individual classifiers are used, so they can use their native labels

        ### Prepare and assign the new train and testloaders for retraining the buffered task data
        self.prepare_buffered_dataset()

        ### Get all tasks for the newly merged subnetwork
        tasks = self.tasks_by_subnet[subnet]

        ### This function merges the individual classifiers of each task belonging to the subnetwork for joint training purposes
        self.network.merge_task_classifiers(tasks)

        ### Need to just make sure everything is on same device prior to sharing decision
        if self.args.cuda:
            self.network.model = self.network.model.cuda()

        ### Reapply the new mask and then prune the appropriate weights from the classifier. This pruning will be applied to all constituent classifers when its later split
        print("Before outgoing omitted in retraining, subnetID is: ", self.subnetID)
        self.network.apply_mask(self.all_task_masks, self.subnetID)
        self.zero_outgoing_omitted()

        print("Retraining merged classifier size: ", self.network.model.classifier.weight.data.size())

        ### Get number of FLOPs from training on all related buffered task data prior to pruning        
        self.FLOPcounts["retraining"] += self.get_FLOPS(self.subnetID, epochs=self.args.train_epochs, 
                                            numsamples=(self.args.buffernum*10*len(tasks)), step="train")

        ### Retrain offline on buffered task data for current merged subnetwork
        self.train_offline(self.args.train_epochs, self.taskid, save=False, savename=self.retrain_path, mergedclassifier=True)
        self.prune(merged = True)

        ### Get number of FLOPs from training on all related buffered task data after pruning        
        self.FLOPcounts["retraining"] += self.get_FLOPS(self.subnetID, epochs=self.args.finetune_epochs, 
                                            numsamples=(self.args.buffernum*10*len(tasks)), step="train")

        self.train_offline(self.args.finetune_epochs, self.taskid, save=False, savename=self.retrain_path, best_accuracy=0, mergedclassifier=True)

        
        ### Reassign the new weights to each individual task classifier in the network
        self.network.split_task_classifiers(tasks, deletemerged=True)
        self.network.model.classifier = self.network.classifiers[tasks[-1]]
        self.network.classifiermerged = None
        self.network.merged_mask = None             
        
        ### Recalculate the subnetwork's Acc values on the different buffered tasks 
        self.store_ID_ACCS(subnet)
    

    
    
    ### Assume the network is fully unmasked prior to calling this function
    def merge_subnetworks(self, cluster):

        print("\n\nMerging the clustered subnetworks: ", cluster)
        print("Initial tasks by subnet: ", self.tasks_by_subnet)
        for c in cluster:
            for task in self.tasks_by_subnet[c]:
                ### Undo the masking of tasks belonging to merging subnetworks due to the changing masks. They are re-initialized appropriately after merging 
                print("Resetting classifier: ", task, " mask for subnet: ", c)
                self.network.classifier_masks[task] = torch.ByteTensor(self.network.classifiers[task].weight.data.size()).fill_(1)
            
        ### Currently we're not using weight sharing, so share history should remain empty
        self.ID_ACCS =         utils.update_ID_ACCS(self.ID_ACCS.copy(), cluster, self.tasks_by_subnet.copy())
        self.ACCthresholds =   utils.update_ACC_thresholds(self.ACCthresholds.copy(), cluster, self.tasks_by_subnet.copy())
        self.tasks_by_subnet =  utils.update_tasks_by_subnet(self.tasks_by_subnet.copy(), cluster)
        self.all_task_masks = utils.update_all_task_masks(self.all_task_masks.copy(), cluster)
                

        ### Now that we've shifted all of the dictionaries for the merge, we need to offset by 1 when indexing for subnetB because its ID is 1 lower due to removing subnetA
        ### We want to now set the current subnetwork to the last subnetwork in the cluster which everything was merged into, shifted down to account for removing the merged subnets
        self.subnetID = cluster[-1] - (len(cluster)-1)

        ### It doesn't really matter what we set this to, we'll be changing the classifier during the retraining anyways as it needs to be merged for all subnet tasks
        print("Post-Merge Tasks by Subnet: ", self.tasks_by_subnet)
        print("Post-Merge All_task_masks keys: ", self.all_task_masks.keys())
        print("Post-Merge ID ACCS keys: ", self.ID_ACCS.keys())
        print("Post-Merge ID Thresholds: ", self.ACCthresholds)
        self.network.set_dataset(self.tasks_by_subnet[self.subnetID][-1])


        self.zero_outgoing_omitted()

        
        
        ### Call the necessary function to retrain, prune, and finetune the subnetwork and any dependent subnetworks, recursively.
        self.retrain_subnetworks(self.subnetID)

        ### Now the subnetworks masks have been merged, the subnetwork has been retrained (and any existing dependent subnets), and all dictionaries have been updated accordingly





    ### Cluster the subnetworks by connectivity with KNN clustering, then merge as needed to have K subnetworks
    def cluster_and_merge(self):
        print("Clustering")
        clusters = self.cluster_subnetworks_by_conns()
        # print("Clusters collected")
        numclusters = self.args.clusterNum
        subnets_by_cluster = {}
        for c in range(numclusters):
            subnets_by_cluster[c] = []

            ### for each cluster, go through all subnetworks and if multiple subnetworks are in the cluster, merge them
            ### Because we shouldnt change the number of subnetworks while looping, first we just get the indices of mergeable subnetworks
            
            for i,subnet in enumerate(clusters):
                if subnet == c:
                    subnets_by_cluster[c].append(i)
                   
        self.eval_dicts['mergehistory'][self.set_num] = subnets_by_cluster.copy()            
        for c in range(numclusters):
            if len(subnets_by_cluster[c])>1:
                self.network.unmask_network()
                self.merge_subnetworks(subnets_by_cluster[c])
                
                
                ### Since we merged subnetA into subnetB, we need to decrement all subnet indices > subnetA to reflect the shift in subnetIDs
                
                subnets = subnets_by_cluster[c]

                ### for every subnetwork that was removed when merging the current cluster, we need to shift all subsequent subnetworks down 1
                ### Using the built-in reversed function to go backwards, shifting to avoid shifting the larger IDs before accessing them with our stored index
                ### Dont need to shift the final subnetwork because its the one that the others were merged into
                print("Initial subnets_by_cluster: ", subnets_by_cluster)
                for i in reversed(subnets[:-1]):
                    for k in range(numclusters):
                        for j in range(len(subnets_by_cluster[k])):
                            if subnets_by_cluster[k][j] > i:
                                subnets_by_cluster[k][j] = subnets_by_cluster[k][j] - 1
                    print("Removed i to get subnets_by_cluster: ", subnets_by_cluster)












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
    unmasked: Whether or not a task mask has been applied to the network
    Note: For backprop operations we use the approximate ratio of 2:1 back:forward FLOPS per layer
            Also for simplicity we omit the FLOPs from the classifiers. All classifiers are 10-label for experiments, with the final layer being 512 neurons
                as such these costs are expected to be fairly negligible overall
    """
    def get_FLOPS(self, subnetID, epochs=1, numsamples=5000, step="train", unmasked=False):
        
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
                ### Used when all subnetworks are unmasked, primarily when making predictions on a new set's task ID
                if unmasked == True:
                    ### Forward pass on all samples with the unmasked network
                    FLOPS += (outputsize * kernelsize * totalfilters * numsamples)
                else:
                    ### Forward pass on all samples with the current subnetwork only
                    FLOPS += (outputsize * kernelsize * numfilters * numsamples)

            ### Train the network for some number of batches and epochs        
            elif step == "train":
                ### For a 2:1 back:forward ratio of FLOPs training takes 3 times as many flops as a forward pass, and is done for some number of epochs on all samples
                FLOPS += 3 * (outputsize * kernelsize * numfilters * numsamples * epochs)
            
        return FLOPS