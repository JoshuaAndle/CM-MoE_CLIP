import torch
import sys
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets, transforms
from copy import deepcopy


# import constants as C
from AuxiliaryScripts import clmodels
from AuxiliaryScripts import utils
import copy

class Network():
    def __init__(self, args, pretrained="True"):
        self.args = args
        self.arch = args.arch
        self.cuda = args.cuda

        self.preprocess = None
        self.model = None
        self.pretrained = pretrained

        self.classifiers, self.classifier_masks = {}, {}
        self.classifiermerged = None        

        ### This is a weight mask for the temporary merged classifier created during the retraining of merged subnetworks. Its initialized and reset before/after retraining
        self.merged_mask = None


        if self.arch == "modresnet18":
            self.model = clmodels.modifiedresnet18()
        elif self.arch == "vgg16":
            self.model = clmodels.vgg16()        
        else:
            sys.exit("Wrong architecture")

        if self.cuda:
            self.model = self.model.cuda()
    
        """
            When to use the backup statedict:
                Update it at the end of training a task, once all the weights for the task are finalized, or when pruning weights. 
                Load it at the start of a task, resetting the network masking to include all previously frozen and omitted weights.
        """
        
        self.backupmodel = copy.deepcopy(self.model).cuda()
        self.initialmodel = copy.deepcopy(self.model).cuda()


    """
    The Network class is responsible for low-level functions which manipulate the model, such as training, evaluating, or selecting the classifier layer
    """
    
    def add_dataset(self, dataset, num_classes):
        """Adds a new dataset to the classifier."""
        if dataset not in self.classifiers.keys():
            if self.arch == 'resnet18' or self.arch == "modresnet18":
                self.classifiers[dataset] = (nn.Linear(512, num_classes))
            elif self.arch == 'vgg16':
                self.classifiers[dataset] = (nn.Linear(4096, num_classes))
        
            self.classifier_masks[dataset] = torch.ByteTensor(self.classifiers[dataset].weight.data.size()).fill_(1)

    def set_dataset(self, dataset):
        """Change the active classifier."""
        if dataset == "merged":
            assert self.classifiermerged is not None, "classifiermerged has not been created"
            self.model.classifier = self.classifiermerged
            self.backupmodel.classifier = self.classifiermerged
        else:
            assert dataset in self.classifiers.keys(), f"dataset {dataset} is not in classifier keys {self.classifiers.keys()}"
            self.model.classifier = self.classifiers[dataset]
            self.backupmodel.classifier = self.classifiers[dataset]
        if self.cuda:
            self.model.cuda()







    """
    Need to adjust make_grads_zero to also ensure that all incoming weights to a frozen filter are zeroed, and all weights out of an omitted filter are zeroed as well
    """
    ### Set all frozen and pruned weights' gradients to zero for training
    def make_grads_zero(self, all_task_masks, subnetID, taskID, mergedClassifier=False):
        """Sets grads of fixed weights to 0."""

        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                ### should give 1s for every non-frozen, non-pruned weight, and 0 for the rest
                trainable_mask = utils.get_trainable_mask(module_idx, all_task_masks, subnetID)
                
                ### Omit incoming weights to trainable filters if they correspond to omitted filters in the parent layer.
                # trainable_mask = torch.minimum(trainable_mask.cuda(), omit_mask[module_idx].cuda())

                # Set grads of all weights not belonging to current dataset to 0.
                if module.weight.grad is not None:
                    module.weight.grad.data[trainable_mask.eq(0)] = 0
                if taskID>0 and module.bias is not None:
                    print('THIS SHOULD NEVER RUN SINCE BIASES ARE NONE')
                    module.bias.grad.data.fill_(0)

        ### Zero gradients of all weights in the classifier connected to pruned or omitted filters
        # self.model.classifier.weight.grad.data[all_task_masks[taskID]["fc"].eq(0)] = 0
        if mergedClassifier == False:
            self.model.classifier.weight.grad.data[self.classifier_masks[taskID].eq(0)] = 0
        else:
            self.model.classifier.weight.grad.data[self.merged_mask.eq(0)] = 0
  
  
  
  
        
    ### Just checks how many parameters per layer are now 0 post-pruning
    def check(self, verbose=False):
        """Makes sure that the layers are pruned."""

        for layer_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                weight = module.weight.data
                num_params = weight.numel()
                num_zero = weight.view(-1).eq(0).sum()
                if len(weight.shape) > 2:
                    filter_mask = torch.abs(weight).le(0.000001).all(dim=3).all(dim=2).all(dim=1)
                else:
                    filter_mask = torch.abs(weight).le(0.000001).all(dim=1)
                
                num_filters = filter_mask.numel()
                num_pruned_filters = filter_mask.view(-1).sum()


                if verbose:
                    print('Layer #%d: Pruned Weights %d/%d (%.2f%%), Pruned Filters %d/%d (%.2f%%)' %
                          (layer_idx, num_zero, num_params, 100 * num_zero / num_params, num_pruned_filters, num_filters, 100 * num_pruned_filters / num_filters))









       
    """
        Mask manipulation can be done easily by the combination of three functions:
        1. update_backup: After editing weights in a subnetwork, this commits the changes to the composite backup model in self.network
        2. unmask_network: When changing subnetworks, this unmasks all weights, reverting them to their frozen values from self.network.backupmodel
        3. apply_mask: Apply masking for a given subnetwork by zeroing all omitted weights
    """
    
    ### Done after training or finetuning. Updates the backup model to reflect changes in the model from training, pruning, or merging
    def update_backup(self, tasknum = -1, taskmask = None):
        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                layermask = taskmask[module_idx]

                for module_idx2, module2 in enumerate(self.backupmodel.shared.modules()):
                    if module_idx2 == module_idx:
                        module2.weight.data[layermask.eq(1)] = module.weight.data.clone()[layermask.eq(1)]

       
       
    ### This just reloads the backup composite network
    def unmask_network(self):
        self.model.shared = copy.deepcopy(self.backupmodel.shared).cuda()             
       
       
      
    ### Applies appropriate mask to recreate task model for inference
    def apply_mask(self, all_task_masks, subnetID):
        """To be done to retrieve weights just for a particular dataset"""
        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                module.weight.data[all_task_masks[subnetID][module_idx].eq(0)] = 0.0





    """
        Merge and unmerged classifiers for use when doing joint training of merged subnetworks
        1. merge_task_classifiers: Temporarily merge individual classifiers for joint training
        2. split_task_classifiers: Copy merged classifier's weights accordingly to each constituent task's individual classifier
    """




    def merge_task_classifiers(self,tasks):
        ### Classifier merged is a temporary classifier layer formed by stacking all of the subnet-specific task classifiers. It will be used for joint retraining
        ###    and then the trained weights will be split back up into their task-specific classifier tensors
        weightsmerged = None
        for t in tasks:
            if weightsmerged == None:
                ### Get the first classifier to build off of
                ### Not copying weights because theres no concern of overwriting the original, we intentionally do that after the training anyways
                weightsmerged = copy.deepcopy(self.classifiers[t].weight.data)
            else:
                weightsmerged = torch.cat((weightsmerged, copy.deepcopy(self.classifiers[t].weight.data)), dim=0)

        self.classifiermerged = nn.Linear(weightsmerged.size()[1], weightsmerged.size()[0])
        self.classifiermerged.weight.data = copy.deepcopy(weightsmerged)

        self.set_dataset("merged")
        print("Merged classifier size: ",  self.model.classifier.weight.data.size())

        ### We'll simply initialize an all-1 mask which will then be pruned by zeroing outgoing omitted weights
        self.merged_mask = torch.ByteTensor(self.classifiermerged.weight.data.size()).fill_(1)

        
        
        
        
    ### Name is slightly misleading, this function takes the merged classifier and overwrites the constituent tasks' individual classifiers with the jointly trained weights
    def split_task_classifiers(self, tasks, deletemerged=True):
        cumulative_num_classes = 0
        for t in tasks:
            numclasses = self.classifiers[t].weight.data.size()[0]

            ### Get the slice of weights from merged classifier corresponding to the given task and assign them back to their own classifier
            self.classifiers[t].weight.data = copy.deepcopy(self.model.classifier.weight.data[cumulative_num_classes:(cumulative_num_classes + numclasses)])

            ### Shouldnt actually need to update the masks since theyre iterated over for the subnetwork within zero_outgoing_forward which is the only function that changes them
            cumulative_num_classes += numclasses



