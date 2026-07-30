"""
Does standard subnetwork training on all tasks

"""



from __future__ import division, print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import json
import warnings
import copy
import time

import numpy as np
import torch
from torch.optim.lr_scheduler  import MultiStepLR

from AuxiliaryScripts import utils
from AuxiliaryScripts.replaymanager import Manager

# To prevent PIL warnings.
warnings.filterwarnings("ignore")

###General flags
FLAGS = argparse.ArgumentParser()
FLAGS.add_argument('--run_id', type=str, default="000", help='Id of current run.')
FLAGS.add_argument('--cuda', action='store_true', default=True, help='use CUDA')
FLAGS.add_argument('--arch', choices=['resnet18', 'modresnet18', 'vgg16'], default='modresnet18', help='Architectures')
FLAGS.add_argument('--dataset', type=str, choices=['online_mixed_cifar_pmnist','online_cifar_rotmnist','online_cifar_jump20rotmnist','online_cifar_jump30rotmnist'], default='online_mixed_cifar_pmnist', help='Name of dataset')
FLAGS.add_argument('--set_num', type=int, default=0, help='Current task number.')
FLAGS.add_argument('--save_prefix', type=str, default='../checkpoints/', help='Location to save model')
# Training options.
FLAGS.add_argument('--lr', type=float, default=0.1, help='Learning rate')
FLAGS.add_argument('--Milestones', nargs='+', type=int, default=[30,60,90])
FLAGS.add_argument('--Gamma', type=float, default=0.1)   
# Pruning options.
FLAGS.add_argument('--prune_method', type=str, default='structured', choices=['structured'], help='Pruning method to use')
FLAGS.add_argument('--sparsity_type', type=str, default='unfrozen_only', choices=['unfrozen_only'], help='Whether # filters pruned is based on layer width or only # unfrozen filters')
FLAGS.add_argument('--prune_perc_per_layer', type=float, default=0.65, help='% of neurons to prune per layer')
FLAGS.add_argument('--merged_prune_perc_per_layer', type=float, default=0.3, help='% of neurons to prune per layer after merging already-pruned subnetworks')

FLAGS.add_argument('--train_epochs', type=int, default=20, help='Number of epochs to train for')
FLAGS.add_argument('--online_epochs', type=int, default=5, help='Number of epochs to train for')
FLAGS.add_argument('--finetune_epochs', type=int, default=20, help='Number of epochs to finetune for after pruning')

# Misc options
FLAGS.add_argument('--reinit', action='store_false', default=True, help='Reininitialize pruned weights to non-zero values')
FLAGS.add_argument('--userelu', action='store_false', default=True, help='Whether or not to use relu idxs for capturing activations')
FLAGS.add_argument('--buffernum', type=int, default=50, help='Number of images to buffer for a each class')

FLAGS.add_argument('--batch_size', type=int, default=512, help='Batch size')
FLAGS.add_argument('--offline_batch_size', type=int, default=512, help='Batch size for offline retraining after merges and pruning')


FLAGS.add_argument('--clusterNum', type=int, default=2, help='Number of clusters to maintain')
FLAGS.add_argument('--maxSubnets', type=int, default=6, help='Number of subnetworks allowed before clustering')
FLAGS.add_argument('--IDscore', choices=['acc'], default='acc', help='Whether to use accuracy for identifying subnetwork and classifier to use')
FLAGS.add_argument('--unusedthreshold', type=int, default=10, help='Number of tasks allowed without reusing a classifier before it and its data are removed')

FLAGS.add_argument('--prune_batch', type=int, default=10, help='Number of training batches before pruning online')
FLAGS.add_argument('--thresholdwindow', type=int, default=10, help='Number of recent accuracies to consider for ID ACC')
FLAGS.add_argument('--setsize', type=int, default=5000, help='Number images per set Z')


###################################################################################################################################################
###
###     Main function
###
###################################################################################################################################################

#***# Save structure: Runid is the experiment, different task orders are subdirs that share up to the last common task so that the task can be reused/located just by giving the runid and task sequence and can be shared between multiple alternative orders for efficiency
###    Basically this just means 6 nested directories, which are nested in order of task order for the given experiment. So all subdirs of the outermost directory 2 have task 2 as the first task and can share the final dict from task 2 amongst eachother for consistency and efficiency
def main():
    args = FLAGS.parse_args()
    ### Early termination conditions
    if args.prune_perc_per_layer <= 0:
        print("non-positive prune perc",flush = True)
        return
    torch.cuda.set_device(0)
    
    
    
    
    ### Determines which sets and tasks are included in the overall sequence
    if args.dataset in ['online_mixed_cifar_pmnist']: 
        taskset = [*range(0,28,1)]
    elif args.dataset == 'online_cifar_rotmnist':
        taskset = [*range(0,23,1)]
    elif args.dataset == 'online_cifar_jump20rotmnist':
        taskset = [*range(0,20,1)]
    elif args.dataset == 'online_cifar_jump30rotmnist':
        taskset = [*range(0,19,1)]
    else: 
        print("Incorrect dataset name for args.dataset")
        return 0

    num_classes_by_task = utils.get_numclasses(args.dataset)


    ###################
    ##### Prepare Checkpoint and Manager
    ###################
    args.save_prefix = os.path.join("../checkpoints/", str(args.dataset), (str(args.prune_perc_per_layer) + "_" + str(args.merged_prune_perc_per_layer)), (str(args.clusterNum) + '_' + str(args.maxSubnets) + "_" + str(args.unusedthreshold)), str(args.arch), str(args.run_id), str(args.lr))
    os.makedirs(args.save_prefix, exist_ok = True)

    ### If no checkpoint is found, the default value will be None and a new one will be initialized in the Manager
    ckpt = None

    if args.set_num != 0:
        ### Path to load previous task's checkpoint, if not starting at task 0
        previous_task_path = os.path.join(args.save_prefix, str(args.set_num-1), "final.pt")
        
        ### Reloads checkpoint depending on where you are at for the current task's progress (t->c->p)    
        if os.path.isfile(previous_task_path) == True:
            ckpt = torch.load(previous_task_path)
        else:
            print("No checkpoint file found at ", previous_task_path)
            if args.set_num > 0:
                return 0
    
    if ckpt != None:
        manager = ckpt['manager']
    else:
        ### Initialize the manager using the checkpoint.
        ### Manager handles pruning, connectivity calculations, and training/eval
        manager = Manager(args, ckpt, num_classes_by_task, first_task_classnum=num_classes_by_task[args.set_num])












    ########################################################################################################################################################
    ##### Loop Through Tasks    ############################################################################################################################
    ########################################################################################################################################################
    
    
    start_time_all = time.time()

    ### Logic for looping over remaining tasks
    for task in taskset[args.set_num:]:
        print("###########################################################################################")
        print("##################################### Task ", task, "##########################################")
        print("###########################################################################################\n")
        ### Revert any masking of weights at the start of each new task

        
        ### Append the current task to the nested subdirectories for task order
        save_path = os.path.join(args.save_prefix, str(task))

        print("Task #: ", task, " in sequence for dataset: ", args.dataset)

        ### Update paths as needed for each new set
        os.makedirs(save_path, exist_ok = True)
        trained_path = os.path.join(save_path, "trained.pt")
        finetuned_path = os.path.join(save_path, "final.pt")
        ### I could save this as its own checkpoint in case the code stops while retraining, but it takes more memory and not every task will need it so its inconsistent
        manager.retrain_path = finetuned_path
        
        
        ### The tasknum should always reflect the current task, whether its an existing or new distribution
        manager.set_num = task
        manager.taskid = task
        manager.numclasses = num_classes_by_task[task]
        
        ### Prepare dataloaders for new task
        train_data_loader = utils.get_dataloader(args.dataset, args.batch_size, args.setsize, pin_memory=args.cuda, set_num=task, set="train")
        manager.train_loader = train_data_loader
        manager.trainset_raw = utils.get_dataset(args.dataset, args.batch_size, args.setsize, pin_memory=args.cuda, set_num=task, set="train")

        
        ### Check if the first batch of data in the current "task" of the stream is recognized as matching an existing trained/stored distribution
        in_distribution, subnetID, maxtask = manager.check_if_existing_dist(task)

        ### Always False for the first task, True if the maximum ID metric (Acc) is within a threshold distance of an existing task's In-Distribution value
        if in_distribution == True:
            manager.taskid = maxtask
            ### There is only one subnet since we're using the full network
            manager.subnetID = 0
            manager.network.set_dataset(maxtask)
            manager.update_recency(maxtask)
            manager.eval_dicts['identifiedtasks'][task] = maxtask

            
            manager.eval_dicts['tasks_by_set'][task] = manager.tasks_by_subnet[manager.subnetID]

            ### Retrain the known task jointly with all other buffered tasks' stored data
            manager.online_train(args.online_epochs, manager.taskid)
            
        else:
            manager.ID_ACCS[task] = []
            manager.ACCthresholds[task] = []
            ### We store the keys for the buffered tasks in sequential order of when they were added
            manager.buffered_tasks[task] = {}
            ### The trainset is used to collect all buffered task data at once, since retraining with this data wont occur until after the set is finished training
            manager.buffered_tasks[task] = utils.get_samples_to_buffer(manager.trainset_raw, manager.numclasses, task, args.buffernum)

            ### Treat as new task, train new subnetwork and buffer a subset of the training data
            manager.subnetID = 0
            manager.eval_dicts['newdistributions'][task] = task
            manager.eval_dicts['tasks_by_set'][task] = [task]
            
            ### For the first task we need to initialize this dict
            if manager.subnetID not in list(manager.tasks_by_subnet.keys()):
                manager.tasks_by_subnet[manager.subnetID] = [task]
            else:
                manager.tasks_by_subnet[manager.subnetID].append(task)
            
            manager.update_recency(task)
            manager.network.add_dataset(task, num_classes_by_task[task])
            manager.network.set_dataset(task)

            ### Need to just make sure everything is on same device
            if args.cuda:
                manager.network.model = manager.network.model.cuda()

            ### Train the new task along with all buffered tasks' data
            manager.online_train(args.online_epochs, manager.taskid)



        ### Remove any tasks which haven't been used in the given threshold of time along with their buffered data  
        manager.remove_unused_tasks()
     
        print("Manager all task masks: ", manager.all_task_masks.keys())
        print("alltaskmasks length", len(list(manager.all_task_masks.keys())))
        print("maxsubnets length", args.maxSubnets)

        
        ### Record the current state of networks' task accuracies to analyze performance over the course of training
        manager.eval_dicts['ID_ACCS_history'][task] = manager.ID_ACCS.copy()
        manager.eval_dicts['ACCthresholds_history'][task] = manager.ACCthresholds.copy()
        manager.eval_dicts['training_times'][task] = (time.time() - start_time_all)
        print("\n\n\n\n\n\n\n")


    utils.save_ckpt(manager, savename=finetuned_path)


if __name__ == '__main__':
    main()
