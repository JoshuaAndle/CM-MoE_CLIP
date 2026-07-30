"""
Script for calculating first and second order metrics of transformer activations
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse

import numpy as np
import copy
import torch
import torch.nn as nn
from utils import metric_utils


###General flags
FLAGS = argparse.ArgumentParser()

FLAGS.add_argument('--metric_order', type=str, default='first', choices=['first', 'second'], help='Whether to take a first or second-order metric.')
FLAGS.add_argument('--metric', type=str, default='cos', choices=['cos', 'corr', 'dist', 'energy'], help='Which metric to use.')
FLAGS.add_argument('--first_layer', type=str, choices=['block_input', 'expert_inputs', 
                                                        'expert_down_acts', 'expert_up_acts',
                                                        'dispatcher_combined', 'mlp_output', 'logits_only'], help='Which layer to use for the first activation tensor.')
FLAGS.add_argument('--second_layer', type=str, choices=['block_input', 'expert_inputs', 
                                                        'expert_down_acts', 'expert_up_acts',
                                                        'dispatcher_combined', 'mlp_output'], help='Which layer to use for the second activation tensor.')
FLAGS.add_argument('--branch', type=str, default='image', choices=['image', 'text'], help='Which modality to calculate metric for.')

# FLAGS.add_argument('--score_threshold', type=float, default=2.0, help='The ratio of KMeans score needed for a past subnetwork to be shared through clustering')

FLAGS.add_argument('--num_tasks', type=int, default=6, help='Number of tasks to calculate metric over')
FLAGS.add_argument('--num_subnets', type=int, default=6, help='Number of subnetworks to calculate metric over')
FLAGS.add_argument('--num_blocks', type=int, default=6, help='Number of blocks to calculate metric over')

FLAGS.add_argument('--condition_by_class', action='store_true', default=False, help='Class-conditioning of activations prior to metric calculation')
FLAGS.add_argument('--remove_padding', action='store_true', default=False, help='Remove all zero-padded tokens from text inputs')

FLAGS.add_argument('--fold_samples', action='store_true', default=False, help='Fold the sample dimension with averaging of activations')
FLAGS.add_argument('--fold_tokens', action='store_true', default=False, help='Fold the token dimension with averaging of activations')
FLAGS.add_argument('--fold_embed', action='store_true', default=False, help='Fold the embedding dimension with averaging of activations')
FLAGS.add_argument('--subsample_tokens', action='store_true', default=False, help='Take only the cls or eot tokens as aggregate information of all token relations')

FLAGS.add_argument('--block_set', type=str, default='first', choices=['first', 'last'], help='Which blocks to use.')



# loadpath = "./results/cifar_mnist_2_class/6TASKS/moa-clipMETHOD/10SEED/moa-clip_all-both_SEED10/"
FLAGS.add_argument('--loadpath', type=str, help='Which directory to load from. Needs to contain the "activations" subdirectory.')


def main():
    args = FLAGS.parse_args()

    ### Early termination conditions

    torch.cuda.set_device(0)
    
    print('Arguments =')
    for arg in vars(args):
        print('\t'+arg+':',getattr(args,arg))
    print('-'*100)    
    
    layer_l = args.first_layer
    layer_k = args.second_layer
    metric = args.metric
    metric_order = args.metric_order

    num_tokens = 197 if args.branch == "image" else 77

    num_tasks = args.num_tasks
    num_subnets = args.num_subnets
    num_blocks = args.num_blocks
    branch = args.branch

    condition_by_class = args.condition_by_class
    remove_padding = args.remove_padding
    fold_samples = args.fold_samples
    fold_tokens = args.fold_tokens
    fold_embed = args.fold_embed
    subsample_tokens = args.subsample_tokens


    ### If just looking at logits, there is no need to calculate for each block in the model
    if args.first_layer == "logits_only":
        num_blocks = 1




    #!# Note: I think corr handles condition by class wrong and assumes averaging occured, need to check it out

    preprocessing, condition, padding = "PREPROCESSING", "none", "none"

    if condition_by_class:
        condition = "cond_by_class"
    if remove_padding:
        padding = "remove_padding"

    preprocessing = f"{preprocessing}_{condition}_{padding}"


    fold_dir = "fold_none"
    if fold_samples:
        fold_dir = "fold_samples"
    elif fold_tokens:
        fold_dir = "fold_tokens"
    elif fold_embed:
        fold_dir = "fold_embed"
    elif subsample_tokens:
        fold_dir = "subsample_cls_token"



    loadpath = args.loadpath
    if args.metric_order == "first":
        save_path = os.path.join(loadpath,"metrics",args.metric_order, args.metric, f"{layer_l}-{layer_k}", preprocessing, fold_dir)

    elif args.metric_order == "second":
        save_path = os.path.join(loadpath,"metrics",args.metric_order, args.metric, f"{layer_l}", preprocessing, fold_dir)
    else:
        return

    os.makedirs(save_path, exist_ok=True)
    save_file = f"{save_path}/metric_dict_{args.branch}_{args.block_set}.pt"

    # Make empty dictionary for subnet metric values
    metric_results = {}




    ### Load information dicts shared by all tasks and subnets
    logit_path = os.path.join(loadpath, "logits.pt")
    logit_dict = torch.load(logit_path, map_location=torch.device('cpu')) # Just has the [task][subnet][batch] logit values as 2d tensors of (#imgs, #classes)

    # stat_path = os.path.join(loadpath,  "mem_acc_stats.pt")
    # stat_dict = torch.load(stat_path, map_location=torch.device('cpu')) # ['total_correct', 'total_samples', 'ave_acc'] each of which is [task][subnet]



    block_idxs = []
    for b in range(num_blocks):
        if args.block_set == "first":
            block_idxs.append(b)
        else:
            ### Count backwards from the last block (12 blocks for CLIP)
            start_block = 12 - num_blocks
            block_idxs.append(start_block + b)





    # Loop over all task and subnet combinations, and blocks once that is available
    for task in range(num_tasks):
        if task not in metric_results.keys():
            metric_results[task] = {}

        ### Load label dict containing information about labels and each classes text tokens
        labelpath = os.path.join(loadpath, f"task_{task}_mem_label_dict.pt")
        labels_dict = torch.load(labelpath, map_location=torch.device('cpu')) # ['labels', 'labels_mapped', 'tokens', 'eot_indices']

        # labels = torch.cat(labels_dict["labels"], dim=0)
        # labels_mapped = torch.cat(labels_dict["labels_mapped"], dim=0)
        ### Using labels_mapped only, since it should behave identically when using exposed_classes with all known classes, and will match the text inputs when using batched_exposed_classes
        labels = torch.cat(labels_dict["labels_mapped"], dim=0)
        # eot_indices = torch.cat(torch.tensor(labels_dict["eot_indices"]), dim=0)

        num_batches = len(labels_dict["eot_indices"])
        

        eot_indices = []
        for b in range(num_batches):
            eot_indices.extend(labels_dict["eot_indices"][b])

        # text_labels = []
        ### Batches of text inputs may have non-uniform # samples, so we construct 'identity' labels before concatenating them
        if args.branch == "text": 
            # for b in range(num_batches):
                ### Add labels 0:c for the c class labels present in the given batch b
                # text_labels.append(torch.arange(len(labels_dict["eot_indices"][b])))
            # labels = torch.cat(text_labels, dim=0)

            #!# The initial approach incorrectly assumed each matching index belonged to the same class. This approach is also imperfect, as it may treat
            ###   reoccuring classes across batches as new labels, but it should not be a significant issue for the time being
            ### To make a fully accurate text_label tensor, we would need a function that constructs it using the unmapped labels of each batch to identify repeated classes
            ###     but then also have the conversions into names to handle redundant class labels. If it becomes necessary I will look into it more to replace this naive approach
            labels = torch.arange(len(eot_indices))


        ### For first order metrics, calculate between two layers in the same subnetwork
        if args.metric_order == "first":
            for subnet in range(num_subnets):
                print("Starting Subnet ", subnet, flush=True)

                # logits = torch.cat(logit_dict[task][subnet], dim=0)
                logits = logit_dict[task][subnet]

                key_tsb = f"{task}_{subnet}"


                ### Skip block loop and activation prep to just get logit energy scores
                if args.metric == "energy":
                    ### no preprocessing is used on logits so we can just get the score directly
                    metric_results[task][subnet] = metric_utils.calculate_energy(logits)
                else:
                    if subnet not in metric_results[task].keys():
                        metric_results[task][subnet] = {}

                    for b in range(num_blocks):
                        ### Get the block index that actually corresponds to the bth index we are calculating for
                        #!# Eventually this system should be scrapped for absolute indices since we are saving to a dict anyways
                        block = block_idxs[b]

                        filepath = os.path.join(loadpath, "activations", key_tsb, f"acts_dict_block_{block}.pt")

                        ## Within the loop, load task dict for one task, subnet, and block combination
                        acts_dict = torch.load(filepath, map_location=torch.device('cpu')) # [modality_block][layer][batches]

                        ## Copy the target two layers l and l' and delete the original object for memory purposes
                        branch_key = f"{branch}_{block}"
                        acts_l = copy.deepcopy(acts_dict[branch_key][layer_l]) 
                        acts_k = copy.deepcopy(acts_dict[branch_key][layer_k])


                        ## Preprocess the two activation tensors based on which layers they are
                        ### Concatenate over batches. Experts need to be handled differently due to sample splitting across experts
                        if layer_l in ["expert_inputs", "expert_down_acts", "expert_up_acts"]:
                            acts_l = metric_utils.cat_experts(acts_l)
                            acts_k = metric_utils.cat_experts(acts_k)
                            acts_l = metric_utils.remove_empty_experts(acts_l)
                            acts_k = metric_utils.remove_empty_experts(acts_k)
                        else:
                            acts_l = metric_utils.cat_dict_acts(acts_l)
                            acts_k = metric_utils.cat_dict_acts(acts_k)
                        
                        # print(acts_k.shape)
                        ### Unflattening Expert Inputs. Expert_inputs will never be layer_k
                        if layer_l == "expert_inputs":
                            acts_l = metric_utils.unflatten_expert_inputs(acts_l, num_tokens)


                        temp_expert_results = []
                        final_metric = None
                        ## If the layers are expert layers, initialize an empty temp list and loop over all nonzero expert tensors:
                        if layer_l in ["expert_inputs", "expert_down_acts", "expert_up_acts"]:
                            ### Both layers in the subnet should have the same exact expert routing, and therefor the same # of experts used
                            assert(len(acts_l) == len(acts_k))

                            gates = acts_dict[branch_key]["gates"]
                            gates = metric_utils.cat_dict_acts(gates)
                            # print("Gates shape: ", gates.shape, " and labels length: ", len(labels))

                            exp_labels = metric_utils.get_expert_labels(gates,labels)


                            # print(f"Length of acts_l {len(acts_l)} and exp_labels {len(exp_labels)}")



                            # else:
                            #     ### Text labels are constructed at start of task loop
                            #     # if task == 0:
                            #     #     print("Text labels: ", labels)
                            #     exp_labels = metric_utils.get_expert_labels(gates, labels)
                            assert len(exp_labels) == len(acts_l), f"len of exp_labels {len(exp_labels)} does not match len of acts_l {len(acts_l)}"

                            for exp in range(len(acts_l)):
                                ### Calculate metric for each expert index separately and store results into the empty temp list
                                metric_result = metric_utils.get_metric_argdict(acts_l[exp], acts_k[exp], exp_labels[exp], args, eot_indices, logits)
                                temp_expert_results.append(metric_result)
                            ### Average over all experts for the given task, subnet, block combination
                            final_metric = torch.mean(torch.tensor(temp_expert_results)).item()

                        ## Else if the layers are shared (block input, dispatch combined, mlp output) directly calculate metric for the full layers
                        else:
                            final_metric = metric_utils.get_metric_argdict(acts_l, acts_k, labels, args, eot_indices, logits).item()

                        assert(final_metric is not None)

                        ## Store metric into the empty results dictionary
                        metric_results[task][subnet][b] = final_metric


        ### For second order metrics, calculate between two subnetworks using the same layer
        elif args.metric_order == "second":
            ### Since each subnet uses different experts we can't reasonably compare between subnets on these layers
            assert(layer_l not in ["expert_inputs", "expert_down_acts", "expert_up_acts"])

            ### Loop over each potentially Out-of-Distribution subnetwork and compare metric with ID subnetwork acts
            #!# This does include comparing the ID subnet against itself, which could be skipped for time saving but we include here
            for OoD_subnet in range(num_subnets):
                print("Starting Subnet ", OoD_subnet, flush=True)

                # logits = torch.cat(logit_dict[task][OoD_subnet], dim=0)
                logits = logit_dict[task][OoD_subnet]

                ID_subnet = task
                ID_key_tsb = f"{task}_{ID_subnet}"
                OoD_key_tsb = f"{task}_{OoD_subnet}"

                ### Skip block loop and activation prep to just get logit energy scores
                if args.metric == "energy":
                    ### no preprocessing is used on logits so we can just get the score directly
                    metric_results[task][OoD_subnet] = metric_utils.calculate_energy(logits)
                else:
                    if OoD_subnet not in metric_results[task].keys():
                        metric_results[task][OoD_subnet] = {}

                    for b in range(num_blocks):
                        block = block_idxs[b]

                        OoD_filepath = os.path.join(loadpath, "activations", OoD_key_tsb, f"acts_dict_block_{block}.pt")
                        ID_filepath = os.path.join(loadpath, "activations", ID_key_tsb, f"acts_dict_block_{block}.pt")

                        branch_key = f"{branch}_{block}"
                        ## Within the loop, load task dict for one task, subnet, and block combination
                        acts_dict = torch.load(OoD_filepath, map_location=torch.device('cpu'))
                        ## Copy the target layer l
                        acts_OoD = copy.deepcopy(acts_dict[branch_key][layer_l])

                        ## Reload the act dict using the ID subnet for the given task
                        acts_dict = torch.load(ID_filepath, map_location=torch.device('cpu'))
                        acts_ID = copy.deepcopy(acts_dict[branch_key][layer_l])


                        ## Preprocess the two activation tensors based on which layers they are
                        ### Concatenate over batches. 
                        acts_OoD = metric_utils.cat_dict_acts(acts_OoD)
                        acts_ID  = metric_utils.cat_dict_acts(acts_ID)

                        final_metric = None
                        final_metric = metric_utils.get_metric_argdict(acts_OoD, acts_ID, labels, args, eot_indices, logits).item()

                        assert(final_metric is not None)

                        ## Store metric into the empty results dictionary
                        metric_results[task][OoD_subnet][b] = final_metric




    # Save metric results
    torch.save(metric_results, save_file)

    print("Saved to ", save_file, "\n\n")
    print(metric_results, "\n\n")






if __name__ == '__main__':
    main()
