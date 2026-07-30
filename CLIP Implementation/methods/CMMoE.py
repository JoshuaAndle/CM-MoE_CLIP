import gc
import random
import time
import logging
import datetime
import os.path as osp
# from tqdm import tqdm
import shutil 

import os
import copy

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans, AgglomerativeClustering


import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision import transforms

from methods._trainer import _Trainer
from utils.train_utils import select_optimizer, select_scheduler, exp_lr_scheduler
from utils.memory import Memory, MemoryBatchSampler, MemorySubnetBatchSampler, MemorySubnetSampler
from utils.online_sampler import OnlineSampler, OnlineMultiDatasetSampler, OnlineTestSampler
# from utils.memory import MemoryBatchSampler
from utils import connectivity
from models.AutoEncoder import AutoEncoder, Alexnet_FE
from utils import metric_utils

from models.clip import peft # import prepare_acts_dict, get_acts_dict, set_sample_count, get_sample_count
from models.clip.peft import ResidualAttentionBlock_MoA


logger = logging.getLogger()


#!# Used for both lora-clip and adapter-clip. 
###     The only difference is that peft_method is lora for lora_clip and adapter for adapter_clip
class CMMoE(_Trainer):

    def __init__(self, args, **kwargs):
        super(CMMoE, self).__init__(args, **kwargs)
        self.visible_classes = self.args.visible_classes
        self.known_tasks = []
        self.finetune_epochs = 1

        self.set_store_layers_flag = False
        # self.act_batches_to_record = 10
        self.max_act_samples = 100000


        self.metric_modality = self.args.metric_modality
        self.metric_order = self.args.metric_order
        self.metric = self.args.metric
        self.first_layer = self.args.first_layer
        self.second_layer = self.args.second_layer
        self.num_blocks = self.args.num_blocks
        self.num_clusters = self.args.num_clusters
        self.experts_per_subnet = self.args.experts_per_subnet
        self.condition_by_class = self.args.condition_by_class
        self.remove_padding = self.args.remove_padding
        self.subsample_tokens = self.args.subsample_tokens
        self.block_set = self.args.block_set


        all_metrics = []
        if self.experiment_type == "metric_calculation":
            ### Note: All of these metrics work best on the final blocks of the network
            self.block_set = "last"
            all_metrics.append({"metric": "energy", 
                                "metric_order": "first",  
                                "metric_modality": "image",   
                                "first_layer": "block_input",         
                                "second_layer": "dispatcher_combined"})
            all_metrics.append({"metric": "cos",    
                                "metric_order": "second", 
                                "metric_modality": "hybrid",  
                                "first_layer": "dispatcher_combined", 
                                "second_layer": "dispatcher_combined"})
            all_metrics.append({"metric": "dist",   
                                "metric_order": "first",  
                                "metric_modality": "hybrid",  
                                "first_layer": "block_input",         
                                "second_layer": "dispatcher_combined"})
            all_metrics.append({"metric": "dist",   
                                "metric_order": "second", 
                                "metric_modality": "hybrid",  
                                "first_layer": "mlp_output",          
                                "second_layer": "mlp_output"})
        self.metric_suite = all_metrics





        ### Track which experts have been frozen and which have been assigned to each task
        self.experts_by_subnet = {}
        self.experts_index_by_subnet = {}
        self.autoencoders = {}
        self.optimizer_encoder = None


        ### Pretrained Alexnet model used for producing features for AutoEncoder inputs
        pretrained_alexnet = models.alexnet(weights="IMAGENET1K_V1")

        for k, v in pretrained_alexnet.named_parameters():
            v.requires_grad = False
        # Derives a feature extractor model from the Alexnet model
        self.feature_extractor = Alexnet_FE(pretrained_alexnet)

        self.batch_counter = 0

        self.subnet = -1
        self.tasks_by_subnet = {}






        stored_blocks = []
        if self.block_set == "first":
            stored_blocks = list(range(self.num_blocks))
        else:
            total_blocks = len(self.args.adapter_blocks_text)
            stored_blocks = list(range(total_blocks - self.num_blocks, total_blocks))

        self.stored_blocks = stored_blocks
        print("\n\n\nSetting stored blocks: ", self.stored_blocks)
        print("Peft encoders: ", self.args.peft_encoder)
        print("Metric modality: ", self.args.metric_modality)
        print("Adapter blocks image: ", self.args.adapter_blocks_image)
        print("Adapter blocks text: ", self.args.adapter_blocks_text)


        ### Asserts to ensure compatible peft and metric calculation settings before training starts

        if sum(self.args.adapter_blocks_image) == 0:
            print("Checking assert for metric modality text matching no image adapters")
            assert (self.metric == "energy" or self.args.metric_modality == "text"), f"No image adapters are being used, but metric modality is not text"

        if sum(self.args.adapter_blocks_text) == 0:
            print("Checking assert for metric modality text matching no text adapters")
            assert (self.metric == "energy" or self.args.metric_modality == "image"), f"No text adapters are being used, but metric modality is not image"


        if self.args.peft_encoder == "text":
            print("Checking assert for metric modality text matching peft encoder")
            assert self.args.metric_modality in ["text", "hybrid"], f"peft encoder {self.args.peft_encoder} not compatible with metric modality {self.args.metric_modality}"

        elif self.args.peft_encoder == "image":
            print("Checking assert for metric modality image matching peft encoder")
            assert self.args.metric_modality in ["image", "hybrid"], f"peft encoder {self.args.peft_encoder} not compatible with metric modality {self.args.metric_modality}"

        elif self.args.peft_encoder == "none" or (sum(self.args.adapter_blocks_image) + sum(self.args.adapter_blocks_text)) == 0:
            print("Checking assert for metric energy")
            assert self.metric == "energy", "peft encoder set to None but metric requires adapter layers"

        if self.args.metric_modality in ["image", "both"]:
            print("Checking assert for block adapters image")
            for b in self.stored_blocks:
                assert self.args.adapter_blocks_image[b] == True, f"Image encoder {b} not set to use adapters but set for metrics calculation"

        if self.args.metric_modality in ["text", "both"]:
            print("Checking assert for block adapters text")
            for b in self.stored_blocks:
                assert self.args.adapter_blocks_text[b] == True, f"Text encoder {b} not set to use adapters but set for metrics calculation"




    
    ##############################################################################################
    ###  Merging Operations
    ##############################################################################################


    def get_removed_experts(self, idx, modal, block, subnets_to_merge):
        """
        Needs to take the set of experts, and depending on the implemented selection method either:
            1. Access the choose maps of the given block to dictate which experts to remove
            2. Access the stored activations and compute a metric to dictate which experts to remove

        Note: For now we are simply selecting the top-k experts where k is the number that would be frozen for a new subnet
        """

        all_experts = []
        for s in subnets_to_merge:
            all_experts.extend(block.frozen_experts[s])
        all_experts.sort()

        #!# Note: We need to preserve the choosemap values of other subnetworks in case we are making multiple merge operations
        # if modal == "visual":            
        #     choose_map = copy.deepcopy(block.choose_map_image)
        # else:
        choose_map = copy.deepcopy(block.choose_map)
        
        if idx == 0:
            print(f"Choosemap for modal {modal} is {choose_map}")

        other_experts = [i for i in range(len(choose_map)) if i not in all_experts]
        choose_map[other_experts] = -1

        if idx == 0:
            print(f"Masked choosemap is {choose_map}")
            # if modal == "visual":            
            #     print("Original: ", block.choose_map_image)
            # else:
            print("Original: ", block.choose_map)
            

        ### Convert expert choice counts into percentages, if we want to keep the experts responsible for some percentage of samples
        top_values_v, top_indices_v = torch.topk(choose_map, self.experts_per_subnet)

        #!# May need to make sure the format of expert indices matches properly here if issues arise
        experts_to_remove = [e for e in all_experts if e not in top_indices_v]
        experts_to_remove.sort()

        if idx == 0:
            print("Experts to remove ", experts_to_remove)


        return all_experts, experts_to_remove




    def merge_block(self, idx, modal, block, subnets_to_merge):
        """
        Needs to consider which experts are being removed and kept for the given block
        Then call the merging process within the clip model block itself
        Lastly update any method class variables and dicts accordingly before returning any needed values
        """
        all_experts, experts_to_remove = self.get_removed_experts(idx, modal, block, subnets_to_merge)

        expert_mappings = block.merge_subnets(subnets_to_merge, all_experts, experts_to_remove, verbose=False)


        """
        To update the class method dicts, we:
            1. Remove all stored frozen expert values for merged tasks
            2. Remap all of the remaining expert values to reflect their new list indices within the CLIP model block
            3. Store a new set of frozen experts for the subnetwork that resulted from the merging operation
        """
        
        ### 1. Remove frozen expert entries for removed subnetworks
        for i, s in enumerate(subnets_to_merge):
            ### For the remaining subnet we just reset it so that it maintains its position within the dictionary
            if i == 0:
                self.experts_index_by_subnet[s][modal][idx] = []
                self.experts_by_subnet[s][modal][idx] = []
            else:                
                del self.experts_index_by_subnet[s][modal][idx]
                del self.experts_by_subnet[s][modal][idx]


        ### 2. For all unmerged subnetworks, we need to remap the frozen expert idxs to reflect the updated contiguous values within the clip model
        for s in self.experts_by_subnet.keys():
            temp_expert_name_list, temp_index_list = [], []        

            if idx in self.experts_index_by_subnet[s][modal].keys():
                for e in self.experts_index_by_subnet[s][modal][idx]:
                    if modal == "visual":
                        mod_str = "visual."
                    else: 
                        mod_str = ""
                    temp_index_list.append(expert_mappings[e])
                    temp_expert_name_list.append(f'{mod_str}transformer.resblocks.{idx}.adaptmlp_list.{expert_mappings[e]}.down_proj.weight')
                    temp_expert_name_list.append(f'{mod_str}transformer.resblocks.{idx}.adaptmlp_list.{expert_mappings[e]}.down_proj.bias')
                    temp_expert_name_list.append(f'{mod_str}transformer.resblocks.{idx}.adaptmlp_list.{expert_mappings[e]}.up_proj.weight')
                    temp_expert_name_list.append(f'{mod_str}transformer.resblocks.{idx}.adaptmlp_list.{expert_mappings[e]}.up_proj.bias')

                    self.experts_index_by_subnet[s][modal][idx] = temp_index_list
                    self.experts_by_subnet[s][modal][idx] = temp_expert_name_list



        ### 3. Add in the new frozen experts for the remaining subnetwork from the merging operation        
        remaining_subnet = subnets_to_merge[0]
        frozen_expert_idxs = block.frozen_experts[remaining_subnet]
        temp_index_list = frozen_expert_idxs
        temp_expert_name_list = []

        for e in frozen_expert_idxs:
            if modal == "visual":
                mod_str = "visual."
            else: 
                mod_str = ""

            temp_expert_name_list.append(f'{mod_str}transformer.resblocks.{idx}.adaptmlp_list.{e}.down_proj.weight')
            temp_expert_name_list.append(f'{mod_str}transformer.resblocks.{idx}.adaptmlp_list.{e}.down_proj.bias')
            temp_expert_name_list.append(f'{mod_str}transformer.resblocks.{idx}.adaptmlp_list.{e}.up_proj.weight')
            temp_expert_name_list.append(f'{mod_str}transformer.resblocks.{idx}.adaptmlp_list.{e}.up_proj.bias')


        self.experts_index_by_subnet[remaining_subnet][modal][idx] = temp_index_list
        self.experts_by_subnet[remaining_subnet][modal][idx] = temp_expert_name_list




    def merge_subnetworks(self, subnets_to_merge):
        """
        1. Look over all residual blocks in the network
        2. Call the merge block function for each
             Merge block handles removal and remapping of experts for the given block, since each block has its own experts
        3. Remove the necessary subnet ids from dicts and variables as needed
        """
        self.temp_time = time.time()

        print("\n\n\n")
        print("-"*20)
        print("Merging Subnetworks ", subnets_to_merge)
        print("Pre-merge tasks by subnet: {}".format(self.tasks_by_subnet))
        print("Pre-merge values of dicts for 1st visual block of model:")
        for subnet in self.tasks_by_subnet.keys():
            print("Expert Indices for subnet {}: {}".format(subnet, self.experts_index_by_subnet[subnet]["visual"][0]))


        for i in range(len(self.model.clip_model.visual.transformer.resblocks)):
            if self.args.adapter_blocks_image[i] == True:
                self.merge_block(i, "visual", self.model.clip_model.visual.transformer.resblocks[i], subnets_to_merge)
            if self.args.adapter_blocks_text[i] == True:        
                self.merge_block(i, "text", self.model.clip_model.transformer.resblocks[i], subnets_to_merge)


        pooled_tasks = self.tasks_by_subnet[subnets_to_merge[0]]
        keys = list(self.tasks_by_subnet.keys())
        for key in keys:
            if key in subnets_to_merge[1:]:
                pooled_tasks.extend(self.tasks_by_subnet[key])
                del self.tasks_by_subnet[key]

        pooled_tasks.sort()
        self.tasks_by_subnet[subnets_to_merge[0]] = pooled_tasks

        for subnet in subnets_to_merge[1:]:
            del self.experts_by_subnet[subnet]
            del self.experts_index_by_subnet[subnet]



        print("\n\n\nPost-merge tasks by subnet: {}".format(self.tasks_by_subnet))
        print("Post-merge values of dicts for 1st visual block of model:")
        for subnet in self.tasks_by_subnet.keys():
            print("Expert Indices for subnet {}: {}".format(subnet, self.experts_index_by_subnet[subnet]["visual"][0]))

        print("")
        for subnet in self.tasks_by_subnet.keys():
            print("Expert names for subnet {}: {}".format(subnet, self.experts_by_subnet[subnet]["visual"][0]))


        print("-"*20)


        self.result_dicts[self.task_id]["times"]["merging"] += (time.time() - self.temp_time)
        self.temp_time = time.time()

        self.subnet = subnets_to_merge[0]
        self.finetune_subnetwork()



    ##############################################################################################
    ###  Clustering Operations
    ##############################################################################################


    def predict_acc_clusters(self):
        """
        Gets the accuracy data from activation collection process and predicts optimal clusters based on subnet accuracies
        """
        acc_path = os.path.join(self.log_dir, f"mem_acc_stats.pt")
        stat_dict = torch.load(acc_path, weights_only=False)
        accs_dict = stat_dict["ave_acc"] # dict layout: [tasks][subnets]

        acc_vects = []
        for s in accs_dict[0].keys():
            subnet_accs = []
            for t in accs_dict.keys():
                subnet_accs.append(accs_dict[t][s])
            acc_vects.append(subnet_accs)

        print("Acc vects for clustering: ", acc_vects, flush=True)
        kmeans =   KMeans(n_clusters=self.num_clusters, n_init='auto', random_state=0)
        clusters = kmeans.fit_predict(acc_vects)

        return clusters



    def get_metric_clusters(self, metric_dict):
        """
        Takes as input the metric_dict from get_metric_dict
        Processes metrics to get task-metric value vectors for each subnetwork
        Outputs a list of K-means cluster assignments for each subnetwork
        """
        subnet_vectors = []

        tasks = list(metric_dict.keys())
        subnets = list(metric_dict[tasks[0]].keys())
        num_blocks = len(metric_dict[tasks[0]][subnets[0]].keys())

        modalities = []
        if self.metric_modality in ["image", "hybrid"]:
            modalities.append("image")
        if self.metric_modality in ["text", "hybrid"]:
            modalities.append("text")

        if self.metric == "energy":
            num_blocks = 1
            modalities = ["image"]

        for subnet in subnets:
            subnet_vect = []
            ### We concatenate the per-task averages of metrics for both modalities to get a vector of length 2*T
            for branch in modalities:
                for task in tasks:
                    metrics_by_block = []
                    for block in range(num_blocks):
                        metrics_by_block.append(metric_dict[task][subnet][block][branch])

                    subnet_vect.append(torch.mean(torch.tensor(metrics_by_block)))

            subnet_vectors.append(subnet_vect)


        print("Clustering subnet vectors: ", subnet_vectors)
        ### Subnet vectors has shape (# subnets, 2*# tasks)
        kmeans =   KMeans(n_clusters=self.num_clusters, n_init='auto', random_state=0)
        clusters = kmeans.fit_predict(subnet_vectors)

        return clusters



    def get_metric_dict(self):
 
        metric = self.metric
        metric_order = self.metric_order
        layer_l = self.first_layer
        layer_k = self.second_layer
        num_blocks = self.num_blocks

        condition_by_class = self.condition_by_class
        remove_padding = self.remove_padding
        subsample_tokens = self.subsample_tokens


        condition = "cond_by_class" if condition_by_class else "none"
        padding = "remove_padding" if remove_padding else "none"
        fold_dir = "subsample_cls_token" if subsample_tokens else "fold_none"

        loadpath = self.log_dir
        # loadpath=f"./results/{self.dataset_name}/{self.n_tasks}TASKS/CMMoEMETHOD/{self.rnd_seed}SEED/CMMoE_{self.visible_classes}-both_SEED{self.rnd_seed}/"

        if self.metric_order == "first":
            save_path = os.path.join(loadpath,"metrics",metric_order, metric, f"{layer_l}-{layer_k}", 
                f"PREPROCESSING_{condition}_{padding}", fold_dir)

        else:
            save_path = os.path.join(loadpath,"metrics",metric_order, metric, f"{layer_l}", 
                f"PREPROCESSING_{condition}_{padding}", fold_dir)


        os.makedirs(save_path, exist_ok=True)
        save_file = f"{save_path}/metric_dict_{self.block_set}.pt"

        metric_results = {}

        if metric == "energy":
            ### Load information dicts shared by all tasks and subnets
            logit_path = os.path.join(loadpath, "logits.pt")
            logit_dict = torch.load(logit_path, map_location=torch.device('cpu'), weights_only=False) # Just has the [task][subnet][batch] logit values as 2d tensors of (#imgs, #classes)


        block_idxs = []
        for b in range(num_blocks):
            if self.block_set == "first":
                block_idxs.append(b)
            else:
                ### Count backwards from the last block (12 blocks for CLIP)
                start_block = 12 - num_blocks
                block_idxs.append(start_block + b)


        ### Note: Energy is arbitrarily caculated under "image" only to avoid duplicate calculation, since it uses the multimodal logits
        modalities = []
        if (self.metric_modality in ["image", "hybrid"] and self.args.peft_encoder in ["image", "both"]) or metric == "energy":
            modalities.append("image")
        if (self.metric_modality in ["text", "hybrid"] and self.args.peft_encoder in ["text", "both"]) and metric != "energy":
            modalities.append("text")

        for task in self.known_tasks:
            if task not in metric_results.keys():
                metric_results[task] = {}

            ### Load label dict containing information about labels and each classes text tokens
            labelpath = os.path.join(loadpath, f"task_{task}_mem_label_dict.pt")
            labels_dict = torch.load(labelpath, map_location=torch.device('cpu'), weights_only=False) # ['labels', 'labels_mapped', 'tokens', 'eot_indices']

            ### Using labels_mapped only, since it should behave identically when using exposed_classes with all known classes, and will match the text inputs when using batched_exposed_classes
            labels = torch.cat(labels_dict["labels_mapped"], dim=0)            

            eot_indices = []
            for batch in range(len(labels_dict["eot_indices"])):
                eot_indices.extend(labels_dict["eot_indices"][batch])

            #!# To make a fully accurate text_label tensor, we would need a function that constructs it using the unmapped labels of each batch to identify repeated classes
            ###     but then also have the conversions into names to handle redundant class labels. If it becomes necessary I will look into it more to replace this naive approach
            text_labels = torch.arange(len(eot_indices))


            ### For first order metrics, calculate between two layers in the same subnetwork
            if self.metric_order == "first":
                for subnet in self.tasks_by_subnet.keys():
                    print("Starting Subnet ", subnet, flush=True)
                    metric_results[task][subnet] = {}

                    # key_tsb = f"{task}_{subnet}"

                    ### Skip block loop and activation prep to just get logit energy scores
                    if metric == "energy":
                        ### no preprocessing is used on logits so we can just get the score directly
                        logits = logit_dict[task][subnet]
                        metric_results[task][subnet][0] = {}
                        metric_results[task][subnet][0]["image"] = metric_utils.calculate_energy(logits)
                    else:
                        for b in range(num_blocks):
                            metric_results[task][subnet][b] = {}
                            ### Get the block index that actually corresponds to the bth index we are calculating for
                            #!# Eventually this system should be scrapped for absolute indices since we are saving to a dict anyways
                            block = block_idxs[b]

                            # filepath = os.path.join(loadpath, "activations", key_tsb, f"acts_dict_block_{block}.pt")
                            filepath = os.path.join(loadpath, "activations", f"acts_dict_{task}.pt")

                            ### Within the loop, load task dict for one task, subnet, and block combination
                            acts_dict = torch.load(filepath, map_location=torch.device('cpu'), weights_only=False) # [modality_block][layer][batches]
                            # print("Acts dict loaded: ", acts_dict)
                            for branch in modalities:

                                ### Copy the target two layers l and l' and delete the original object for memory purposes
                                # branch_key = f"{branch}_{block}"
                                # acts_l = copy.deepcopy(acts_dict[branch_key][layer_l]) 
                                # acts_k = copy.deepcopy(acts_dict[branch_key][layer_k])
                                acts_l = copy.deepcopy(acts_dict[branch][subnet][block][layer_l]) 
                                acts_k = copy.deepcopy(acts_dict[branch][subnet][block][layer_k])
                                

                                ### Preprocess the two activation tensors based on which layers they are
                                ### Concatenate over batches. Experts need to be handled differently due to sample splitting across experts
                                if layer_l in ["expert_down_acts"]:
                                    acts_l = metric_utils.cat_experts(acts_l)
                                    acts_k = metric_utils.cat_experts(acts_k)
                                    acts_l = metric_utils.remove_empty_experts(acts_l)
                                    acts_k = metric_utils.remove_empty_experts(acts_k)
                                # else:
                                #     acts_l = metric_utils.cat_dict_acts(acts_l)
                                #     acts_k = metric_utils.cat_dict_acts(acts_k)
                                

                                temp_expert_results = []
                                final_metric = None
                                ### If the layers are expert layers, initialize an empty temp list and loop over all nonzero expert tensors:
                                if layer_l in ["expert_down_acts"]:
                                    ### Both layers in the subnet should have the same exact expert routing, and therefor the same # of experts used
                                    assert(len(acts_l) == len(acts_k))

                                    gates = acts_dict[branch][subnet][block]["gates"]
                                    gates = metric_utils.cat_dict_acts(gates)
                                    # print("Gates shape: ", gates.shape, " and labels length: ", len(labels))

                                    exp_labels = metric_utils.get_expert_labels(gates,labels)

                                    assert len(exp_labels) == len(acts_l), f"len of exp_labels {len(exp_labels)} does not match len of acts_l {len(acts_l)}"

                                    for exp in range(len(acts_l)):
                                        ### Calculate metric for each expert index separately and store results into the empty temp list
                                        metric_result = metric_utils.get_metric(acts_l[exp], acts_k[exp], branch, exp_labels[exp], self.args, eot_indices)
                                        temp_expert_results.append(metric_result)
                                    ### Average over all experts for the given task, subnet, block combination
                                    final_metric = torch.mean(torch.tensor(temp_expert_results)).item()

                                ### Else if the layers are shared (block input, dispatch combined, mlp output) directly calculate metric for the full layers
                                else:
                                    final_metric = metric_utils.get_metric(acts_l, acts_k, branch, labels, self.args, eot_indices).item()

                                assert(final_metric is not None)

                                ### Store metric into the empty results dictionary
                                metric_results[task][subnet][b][branch] = final_metric


            ### For second order metrics, calculate between two subnetworks using the same layer
            elif self.metric_order == "second":
                ### Since each subnet uses different experts we can't reasonably compare between subnets on these layers
                assert(layer_l not in ["expert_inputs", "expert_down_acts", "expert_up_acts"])

                ### Loop over each potentially Out-of-Distribution subnetwork and compare metric with ID subnetwork acts
                #!# This does include comparing the ID subnet against itself, which could be skipped for time saving but we include here
                for OoD_subnet in self.tasks_by_subnet.keys():
                    print("Starting Subnet ", OoD_subnet, flush=True)

                    ID_subnet = None
                    for s in self.tasks_by_subnet.keys():
                        if task in self.tasks_by_subnet[s]:
                            ID_subnet = s

                    # ID_key_tsb = f"{task}_{ID_subnet}"
                    # OoD_key_tsb = f"{task}_{OoD_subnet}"

                    if OoD_subnet not in metric_results[task].keys():
                        metric_results[task][OoD_subnet] = {}
                    

                    ### Skip block loop and activation prep to just get logit energy scores
                    if metric == "energy":
                        ### no preprocessing is used on logits so we can just get the score directly
                        metric_results[task][OoD_subnet][0] = {}
                        metric_results[task][OoD_subnet][0]["image"] = metric_utils.calculate_energy(logits)
                    else:
                        for b in range(num_blocks):
                            metric_results[task][OoD_subnet][b] = {}
                            block = block_idxs[b]

                            for branch in modalities:
                                # branch_key = f"{branch}_{block}"

                                # OoD_filepath = os.path.join(loadpath, "activations", OoD_key_tsb, f"acts_dict_block_{block}.pt")
                                # ID_filepath = os.path.join(loadpath, "activations", ID_key_tsb, f"acts_dict_block_{block}.pt")

                                # ## Within the loop, load task dict for one task, subnet, and block combination
                                # acts_dict = torch.load(OoD_filepath, map_location=torch.device('cpu'), weights_only=False)
                                # acts_OoD = copy.deepcopy(acts_dict[branch_key][layer_l])

                                # ## Reload the act dict using the ID subnet for the given task
                                # acts_dict = torch.load(ID_filepath, map_location=torch.device('cpu'), weights_only=False)
                                # acts_ID = copy.deepcopy(acts_dict[branch_key][layer_l])

                                # ## Preprocess the two activation tensors based on which layers they are
                                # ### Concatenate over batches. 
                                # acts_OoD = metric_utils.cat_dict_acts(acts_OoD)
                                # acts_ID  = metric_utils.cat_dict_acts(acts_ID)

                                filepath = os.path.join(loadpath, "activations", f"acts_dict_{task}.pt")
                                acts = torch.load(filepath, map_location=torch.device('cpu'), weights_only=False)
                                acts_ID = copy.deepcopy(acts[branch][ID_subnet][block][layer_l])
                                acts_OoD = copy.deepcopy(acts[branch][OoD_subnet][block][layer_l])


                                final_metric = None
                                final_metric = metric_utils.get_metric(acts_OoD, acts_ID, branch, labels, self.args, eot_indices).item()

                                assert(final_metric is not None)

                                ## Store metric into the empty results dictionary
                                metric_results[task][OoD_subnet][b][branch] = final_metric




        torch.save(metric_results, save_file)
        print("Saved to ", save_file, "\n\n")

        return metric_results















    def collect_mem_acts(self):

        self.print_memory(verbose=False)
        print("\n\n\n\nCollecting activations for all subnetworks on all known tasks", flush=True)
        self.model.clip_model.eval()

        assert self.visible_classes == 'all'

        acc_path = os.path.join(self.log_dir, f"mem_acc_stats.pt")
        logit_path = os.path.join(self.log_dir, f"logits.pt")
        task_acts_path = os.path.join(self.log_dir,"activations")
        os.makedirs(task_acts_path, exist_ok=True)

        # batches_to_record = self.act_batches_to_record
        max_act_samples = self.max_act_samples
        ### Prior to starting we reset all choosemaps so we can track expert use on memory samples
        #!# Note: We do this so that expert merging does not get biased by certain tasks having more or less overall samples
        # for i in range(len(self.model.clip_model.visual.transformer.resblocks)):
        #     self.model.clip_model.visual.transformer.resblocks[i].reset_choosemap()
        #     self.model.clip_model.transformer.resblocks[i].reset_choosemap()
        for name, module in self.model.clip_model.named_modules():
            if isinstance(module, ResidualAttentionBlock_MoA):
                module.reset_choosemap()


        stored_blocks = self.stored_blocks
        # stored_blocks = []
        # if self.block_set == "first":
        #     stored_blocks = list(range(self.num_blocks))
        # else:
        #     total_blocks = len(self.model.clip_model.transformer.resblocks)
        #     stored_blocks = list(range(total_blocks - self.num_blocks, total_blocks))


        total_correct, total_samples, ave_acc, logits_dict = {}, {}, {}, {}
    
        with torch.no_grad():
            for mem_task in self.known_tasks:

                self.model.clip_model.set_task(mem_task)
                label_path = os.path.join(self.log_dir, f"task_{mem_task}_mem_label_dict.pt")

                task_labels, task_labels_mapped, task_tokens, task_eot_indices = [], [], [], []
                total_correct[mem_task], total_samples[mem_task], ave_acc[mem_task] = {}, {}, {}
                logits_dict[mem_task] = {}

                ### Set up a dataloader from memory for only classes from the given mem_task
                memory_sampler = MemorySubnetSampler(self.memory, [mem_task], shuffle=False)
                self.memory_dataloader = DataLoader(self.train_dataset,
                                                    batch_size=self.memory_batchsize,
                                                    sampler=memory_sampler,
                                                    num_workers=self.n_worker)

                # print("Length of memory sampler: ", len(memory_sampler))
                ### Get text tokens for current task 
                if self.visible_classes == 'all':
                    train_class_name_list = self.train_dataset.classes_names_by_task[mem_task]
                    self.model.reset_class_names(train_class_name_list)
                    # print("Train class name list length: ", len(train_class_name_list))
                    # print("Train class name list: ", train_class_name_list)

                ### Set up the master dict for task act collection in resblocks
                sample_count = 0
                max_sample_count = min(max_act_samples, len(memory_sampler))
                peft.set_sample_count(sample_count)
                peft.prepare_acts_dict(mem_task, 
                                        stored_blocks, 
                                        subnets = list(self.tasks_by_subnet.keys()),
                                        num_samples = max_sample_count, 
                                        num_classes = len(train_class_name_list),
                                        store_layers = self.store_layers)



                ### For each subnet we get and store the text eot embeddings on the class labels of mem_task
                for subnet in self.tasks_by_subnet.keys():
                    for b in stored_blocks:
                        if self.args.adapter_blocks_text[b] == True:
                            self.model.clip_model.transformer.resblocks[b].set_activation_saving(True)
                    _ = self.model(None,self.model.text_tokens, is_train=False, val_subnet_id=subnet)



                for i, (x, y) in enumerate(self.memory_dataloader):
                    # if i == batches_to_record:
                    if sample_count + y.shape[0] > max_act_samples:
                        print("Breaking acts early", flush=True)
                        break


                    if self.visible_classes == 'batch':
                        self.add_new_class(y)
                        train_class_name_list = self.batch_exposed_classes_names
                        self.model.reset_class_names(train_class_name_list)


                    task_labels.append(y.clone())
                    task_tokens.append(self.model.text_tokens.clone())
                    task_eot_indices.append(self.model.eot_indices)

                    for j in range(len(y)):
                        label = y[j].item()
                        label_name = self.train_dataset.classes_names[label]
                        y[j] = train_class_name_list.index(label_name)

                    task_labels_mapped.append(y.clone())

                    x, y = x.to(self.device), y.to(self.device)

                    ### For each batch we get the image encoder activations
                    for subnet in self.tasks_by_subnet.keys():

                        for b in stored_blocks:
                            if self.args.adapter_blocks_image[b] == True:
                                self.model.clip_model.visual.transformer.resblocks[b].set_activation_saving(True)


                        mem_time = time.time()
                        if self.metric == "energy":
                            logit, _, _ = self.model(x,self.model.text_tokens, is_train=False, val_subnet_id=subnet, apply_softmax=False)
                            _, preds = logit.topk(self.topk, 1, True, True)

                            if subnet not in logits_dict[mem_task].keys():
                                logits_dict[mem_task][subnet] = [logit]
                            else:
                                logits_dict[mem_task][subnet].append(logit)

                        else:
                            image_embeddings = self.model(x, None, is_train=False, val_subnet_id=subnet)
                            # text_embeddings =  self.model(None, self.model.text_tokens, is_train=False, val_subnet_id=subnet)
                            # logit = self.model.clip_model.logit_scale.exp() * image_embeddings @ text_embeddings.t() 
                            # _, preds = logit.topk(self.topk, 1, True, True)



                        if subnet not in ave_acc[mem_task].keys():
                            # total_correct[mem_task][subnet]  = torch.sum(preds == y.unsqueeze(1)).item()
                            total_samples[mem_task][subnet] = y.size(0)
                            print(f"Storing subnet {subnet} into acc dict for task {mem_task}.")
                        else:
                            # total_correct[mem_task][subnet]  += torch.sum(preds == y.unsqueeze(1)).item()
                            total_samples[mem_task][subnet] += y.size(0)

                        # ave_acc[mem_task][subnet] = total_correct[mem_task][subnet] / total_samples[mem_task][subnet]
                        ave_acc[mem_task][subnet] = -1 / total_samples[mem_task][subnet]

                    ### Update counters for proper act storage indexing in resblocks
                    sample_count += y.shape[0]
                    peft.set_sample_count(sample_count)

                    # self.print_memory(verbose=False)
                    # print("FDs for memtask ", mem_task, ":", len(os.listdir("/proc/self/fd")))

                ### Save the accummulated activation values for the given task's samples
                task_acts_dict = peft.get_acts_dict()
                # print("Task acts dict type: ", type(task_acts_dict))
                # print("Task acts dict: ", task_acts_dict)

                task_acts_file = os.path.join(task_acts_path, f"acts_dict_{mem_task}.pt")
                torch.save(task_acts_dict, task_acts_file)

                task_label_dict = {"labels": task_labels, "labels_mapped": task_labels_mapped, "tokens": task_tokens, "eot_indices": task_eot_indices}
                torch.save(task_label_dict, label_path)




        ### Store the tokens of text label inputs and the per-subnet accuracy stats
        stat_dict = {"total_correct": total_correct, "total_samples": total_samples, "ave_acc": ave_acc}
        torch.save(stat_dict, acc_path)
        torch.save(logits_dict, logit_path)

        choosemaps = {"image": {}, "text": {}}
        for i in range(len(self.model.clip_model.visual.transformer.resblocks)):
            if self.args.adapter_blocks_image[i] == True:
                choosemaps["image"][i] = self.model.clip_model.visual.transformer.resblocks[i].choose_map
            if self.args.adapter_blocks_text[i] == True:
                choosemaps["text"][i] = self.model.clip_model.transformer.resblocks[i].choose_map
        self.result_dicts[self.task_id]["metrics"]["clustering choosemaps"] = choosemaps



    def cluster_and_merge(self):

        num_tasks = len(self.known_tasks)
        num_subnets = len(self.tasks_by_subnet.keys())
        num_blocks = self.num_blocks

        ######################################
        ### Getting subnet activations
        ######################################
        self.temp_time = time.time()
        
        self.collect_mem_acts()    

        self.result_dicts[self.task_id]["times"]["collect acts"] += (time.time() - self.temp_time)
        self.temp_time = time.time()
        

        ######################################
        ### Getting metrics and clusters
        ######################################

        ### Collect metric for stored activations
        metric_dict = self.get_metric_dict()
        for key in metric_dict.keys():
            print(metric_dict[key])


        self.result_dicts[self.task_id]["times"]["metric calculation"] += (time.time() - self.temp_time)
        self.temp_time = time.time()

        ### Determine predicted "optimal clustering decisions" and store for current task
        # print("Before Acc cluster calculation", flush=True)
        # predicted_clusters = self.predict_acc_clusters()
        # self.result_dicts[self.task_id]["clustering"]["predicted clusters"] = predicted_clusters

        ### Make clustering decisions on metric vectors
        ### Get the cluster values for each subnetwork
        print("Before cluster calculation")
        clusters = self.get_metric_clusters(metric_dict)
        self.result_dicts[self.task_id]["clustering"]["merged clusters"] = clusters

        self.result_dicts[self.task_id]["times"]["clustering"] += (time.time() - self.temp_time)
        self.temp_time = time.time()

        ######################################
        ### Merging
        ######################################

        print("\n\n\n\n\n\n\n\nStarting Merging\n")
        subnets_precluster = list(self.tasks_by_subnet.keys())
        print("Subnets pre-clustering: ", subnets_precluster)

        ### Apply merging on clusters
        for c in range(max(clusters)+1):
            cluster_idxs = np.where(clusters == c)[0]
            if len(cluster_idxs) == 1:
                continue

            print("Cluster indices: ", cluster_idxs)
            subnets_to_merge = [subnets_precluster[c] for c in cluster_idxs]
            print(f"Subnets belonging to cluster {c}: {subnets_to_merge}")
            self.merge_subnetworks(subnets_to_merge)





        #!# Potentially temporary solution, may be best to store each merge in a unique path instead
        acts_path = os.path.join(self.log_dir, "activations")
        metrics_path = os.path.join(self.log_dir, "metrics")
        shutil.rmtree(acts_path)
        shutil.rmtree(metrics_path)

        os.remove(os.path.join(self.log_dir, "logits.pt"))
        for task in self.known_tasks:
            os.remove(os.path.join(self.log_dir, f"task_{task}_mem_label_dict.pt"))














    ##############################################################################################
    ###  Task Training Operations
    ##############################################################################################




    def finetune_subnetwork(self):
        print("Beginning Finetuning")
        self.temp_time = time.time()
        ### Reset frozen parameters
        for k, v in self.model.named_parameters():
            v.requires_grad = True
        self.freeze_experts()


        self.update_schedule(reset=True)
        self.reset_opt()
        
        self.model.clip_model.set_subnet(self.subnet)
        self.model.train()


        ### Set up a dataloader from memory for only classes from the given mem_task
        memory_sampler = MemorySubnetSampler(self.memory, self.tasks_by_subnet[self.subnet], shuffle=True)
        self.memory_dataloader = DataLoader(self.train_dataset,
                                            batch_size=self.memory_batchsize,
                                            sampler=memory_sampler,
                                            num_workers=self.n_worker)

        total_loss, total_correct, total_num_data = 0.0, 0.0, 0.0
        for epoch in range(self.finetune_epochs):
            print(f"Beginning epoch {epoch}")
            for b, (images, labels) in enumerate(self.memory_dataloader):

                ### Resets the exposed batch classes + text tokens for each batch
                self.add_new_class(labels)  
                if self.visible_classes == 'batch':
                    train_class_list = self.batch_exposed_classes
                    train_class_name_list = self.batch_exposed_classes_names
                    self.model.reset_class_names(train_class_name_list)
                else:
                    train_class_list = self.exposed_classes
                    train_class_name_list = self.exposed_classes_names
                    self.model.update_class_names(train_class_name_list)

                text_tokens = self.model.text_tokens
  
                # for j in range(len(labels)):
                #     labels[j] = train_class_list.index(labels[j].item())
                for j in range(len(labels)):
                    label = labels[j].item()
                    label_name = self.train_dataset.classes_names[label]
                    labels[j] = train_class_name_list.index(label_name)


                images,labels = images.to(self.device), labels.to(self.device)


                ### Train clip model adapters
                with torch.amp.autocast('cuda', enabled=self.use_amp):            
                    logit, image_features, text_features = self.model(images, text_tokens)
                    loss = self.criterion(logit, labels)
                _, preds = logit.topk(self.topk, 1, True, True)

                ### Accumulate gradients with the online batch and memory batch before zeroing again
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()


                total_loss += loss.item()
                total_correct += torch.sum(preds == labels.unsqueeze(1)).item()
                total_num_data += labels.size(0)
                
            self.report_training(total_num_data, total_loss / total_num_data, total_correct / total_num_data, step="finetuning")

            self.update_schedule()

        self.result_dicts[self.task_id]["times"]["finetuning"] += (time.time() - self.temp_time)
        self.temp_time = time.time()






    def offline_evaluate(self, test_task, use_current_subnet=False, add_task=False):
        """
        Returns the average test accuracy for a given task.
        If args.unknown_test_task_id is set to True, will predict the task ID and use the appropriate subnetwork        
        """
        print("Running offline eval for test task: ", test_task)

        if add_task == True:
            if test_task not in self.known_tasks:
                self.known_tasks.append(test_task)

        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        label, pred_list = [], []

        #!# Currently this can cause issues if not specifying an existing task. Will make a new subnet idx without initializing the subnet

        if use_current_subnet == True:
            ID_subnet, task_id = self.subnet, test_task
        else:
            if self.unknown_test_task_id == True:
                ID_subnet, task_id = self.select_subnet(task_id = None)
            else:
                ID_subnet, task_id = self.select_subnet(task_id = test_task)


        self.model.eval()
        text_embeddings = None

        with torch.no_grad():
            for i, data in enumerate(self.test_dataloader):
                x, y = data

                if self.debug and (i + 1) * self.temp_batchsize >= 400:
                    # print("Temp test batch size ", self.temp_batchsize, " images shape: ", images.shape)
                    break
                
                self.add_new_class(y, mode="test")

                new_classes_added = False
                if self.visible_classes == 'batch':
                    test_class_list = self.batch_exposed_classes
                    test_class_name_list = self.batch_exposed_classes_names
                    new_classes_added = self.model.reset_class_names(test_class_name_list)
                else:
                    test_class_list = self.exposed_classes
                    test_class_name_list = self.exposed_classes_names
                    new_classes_added = self.model.reset_class_names(test_class_name_list)


                ### Makes a contiguous set of labels to match the order of class names input into the text encoder
                # for j in range(len(y)):
                    # y[j] = test_class_list.index(y[j].item())
                for j in range(len(y)):
                    label = y[j].item()
                    label_name = self.test_dataset.classes_names[label]
                    y[j] = test_class_name_list.index(label_name)

                x,y = x.to(self.device), y.to(self.device)

                text_tokens = self.model.text_tokens

                self.model.clip_model.set_task(task_id)
                self.model.clip_model.set_subnet(ID_subnet)

                ### Evaluate using model set to the identified best-fit known task routing
                #*# Needs to be changed to reflect subnet use
                # logit, _, _ = self.model(x, text_tokens, is_train=False, val_subnet_id=ID_subnet)
                ### Since model is not being updated we only calculate text features if classes change
                if (new_classes_added == True) or (self.visible_classes == "batch") or (i == 0): 
                    # print("Calculating text embeddings for evaluation")
                    text_embeddings = self.model(None,text_tokens,is_train=False,val_subnet_id=ID_subnet)
                    text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

                image_embeddings = self.model(x,None,is_train=False,val_subnet_id=ID_subnet)
                image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
    
                logits_per_image = self.model.clip_model.logit_scale.exp() * image_embeddings @ text_embeddings.t()
                logit = logits_per_image.softmax(dim=-1)



                # pred = torch.argmax(logit, dim=-1)
                _, preds = logit.topk(self.topk, 1, True, True)
                total_correct += torch.sum(preds == y.unsqueeze(1)).item()
                total_num_data += y.size(0)

                # xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(y, pred)
                # correct_l += correct_xlabel_cnt.detach().cpu()
                # num_data_l += xlabel_cnt.detach().cpu()

                # label += y.tolist()
                # pred_list += pred.tolist()

        total_acc = total_correct / total_num_data

        return total_acc






    #!# Currently set up as a placeholder that relies on task id instead of using the autoencoders for each task
    def select_subnet(self, task_id=None, allow_zeroshot=False):
        if task_id is None:
            if len(self.autoencoders) == 0:
                task_id = 0
            else:
                x,y,i = next(iter(self.train_dataloader))
                x = x.to(self.device)
                # predict batch image domain:
                input_to_ae = self.feature_extractor(x)
                input_to_ae = F.sigmoid(input_to_ae.view(input_to_ae.size(0), -1))

                encoder_criterion = nn.MSELoss()
                
                model_autoencoder = self.autoencoders[0]
                outputs = model_autoencoder(input_to_ae)
                best_l = encoder_criterion(outputs, input_to_ae)
                best_task_fit = 0
                if len(self.autoencoders) > 1:
                    for i in range(1, len(self.autoencoders)):
                        outputs = self.autoencoders[i](input_to_ae)
                        new_l = encoder_criterion(outputs, input_to_ae)
                        if new_l < best_l:
                            best_l = new_l
                            best_task_fit = i

                ### Changed to defer checking if zero-shot CLIP should be used until after all known tasks are checked
                if best_l > self.threshold and allow_zeroshot == True:
                    best_task_fit = -1

                task_id = best_task_fit

            ### Currently setting to use known task ids while we improve the method
            task_id = test_task
            ID_subnet = None
            for key in self.tasks_by_subnet:
                if task_id in self.tasks_by_subnet[key]:
                    ID_subnet = key

            assert(ID_subnet is not None)


        ### Try finding the subnetwork responsible for this task, or create a new one
        subnetwork = -1
        for subnet, task_list in self.tasks_by_subnet.items():
            print("Checking subnet {} and task_list {}".format(subnet, task_list))
            if task_id in task_list:
                subnetwork = subnet
        
        if len(self.tasks_by_subnet.keys()) == 0:
            subnetwork = 0
            self.tasks_by_subnet[0] = [task_id]
        elif subnetwork == -1:
            subnetwork = max(self.tasks_by_subnet.keys()) + 1
            self.tasks_by_subnet[subnetwork] = [task_id]

        print("Selecting subnetwork: ", subnetwork, " for task ID: ", task_id)
        return subnetwork, task_id



    def freeze_experts(self):
        ### Produce a list of all experts frozen for subnetworks other than the current one
        frozen_experts = []
        for key in self.experts_by_subnet.keys():
            ### This prevents revisited tasks from training experts frozen in later tasks, alternatively may want to use "key < task_id"
            if key != self.subnet:
                for modal in ["text", "visual"]:
                    for block in self.experts_by_subnet[key][modal].keys():
                        frozen_experts.extend(self.experts_by_subnet[key][modal][block])
        print("The number of frozen experts is: ", len(frozen_experts))


        ### Note: This doesn't account for potential subnetwork sharing, as it is not considered in this method
        ### Freeze backbone and frozen experts
        for k, v in self.model.clip_model.named_parameters():
            if k in frozen_experts:
                v.requires_grad = False
            if "adaptmlp" not in k and "router" not in k and "noise" not in k:
                v.requires_grad = False


        # print("\nTrainable parameters:")
        # for name, param in self.model.clip_model.named_parameters():
        #     if "adaptmlp" in name and param.requires_grad:
        #         print(name)





    ### Prior to task, set task ID in model and initialize new expert mask, task routers, and task autoencoder
    def online_before_task(self, task_id=None, manual_subnet=None):

        self.batch_counter = 0

        print("Starting online_before_task", flush=True)

        ### During first task we set which layers to store acts for
        if self.set_store_layers_flag == False:
            metric_layers = []
            if self.experiment_type == "metric_calculation":
                for m in range(len(self.metric_suite)):
                    metric_info = self.metric_suite[m]
                    if metric_info["first_layer"] not in metric_layers:
                        metric_layers.append(metric_info["first_layer"])
                    if metric_info["metric_order"] == "second" and metric_info["second_layer"] not in metric_layers:
                        metric_layers.append(metric_info["second_layer"])
                    
            else:
                metric_layers.append(self.first_layer)
                if self.metric_order == "first":
                    metric_layers.append(self.second_layer)

            print("Storing activations for metric layers: ", metric_layers)
            # for i in range(len(self.model.clip_model.visual.transformer.resblocks)):
            # for i in range(len(self.args.adapter_blocks_image)):
            #     if self.args.adapter_blocks_image[i] == True:
            #         self.model.clip_model.visual.transformer.resblocks[i].store_layers.extend(metric_layers)
            #     if self.args.adapter_blocks_text[i] == True:    
            #         self.model.clip_model.transformer.resblocks[i].store_layers.extend(metric_layers)
            for name, module in self.model.clip_model.named_modules():
                if isinstance(module, ResidualAttentionBlock_MoA):
                    module.store_layers.extend(metric_layers)



            self.store_layers = metric_layers
            self.set_store_layers_flag = True


        # for i in range(len(self.model.clip_model.visual.transformer.resblocks)):
        # for i in range(len(self.args.adapter_blocks_image)):
        #     if self.args.adapter_blocks_image[i] == True:
        #         self.model.clip_model.visual.transformer.resblocks[i].reset_choosemap()
        #     if self.args.adapter_blocks_text[i] == True:
        #         self.model.clip_model.transformer.resblocks[i].reset_choosemap()
        for name, module in self.model.clip_model.named_modules():
            if isinstance(module, ResidualAttentionBlock_MoA):
                module.reset_choosemap()



        ### Deferred assignment since self.device isnt set up at init
        self.feature_extractor.to(self.device)



        if manual_subnet is not None:
            ### Manual subnets may be used to allow subnetwork evaluation on a specific OoD task
            subnet = manual_subnet
        elif self.unknown_train_task_id == True:
            subnet, task_id = self.select_subnet(task_id=None)
        else:
            subnet, task_id = self.select_subnet(task_id=task_id)

        if task_id not in self.autoencoders.keys():
            print("Setting up new autoencoder for task: ", task_id)
            self.autoencoders[task_id] = AutoEncoder()
        self.optimizer_encoder = optim.Adam(self.autoencoders[task_id].parameters(), lr=0.003, weight_decay=0.0001)



        self.subnet = subnet
        self.memory.set_task(task_id)
        if task_id not in self.known_tasks:
            self.known_tasks.append(task_id)


        print(f"\n\n Setting clip model to subnetwork {subnet} for task {task_id}")
        ### Telling the model what task is being trained and which subnet is being used for it. Used for routing and activation debugging
        self.model.clip_model.set_task(task_id)
        self.model.clip_model.set_subnet(subnet)



        ### Add experts and router for new task and corresponding dictionary entries to track subnetwork-dedicated experts
        if subnet not in self.experts_by_subnet.keys():
            self.experts_by_subnet[subnet] = {"text":{}, "visual":{}}
            self.experts_index_by_subnet[subnet] = {"text":{}, "visual":{}}
            for i in range(len(self.model.clip_model.visual.transformer.resblocks)):
                self.experts_by_subnet[subnet]['text'][i] = []
                self.experts_by_subnet[subnet]['visual'][i] = []
                self.experts_index_by_subnet[subnet]['text'][i] = []
                self.experts_index_by_subnet[subnet]['visual'][i] = []
                ### Set up new routers for new task in all transformer residual blocks
                # self.model.clip_model.visual.transformer.resblocks[i].init_subnet(subnet)
                # self.model.clip_model.transformer.resblocks[i].init_subnet(subnet)

            ### Set up new routers for new task in all transformer residual blocks
            for name, module in self.model.clip_model.named_modules():
                if isinstance(module, ResidualAttentionBlock_MoA):
                    module.init_subnet(subnet)
        
            self.model.to(self.device)


        self.freeze_experts()

        logger.info("Total parameters:\t{}".format(sum(p.numel() for p in self.model.parameters())))
        logger.info("Trainable parameters:\t{}".format(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))

        self.reset_opt()


        #!# We should eventually consider re-adding the gradient accumulation with memory batches during training
        ###   It would be potentially useful for preventing merged subnets from forgetting
        ###   As a counter-argument though, in an online setting, forgetting may be a positive, as distribution shift causes
        ###      older merged tasks' samples may poorly reflect the current state of the domain's features in the 'world'.
        # if len(self.memory) > 0 and len(self.tasks_by_subnet[self.subnet]) > 1:



        print("Ending online_before_task", flush=True)



    def online_step(self, images, labels, idx):
        self.temp_time = time.time()

        self.add_new_class(labels)

        self.batch_counter += 1

        self.result_dicts[self.task_id]["times"]["label management"] += (time.time() - self.temp_time)
        self.temp_time = time.time()


        # self.result_dicts[self.task_id]["times"]["memory management"] += (time.time() - self.temp_time)
        # self.temp_time = time.time()




        _loss, _acc, _iter = 0.0, 0.0, 0
        for _ in range(int(self.online_iter)):
            self.temp_time = time.time()
            loss, acc = self.online_train([images, copy.deepcopy(labels)])

            #!# I am unsure currently if it would be best to change this to only allow accuracy updates on first iteration, to strictly track evaluation of unseen streamed samples
            _loss += loss
            _acc += acc
            _iter += 1


        self.temp_time = time.time()

        if self.label_type == "caption":
            ### Decided not to implement this setting for now but leaving this code in case we do eventually
            raise ValueError
            # pseudolabels = torch.tensor([self.task_id]*len(idx))
            # pseudonames = torch.tensor([""*len(idx)])
            # assert len(pseudolabels) == len(idx), "Incorrect creation of pseudolabels in online step for captions"
            # self.update_memory(idx, pseudolabels, pseudonames)
        else:
            label_names = [self.train_dataset.classes_names[lab] for lab in labels]
            self.update_memory(idx, labels, label_names)

        self.result_dicts[self.task_id]["times"]["memory management"] += (time.time() - self.temp_time)
        self.temp_time = time.time()


        return _loss / _iter, _acc / _iter



    def online_train(self, data):
        self.model.train()
        total_loss, total_correct, total_num_data, total_mem_loss, total_mem_correct, total_num_mem_data = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        if self.visible_classes == 'batch':
            # train_class_list = self.batch_exposed_classes
            train_class_name_list = self.batch_exposed_classes_names
            self.model.reset_class_names(train_class_name_list)
        else:
            # train_class_list = self.exposed_classes
            train_class_name_list = self.exposed_classes_names
            self.model.reset_class_names(train_class_name_list)
            # self.model.update_class_names(train_class_name_list)

        x, y = data

        # ### Makes a contiguous set of labels to match the order of class names input into the text encoder
        # for j in range(len(y)):
        #     y[j] = train_class_list.index(y[j].item())
        #!# Rather than use the class labels, we make the labels contiguous directly based on the text label input to allow for redundant class idxs
        for j in range(len(y)):
            label = y[j].item()
            label_name = self.train_dataset.classes_names[label]
            y[j] = train_class_name_list.index(label_name)

        x,y = x.to(self.device), y.to(self.device)

        self.result_dicts[self.task_id]["times"]["tokenization"] += (time.time() - self.temp_time)
        self.temp_time = time.time()

        current_autoencoder = self.autoencoders[self.task_id]

        ### Train autoencoder only on the current task data for later task re-identification 
        input_to_ae = self.feature_extractor(x)
        input_to_ae = F.sigmoid(input_to_ae.view(input_to_ae.size(0), -1).to(self.device))

        self.optimizer_encoder = exp_lr_scheduler(self.optimizer_encoder, self.batch_counter, 0.01)
        self.optimizer_encoder.zero_grad()
        current_autoencoder.zero_grad()

        current_autoencoder.to(self.device)
        outputs = current_autoencoder(input_to_ae)

        encoder_criterion = nn.MSELoss()
        loss_autoencoder = encoder_criterion(outputs, input_to_ae)
        loss_autoencoder.backward()
        self.optimizer_encoder.step()



        self.result_dicts[self.task_id]["times"]["training autoencoder"] += (time.time() - self.temp_time)
        self.temp_time = time.time()


        text_tokens = self.model.text_tokens

        ### Train clip model adapters
        with torch.amp.autocast('cuda', enabled=self.use_amp):  
            # with torch.no_grad():
            #     logit_eval, _, _ = self.model(x, text_tokens, is_train=False, val_subnet_id=self.subnet)
            logit, image_features, text_features = self.model(x, text_tokens)
            loss = self.criterion(logit, y)

        # _, preds_eval = logit_eval.topk(self.topk, 1, True, True)
        _, preds = logit.topk(self.topk, 1, True, True)

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # temp = {}
        # y = y.detach().cpu()
        # preds = preds.detach().cpu()
        # for l in range(y.shape[0]):
        #     if y[l].item() not in temp.keys():
        #         temp[y[l].item()] = [preds[l].item()]
        #     else:
        #         temp[y[l].item()].append(preds[l].item())
        # print("Preds dict: ", temp) 


        ### Only update scheduler once the memory batch has been run
        # if self.args.get('grad_analysis', False):
        #     self._grad_analysis(image_features.clone().detach(),
        #                         text_features.clone().detach(),
        #                         y.clone().detach(), train_class_list)

        total_loss += loss.item()
        # total_correct += torch.sum(preds_eval == y.unsqueeze(1)).item()
        total_correct += torch.sum(preds == y.unsqueeze(1)).item()
        total_num_data += y.size(0)




        self.update_schedule()

        self.result_dicts[self.task_id]["times"]["training model"] += (time.time() - self.temp_time)
        self.temp_time = time.time()


        return total_loss, total_correct / total_num_data




    def online_after_task(self, task_id):
        
        self.temp_time = time.time()
        print("Starting online_after_task", flush=True)

        print("Memory memory_per_class: ", self.memory.memory_per_class, " and total size ", self.memory.memory_size)
        # print("Memory class seen count: ", self.memory.seen)


        print("Shape of exposed text tokens: ", self.model.text_tokens.shape)

        # sorted_counts = dict(sorted(self.memory.cls_count.items(), key=lambda item: item[1], reverse=True))
        # print("Memory class count: ", sorted_counts)

        choosemaps = {"image": {}, "text": {}}
        
        ### Assign the most-used experts for the current task for freezing
        for modal in ["image", "text"]:
            if modal == "image":
                adapter_blocks = self.args.adapter_blocks_image
                resblocks = self.model.clip_model.visual.transformer.resblocks
            else:
                adapter_blocks = self.args.adapter_blocks_text
                resblocks = self.model.clip_model.transformer.resblocks

            for i in range(len(adapter_blocks)):
                if adapter_blocks[i] == True:
                    print(f"Updating experts for transformer block {str(i)}", flush=True)

                    choose_map = copy.deepcopy(resblocks[i].choose_map)
        

                    choosemaps[modal][i] = resblocks[i].choose_map

                    ### Get the set of already-frozen experts to avoid reuse            
                    frozen_experts = resblocks[i].frozen_experts["all"]
                                
                    if i == 0:
                        print("Initial choosemaps: ", choose_map)
                        print("Masking frozen experts: ", frozen_experts)

                    choose_map[frozen_experts] = -1

                    if i == 0:
                        print("Masked choosemaps: ", choose_map)

                    # top_values_v, top_indices_v = torch.topk(visual_choose_map, 3)
                    # top_values_t, top_indices_t = torch.topk(text_choose_map, 3)
                    top_values, top_indices = torch.topk(choose_map, self.experts_per_subnet)

                    #!# This is a clunky placeholder until I can change code elsewhere to work with "image" key
                    modal_key = "text"
                    if modal == "image":
                        modal_key = "visual"


                    indices_for_task = []
                    for j in range(len(top_indices)):
                        indices_for_task.append(top_indices[j].item())
                        if modal == "image":
                            self.experts_by_subnet[self.subnet]["visual"][i].append('visual.transformer.resblocks.{}.adaptmlp_list.{}.down_proj.weight'.format(i,top_indices[j]))
                            self.experts_by_subnet[self.subnet]["visual"][i].append('visual.transformer.resblocks.{}.adaptmlp_list.{}.down_proj.bias'.format(i,top_indices[j]))
                            self.experts_by_subnet[self.subnet]["visual"][i].append('visual.transformer.resblocks.{}.adaptmlp_list.{}.up_proj.weight'.format(i,top_indices[j]))
                            self.experts_by_subnet[self.subnet]["visual"][i].append('visual.transformer.resblocks.{}.adaptmlp_list.{}.up_proj.bias'.format(i,top_indices[j]))
                        else:
                            self.experts_by_subnet[self.subnet]["text"][i].append('transformer.resblocks.{}.adaptmlp_list.{}.down_proj.weight'.format(i, top_indices[j]))
                            self.experts_by_subnet[self.subnet]["text"][i].append('transformer.resblocks.{}.adaptmlp_list.{}.down_proj.bias'.format(i, top_indices[j]))
                            self.experts_by_subnet[self.subnet]["text"][i].append('transformer.resblocks.{}.adaptmlp_list.{}.up_proj.weight'.format(i, top_indices[j]))
                            self.experts_by_subnet[self.subnet]["text"][i].append('transformer.resblocks.{}.adaptmlp_list.{}.up_proj.bias'.format(i, top_indices[j]))

                    ### Tracks each task's indices as individual lists so they can be differentiated during merging
                    self.experts_index_by_subnet[self.subnet][modal_key][i].extend(indices_for_task)

                    ### Report frozen expert indices back to the CLIP adapter layer
                    resblocks[i].frozen_experts[self.subnet].extend(indices_for_task)

                    for j in indices_for_task:
                        if j not in resblocks[i].frozen_experts['all']:
                            resblocks[i].frozen_experts['all'].append(j)

                    if i == 0:
                        print(f"Reporting frozen expert indices for modal {modal} block 0: {indices_for_task}")





        # ### Assign the most-used experts for the current task for freezing
        # for i in range(len(self.model.clip_model.visual.transformer.resblocks)):
        #     print(f"Updating experts for transformer block {str(i)}", flush=True)

        #     visual_choose_map = copy.deepcopy(self.model.clip_model.visual.transformer.resblocks[i].choose_map_image)
        #     text_choose_map = copy.deepcopy(self.model.clip_model.transformer.resblocks[i].choose_map_text)

        #     ### Get the set of already-frozen experts to avoid reuse            
        #     visual_frozen_experts = self.model.clip_model.visual.transformer.resblocks[i].frozen_experts["all"]
        #     text_frozen_experts   = self.model.clip_model.transformer.resblocks[i].frozen_experts["all"]
                        
        #     if i == 0:
        #         print("Initial choosemaps: ", visual_choose_map, " ", text_choose_map)
        #         print("Masking frozen experts: ", visual_frozen_experts, " ", text_frozen_experts)

        #     visual_choose_map[visual_frozen_experts] = -1
        #     text_choose_map[text_frozen_experts] = -1

        #     if i == 0:
        #         print("Masked choosemaps: ", visual_choose_map, " ", text_choose_map)

        #     # top_values_v, top_indices_v = torch.topk(visual_choose_map, 3)
        #     # top_values_t, top_indices_t = torch.topk(text_choose_map, 3)
        #     top_values_v, top_indices_v = torch.topk(visual_choose_map, self.experts_per_subnet)
        #     top_values_t, top_indices_t = torch.topk(text_choose_map, self.experts_per_subnet)

        #     indices_for_task, visual_indices_for_task = [], []
        #     for j in range(len(top_indices_v)):
        #         visual_indices_for_task.append(top_indices_v[j].item())
        #         self.experts_by_subnet[self.subnet]["visual"][i].append('visual.transformer.resblocks.{}.adaptmlp_list.{}.down_proj.weight'.format(i,top_indices_v[j]))
        #         self.experts_by_subnet[self.subnet]["visual"][i].append('visual.transformer.resblocks.{}.adaptmlp_list.{}.down_proj.bias'.format(i,top_indices_v[j]))
        #         self.experts_by_subnet[self.subnet]["visual"][i].append('visual.transformer.resblocks.{}.adaptmlp_list.{}.up_proj.weight'.format(i,top_indices_v[j]))
        #         self.experts_by_subnet[self.subnet]["visual"][i].append('visual.transformer.resblocks.{}.adaptmlp_list.{}.up_proj.bias'.format(i,top_indices_v[j]))
        #     for k in range(len(top_indices_t)):
        #         indices_for_task.append(top_indices_t[k].item())
        #         self.experts_by_subnet[self.subnet]["text"][i].append('transformer.resblocks.{}.adaptmlp_list.{}.down_proj.weight'.format(i, top_indices_t[k]))
        #         self.experts_by_subnet[self.subnet]["text"][i].append('transformer.resblocks.{}.adaptmlp_list.{}.down_proj.bias'.format(i, top_indices_t[k]))
        #         self.experts_by_subnet[self.subnet]["text"][i].append('transformer.resblocks.{}.adaptmlp_list.{}.up_proj.weight'.format(i, top_indices_t[k]))
        #         self.experts_by_subnet[self.subnet]["text"][i].append('transformer.resblocks.{}.adaptmlp_list.{}.up_proj.bias'.format(i, top_indices_t[k]))

        #     ### Tracks each task's indices as individual lists so they can be differentiated during merging
        #     self.experts_index_by_subnet[self.subnet]["visual"][i].extend(visual_indices_for_task)
        #     self.experts_index_by_subnet[self.subnet]["text"][i].extend(indices_for_task)

        #     ### Report frozen expert indices back to the CLIP adapter layer
        #     self.model.clip_model.visual.transformer.resblocks[i].frozen_experts[self.subnet].extend(visual_indices_for_task)
        #     self.model.clip_model.transformer.resblocks[i].frozen_experts[self.subnet].extend(indices_for_task)

        #     for j in visual_indices_for_task:
        #         if j not in self.model.clip_model.visual.transformer.resblocks[i].frozen_experts['all']:
        #             self.model.clip_model.visual.transformer.resblocks[i].frozen_experts['all'].extend(visual_indices_for_task)

        #     for k in indices_for_task:
        #         if k not in self.model.clip_model.transformer.resblocks[i].frozen_experts['all']:
        #             self.model.clip_model.transformer.resblocks[i].frozen_experts['all'].extend(indices_for_task)

        #     if i == 0:
        #         print("Reporting visual and text frozen expert indices for block 0: {} - {}".format(visual_indices_for_task, indices_for_task))





        self.result_dicts[self.task_id]["times"]["freezing"] += (time.time() - self.temp_time)
        self.temp_time = time.time()

        self.result_dicts[self.task_id]["metrics"]["choosemaps"] = choosemaps

        self.finetune_subnetwork()

        if len(self.tasks_by_subnet.keys()) == self.max_subnets:
            if self.experiment_type == "train":
                self.cluster_and_merge()
            elif self.experiment_type == "metric_calculation":
                self.calculate_metric_suite()

        ### Record frozen experts after potential merging operation
        frozen_experts = {"image": {}, "text": {}}
        for i in range(len(self.args.adapter_blocks_image)):
            if self.args.adapter_blocks_image[i] == True:
                frozen_experts["image"][i] = self.model.clip_model.visual.transformer.resblocks[i].frozen_experts
            if self.args.adapter_blocks_text == True:
                frozen_experts["text"][i] = self.model.clip_model.transformer.resblocks[i].frozen_experts

        self.result_dicts[self.task_id]["metrics"]["frozen experts"] = frozen_experts

        print("Ending online_after_task", flush=True)








    ##############################################################################################
    ###  Shift Analysis Operations
    ##############################################################################################


    def collect_task_acts(self):
        """
        Get activations for a single subnet and task
        """

        print("\n\n\n\nCollecting activations for all subnetworks on all known tasks", flush=True)
        self.model.clip_model.eval()


        # batches_to_record = self.act_batches_to_record
        max_act_samples = self.max_act_samples

        ### Prior to starting we reset all choosemaps so we can track expert use on memory samples
        #!# Note: We do this so that expert merging does not get biased by certain tasks having more or less overall samples
        # for i in range(len(self.model.clip_model.visual.transformer.resblocks)):
        #     self.model.clip_model.visual.transformer.resblocks[i].reset_choosemap()
        #     self.model.clip_model.transformer.resblocks[i].reset_choosemap()

        for name, module in self.model.clip_model.named_modules():
            if isinstance(module, ResidualAttentionBlock_MoA):
                module.reset_choosemap()

        acc_path = os.path.join(self.log_dir, f"mem_acc_stats.pt")
        logit_path = os.path.join(self.log_dir, f"logits.pt")
        label_path = os.path.join(self.log_dir, f"task_{self.task_id}_mem_label_dict.pt")

        try:
            stat_dict = torch.load(acc_path)
            total_correct = stat_dict["total_correct"]
            total_samples = stat_dict["total_samples"]
            ave_acc = stat_dict["ave_acc"]
        except:
            total_correct, total_samples, ave_acc = {}, {}, {}
 
        try:
            logits_dict = torch.load(logit_path)
        except:
            logits_dict = {}

        total_correct[self.task_id], total_samples[self.task_id], ave_acc[self.task_id] = {}, {}, {}
        logits_dict[self.task_id] = {}
        task_labels, task_labels_mapped, task_tokens, task_eot_indices = [], [], [], []

        subnet = self.subnet
        self.model.clip_model.set_task(self.task_id)
        self.model.clip_model.set_subnet(subnet)

        with torch.no_grad():
            for i, (x, y) in enumerate(self.test_dataloader):
                # if i == batches_to_record:
                if i*self.memory_batchsize > max_act_samples:
                    print("Breaking acts early", flush=True)
                    break

                self.add_new_class(y)
                if self.visible_classes == 'batch':
                    test_class_list = self.batch_exposed_classes
                    test_class_name_list = self.batch_exposed_classes_names
                    self.model.reset_class_names(test_class_name_list)
                else:
                    test_class_list = self.exposed_classes
                    test_class_name_list = self.exposed_classes_names
                    self.model.update_class_names(test_class_name_list)

                task_labels.append(y.clone())
                task_tokens.append(self.model.text_tokens)
                task_eot_indices.append(self.model.eot_indices)

                for j in range(len(y)):
                    label = y[j].item()
                    label_name = self.test_dataset.classes_names[label]
                    y[j] = test_class_name_list.index(label_name)

                task_labels_mapped.append(y.clone())


                x, y = x.to(self.device), y.to(self.device)




                for b in range(4):
                    if self.block_set == "first":
                        raise ValueError
                        #!# Except for correlation, the early blocks underperformed for clustering toy problems
                        # self.model.clip_model.visual.transformer.resblocks[b].set_activation_saving(True)
                        # self.model.clip_model.transformer.resblocks[b].set_activation_saving(True)
                    else:
                        ### Also collect from the last b blocks to compare how useful the activations are at the end of the network
                        self.model.clip_model.visual.transformer.resblocks[-(b+1)].set_activation_saving(True)
                        self.model.clip_model.transformer.resblocks[-(b+1)].set_activation_saving(True)


                logit, _, _ = self.model(x, self.model.text_tokens, is_train=False, val_subnet_id=subnet, apply_softmax=False)
                probs = logit.softmax(dim=-1)
                _, preds = logit.topk(self.topk, 1, True, True)


                # if i < batches_to_record:
                if subnet not in logits_dict[self.task_id].keys():
                    logits_dict[self.task_id][subnet] = [logit]
                else:
                    logits_dict[self.task_id][subnet].append(logit)

                if subnet not in ave_acc[self.task_id].keys():
                    total_correct[self.task_id][subnet]  = torch.sum(preds == y.unsqueeze(1)).item()
                    total_samples[self.task_id][subnet] = y.size(0)
                    print(f"Storing subnet {subnet} into acc dict for task {self.task_id}.")
                else:
                    total_correct[self.task_id][subnet]  += torch.sum(preds == y.unsqueeze(1)).item()
                    total_samples[self.task_id][subnet] += y.size(0)

                ave_acc[self.task_id][subnet] = total_correct[self.task_id][subnet] / total_samples[self.task_id][subnet]

        
        
        ### Store the labels, tokens of text label inputs, and the per-subnet accuracy stats
        task_label_dict = {"labels": task_labels, "labels_mapped": task_labels_mapped, "tokens": task_tokens, "eot_indices": task_eot_indices}
        torch.save(task_label_dict, label_path)

        stat_dict = {"total_correct": total_correct, "total_samples": total_samples, "ave_acc": ave_acc}
        torch.save(stat_dict, acc_path)
        torch.save(logits_dict, logit_path)




    def ood_task_analysis(self):
        """
        Gets metric values for single subnet on all tasks to measure impact of distribution shift and OoD datasets
        Run AFTER all tasks have been added to self.known_tasks from accuracy evaluation in _trainer.py
        """

        num_tasks = len(self.known_tasks)
        num_subnets = len(self.tasks_by_subnet.keys())
        num_blocks = self.num_blocks

        ### Getting subnet activations for given task
        self.known_tasks.sort()
        for test_task in self.known_tasks:
            self.task_id = test_task
            self.test_sampler.set_task(test_task)
            self.collect_task_acts()    

        ### Collect metric for stored activations
        metric_dict = self.get_metric_dict()

        for key in metric_dict.keys():
            print(metric_dict[key])




        #!# Potentially temporary solution, may be best to store each merge in a unique path instead
        acts_path = os.path.join(self.log_dir, "activations")
        shutil.rmtree(acts_path)

        os.remove(os.path.join(self.log_dir, "logits.pt"))
        for task in self.known_tasks:
            os.remove(os.path.join(self.log_dir, f"task_{task}_mem_label_dict.pt"))





    def calculate_metric_suite(self):

        num_tasks = len(self.known_tasks)
        num_subnets = len(self.tasks_by_subnet.keys())
        num_blocks = self.num_blocks



        ######################################
        ### Getting subnet activations
        ######################################
        self.temp_time = time.time()
        
        self.collect_mem_acts()    

        self.result_dicts[self.task_id]["times"]["collect acts"] += (time.time() - self.temp_time)
        self.temp_time = time.time()
        

        ######################################
        ### Getting metrics and clusters
        ######################################

        ### For each metric in the calculation suite, temporarily set the necessary arguments for calculation of that metric
        for m in range(len(self.metric_suite)):
            metric_info = self.metric_suite[m]

            self.metric = metric_info["metric"]
            self.metric_order = metric_info["metric_order"]
            self.metric_modality = metric_info["metric_modality"]
            self.first_layer = metric_info["first_layer"]
            self.second_layer = metric_info["second_layer"]


            ### Calculate and store the corresponding metric dict for the given metric in the suite
            metric_dict = self.get_metric_dict()
            for key in metric_dict.keys():
                print(f"metric dict for {self.metric}_{self.metric_order} key {key} is:\n{metric_dict[key]}")


        ### Once all metrics have been calculated, we remove any temporary files to free space
        self.result_dicts[self.task_id]["times"]["metric calculation"] += (time.time() - self.temp_time)
        self.temp_time = time.time()

        ### If we are just calculating metrics, then we retain them and delete the activations. 
        ### Otherwise we proceed to cluster and merge
        print("Removing activation files", flush=True)
        acts_path = os.path.join(self.log_dir, "activations")
        shutil.rmtree(acts_path)

        os.remove(os.path.join(self.log_dir, "logits.pt"))
        for task in self.known_tasks:
            os.remove(os.path.join(self.log_dir, f"task_{task}_mem_label_dict.pt"))










    ##############################################################################################
    ###  AutoEncoder-Only Training Operations
    ##############################################################################################




    #!# Currently set up as a placeholder that relies on task id instead of using the autoencoders for each task
    def get_ae_loss(self, task_id=None, split="train"):
        if len(self.autoencoders) == 0:
            return
        else:
            dataloader = self.train_dataloader if split == "train" else self.test_dataloader
            x,y,i = next(iter(dataloader))
            print("x type: ", x.type())
            x = x.to(self.device)
            x = x.to(torch.float32)
            # predict batch image domain:
            input_to_ae = self.feature_extractor(x)
            input_to_ae = F.sigmoid(input_to_ae.view(input_to_ae.size(0), -1))

            encoder_criterion = nn.MSELoss()
            
            ae_losses = []
            for t in range(self.n_tasks):
                model_autoencoder = self.autoencoders[t]
                outputs = model_autoencoder(input_to_ae)                
                ae_loss = encoder_criterion(outputs, input_to_ae)
                ae_losses.append(ae_loss.detach().cpu())
            print(f"Output shape {outputs.shape} and Best_Loss shape {ae_losses[0]} for Input shape {input_to_ae.shape}")

        print("ae_losses: ", ae_losses)
        self.result_dicts[task_id]["metrics"]["ae_losses"] = ae_losses
        return




    def online_step_ae_only(self, images, labels, idx):
        self.temp_time = time.time()

        self.add_new_class(labels)

        self.batch_counter += 1

        self.result_dicts[self.task_id]["times"]["label management"] += (time.time() - self.temp_time)
        self.temp_time = time.time()

        for _ in range(int(self.online_iter)):
            self.temp_time = time.time()
            self.online_train_ae_only([images, copy.deepcopy(labels)])

        self.temp_time = time.time()

        return 







    def online_train_ae_only(self, data):
        self.model.train()
        total_loss, total_correct, total_num_data = 0.0, 0.0, 0.0

        if self.visible_classes == 'batch':
            train_class_list = self.batch_exposed_classes
            train_class_name_list = self.batch_exposed_classes_names
            self.model.reset_class_names(train_class_name_list)
        else:
            train_class_list = self.exposed_classes
            train_class_name_list = self.exposed_classes_names
            self.model.update_class_names(train_class_name_list)

        x, y = data


        for j in range(len(y)):
            label = y[j].item()
            label_name = self.train_dataset.classes_names[label]
            y[j] = train_class_name_list.index(label_name)

        x,y = x.to(self.device), y.to(self.device)

        text_tokens = self.model.text_tokens
        self.result_dicts[self.task_id]["times"]["tokenization"] += (time.time() - self.temp_time)
        self.temp_time = time.time()

        print("Using autoencoder for task id: ", self.task_id)
        current_autoencoder = self.autoencoders[self.task_id]

        ### Train autoencoder only on the current task data for later task re-identification 
        input_to_ae = self.feature_extractor(x)
        input_to_ae = F.sigmoid(input_to_ae.view(input_to_ae.size(0), -1).to(self.device))

        self.optimizer_encoder = exp_lr_scheduler(self.optimizer_encoder, self.batch_counter, 0.01)
        self.optimizer_encoder.zero_grad()
        current_autoencoder.zero_grad()

        current_autoencoder.to(self.device)
        outputs = current_autoencoder(input_to_ae)

        # print("Input to ae shape: ", input_to_ae.shape)

        encoder_criterion = nn.MSELoss()
        loss_autoencoder = encoder_criterion(outputs, input_to_ae)
        loss_autoencoder.backward()
        self.optimizer_encoder.step()



        self.result_dicts[self.task_id]["times"]["training autoencoder"] += (time.time() - self.temp_time)
        self.temp_time = time.time()


        self.update_schedule()

        return













    ##############################################################################################
    ###  Class Utility Functions
    ##############################################################################################





    ### Updates memory, just tracking indices and labels
    def update_memory(self, sample, labels, label_names):
        """
        Notes: Should behave differently for caption vs label datasets
            Label datasets: Limit memory capacity per class to protect rarer classes
            Caption datasets: Instead of memory per class it is memory per task

        Otherwise the behavior is largely the same, storing or gradually replacing encountered samples.
        For caption datasets pseudolabels are used with labels being the task_id, so this function acts 
        identically either way.

        In cases where distribution shift occurs for a class, there are two options:
            1. "Shared Label Space": Replace the unshifted samples of a given class with recent shifted samples.
            2. "Distinct Label Spaces": Shifted samples from new task are stored as a separate class. 

        This behavior would entirely be determined by how the labeling is handled by the dataset. For instance,
        if label 3 is "car" in task 0, and later "car" appears in a rotated task, case 1 occurs if it is again 
        labeled as class 3, but if it is given a new label then case 2 will occur. 
        """

        ### Determine what memory indices to place samples in based on capacity constraints
        idx = []
        num_replaced = 0
        num_over_count = 0
        if self.args.use_memory_class_names:
            for name in label_names:
                self.memory.seen[name] += 1

                if self.memory.cls_count[name] < self.memory.memory_per_class:
                    idx.append(-1)
                else:
                    # num_over_count += 1
                    j = torch.randint(0, self.memory.seen[name], (1, )).item()
                    ### Get the jth index used by that class to maintain class balance in the buffer
                    if j < self.memory.memory_per_class:
                        valid_cls_idxs = (self.memory.label_names == name).nonzero()[0]
                        idx.append(valid_cls_idxs[j])
                    else:
                        # idx.append(self.memory.memory_per_class)
                        idx.append(-1)

            ### Place the corresponding samples into memory using the stored indices
            for i, index in enumerate(idx):
                #!# Note: This method skips storage of samples indexed as "-1" once the class capacity is reached
                ###   but this should be fine since we expect all such samples to be equally representative of the 
                ###   distribution of data at the time of collection (i.e. all part of same task)
                if self.memory.cls_count[label_names[i]] >= self.memory.memory_per_class:
                    # if index < self.memory.memory_per_class and index != -1:
                    if index != -1:
                        self.memory.replace_data([sample[i], labels[i].item(), label_names[i]], index)
                        num_replaced += 1
                ### Until buffer is full, no index is passed in so that samples are simply appended to the buffer
                else:
                    self.memory.replace_data([sample[i], labels[i].item(), label_names[i]])



        else:
            for lbl in labels:
                self.memory.seen[lbl.item()] += 1

                if self.memory.cls_count[lbl.item()] < self.memory.memory_per_class:
                    idx.append(-1)
                else:
                    num_over_count += 1
                    j = torch.randint(0, self.memory.seen[lbl.item()], (1, )).item()
                    # if self.task_id == 1:
                    #     print("J is: ", j, " for range 0-", self.memory.seen[lbl.item()], " and limit ", self.memory.memory_per_class)
                    ### Get the jth index used by that class to maintain class balance in the buffer
                    # if j < self.memory.memory_per_class:
                    if j < self.memory.cls_count[lbl.item()]:
                        valid_cls_idxs = (self.memory.labels == lbl.item()).nonzero(as_tuple=True)[0]
                        # if self.task_id == 1:
                        #     print("J is ", j, " Valid idxs: ", valid_cls_idxs)
                        idx.append(valid_cls_idxs[j])
                    else:
                        idx.append(-1) # Appends -1, but since count is >= memory per class it won't get added
        

            ### Place the corresponding samples into memory using the stored indices
            for i, index in enumerate(idx):
                #!# Note: This method skips storage of samples indexed as "-1" once the class capacity is reached
                ###   but this should be fine since we expect all such samples to be equally representative of the 
                ###   distribution of data at the time of collection (i.e. all part of same task)
                if self.memory.cls_count[labels[i].item()] >= self.memory.memory_per_class:
                    # if index < self.memory.memory_per_class and index != -1:
                    if index != -1:
                        self.memory.replace_data([sample[i], labels[i].item(), label_names[i]], index)
                        num_replaced += 1
                ### Until buffer is full, no index is passed in so that samples are simply appended to the buffer
                else:
                    self.memory.replace_data([sample[i], labels[i].item(), label_names[i]])

        # if num_replaced > 0:
        #     print(f"Replaced {num_replaced} memory samples")
        # if num_over_count > 0:
        #     print(f"Checked {num_over_count} extra memory samples")
        # if self.task_id > 0 and num_replaced == 0 and num_over_count == 32:
        #     print("Memory class seen count: ", self.memory.seen)



    def update_schedule(self, reset=False):
        if reset:
            self.scheduler = select_scheduler(self.sched_name, self.optimizer, None)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lr
        else:
            self.scheduler.step()




    def reset_opt(self):
        self.optimizer = select_optimizer(self.opt_name, self.lr, self.model)
        self.scheduler = select_scheduler(self.sched_name, self.optimizer, None)

    def add_new_batch_class(self, labels, mode="train"):
        batch_exposed_classes = []

        for label in labels:
            if label.item() not in self.batch_exposed_classes:
                self.batch_exposed_classes.append(label.item())

        ### Changed to allow multiple labels to map to the same class name for datasets with reoccuring classes under distribution shift
        self.batch_exposed_classes.sort()
        self.batch_exposed_classes_names = []

        dataset = self.train_dataset if mode == "train" else self.test_dataset

        for i in self.batch_exposed_classes:
            if dataset.classes_names[i] not in self.batch_exposed_classes_names:
                self.batch_exposed_classes_names.append(dataset.classes_names[i])
        
        # if mode == "train":
        #     for i in self.batch_exposed_classes:
        #         if self.train_dataset.classes_names[i] not in self.batch_exposed_classes_names:
        #             self.batch_exposed_classes_names.append(self.train_dataset.classes_names[i])
        # else:
        #     for i in self.batch_exposed_classes:
        #         if self.train_dataset.classes_names[i] not in self.batch_exposed_classes_names:
        #             self.batch_exposed_classes_names.append(self.train_dataset.classes_names[i])

        # #!# Needs to check if duplicate labels map to the same class
        # self.batch_exposed_classes_names = [self.train_dataset.classes_names[i]
        #                                         for i in self.batch_exposed_classes]



    def add_new_class(self, labels, mode="train"):
        _old_num = len(self.exposed_classes)
        super().add_new_class(labels)

        self.batch_exposed_classes = []
        self.batch_exposed_classes_names = []
        self.add_new_batch_class(labels, mode)

    def report_training(self, sample_num, train_loss, train_acc, step=""):
        print(
            f"Train | Sample # {sample_num} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
            f"lr {self.optimizer.param_groups[0]['lr']:.6f} | "
            f"Classes {len(self.exposed_classes)} | "
            f"Names {len(self.exposed_classes_names)} | "
            f"Num_Batch_Classes {len(self.batch_exposed_classes)} | "
            f"running_time {datetime.timedelta(seconds=int(time.time() - self.start_time))} | "
            # f"ETA {datetime.timedelta(seconds=int((time.time() - self.start_time) * (self.total_samples-sample_num) / sample_num))}"
        )

        if step in self.result_dicts[self.task_id]["accuracy"].keys():
            self.result_dicts[self.task_id]["accuracy"][step] = train_acc



