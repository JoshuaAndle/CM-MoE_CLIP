"""
Utility functions for calculating first and second order metrics of transformer activations
"""

import os
import argparse

import numpy as np
import copy
import torch
import torch.nn as nn


###################################################################################################################################################
###
###     Act Dict Preprocessing Functions
###
###################################################################################################################################################

### Concatenate across list of batches for each value in the dict
def cat_dict_acts(batch_list):
    return torch.cat(batch_list, dim=0)

def unflatten_expert_inputs(expert_inputs_raw, num_tokens):
    unflattened_experts = []
    for exp in range(len(expert_inputs_raw)):
        expert_temp = expert_inputs_raw[exp]
        unflattened_experts.append(expert_temp.reshape(expert_temp.shape[0], num_tokens, -1))
    return unflattened_experts

### Loop over all experts and concatenate the activations from their batches
def cat_experts(expert_list):
    nonzero_experts = []
    for exp in range(len(expert_list[0])):
        batched_output = []
        for batch in range(len(expert_list)):
            batched_output.append(expert_list[batch][exp])
        ### Append the resulting concatenated acts, which is empty for unused experts
        nonzero_experts.append(torch.cat(batched_output, dim=0))
    return nonzero_experts

### Remove any unused experts after concatenating batches
def remove_empty_experts(expert_list):
    final_expert_list = []
    for exp in range(len(expert_list)):
        if expert_list[exp].shape[0] > 0:
            final_expert_list.append(expert_list[exp])
    return final_expert_list



###################################################################################################################################################
###
###     Metric Functions
###
###################################################################################################################################################






def class_condition(acts, labels, modality):
    ### Given a tensor of shape [batch, token, embed] and corresponding labels tensor, normalize samples by class

    ### CLIP text input is 1 sample per known class label, so conditioning can't use the batch input labels used for images. Instead we normalize within each given class' tokens
    if modality == "text":
        means = acts.mean(dim=(1,2))
        stds = acts.std(dim=(1,2))
        acts -= means.view(-1,1,1).expand(acts.shape)
        acts /= stds.view(-1,1,1).expand(acts.shape)

    else:
        classes = torch.unique(labels)
        for c in classes:
            cls_indices = torch.where(labels==c, 1, 0).eq(1)

            acts[cls_indices] -= acts[cls_indices].mean()
            acts[cls_indices] /= acts[cls_indices].std()

    return acts




def fold_samples(a,labels=None):
    ### Folds embed vectors along sample
    ### Takes inputs as tensor of shape [batch, tokens, embed]

    ### In the absence of labels do simple averaging
    if labels == None:
        return torch.mean(a, dim=0, keepdim=True)


    a_folded = []
    classes = torch.unique(labels)
    for c in classes:
        cls_indices = torch.where(labels==c, 1, 0).eq(1)

        # a_class = a[cls_indices]

        a_folded.append(torch.mean(a[cls_indices], dim=0))


    a_folded = torch.cat(a_folded,dim=0)

    #!# I dont think this was done as intended, it should average regardless and unsqueeze the 0th dimension if needed
    a_folded = a_folded.mean(dim=0, keepdim=True)
            

    return a_folded



def fold_tokens(unfolded_tokens, numpy=False):
    ### Takes inputs as tensor of shape [batch, tokens, embed]
    # if numpy == True:
    #     return np.mean(unfolded_tokens,axis=1)
    # else:
    return torch.mean(unfolded_tokens, dim=1, keepdim=True)


def fold_embed(unfolded_acts):
    ### Takes inputs as tensor of shape [batch, tokens, embed]
    return torch.mean(unfolded_acts, dim=-1, keepdim=True)



def subsample_cls_token(acts):
    ### Assuming both acts has had its batch dimensions concatenated
    cls_acts_subsample = acts[:,0,:].unsqueeze(1)

    ### Returns shape (# images, cls_token, embed)
    return cls_acts_subsample

def subsample_eot_token(acts, eot_indices):
    ### Acts need to be in (# classes, tokens, embed)
    ### Assuming both acts and eot_indices have had their batch dimensions concatenated
    eot_acts_subsample = [acts[i,idx,:] for i,idx in enumerate(eot_indices)]
    eot_acts_subsample = torch.stack(eot_acts_subsample, dim=0)

    ### Returns shape (# classes, eot_token, embed)
    return eot_acts_subsample.unsqueeze(1)


def remove_padding(acts_l, acts_k, eot_indices):
    ### Fully removing padding would result in non-uniform shapes across text input samples, so instead it would need to be done at the level of each individual metric function
    ###    which would mean needing to add it as an argument and include code for handling this use case. Best to just see if other options work for clustering first.
    #!# For now a partial implementation is done where we settle for stripping according to the longest non-padded token sequence to maintain uniform lengths
    # pass
    max_token_idx = torch.max(torch.tensor(eot_indices))

    ### Remove any padding between the max token index and the ends of the token sequences
    if max_token_idx < (acts_l.shape[1]-1):
        acts_l_stripped, acts_k_stripped = acts_l[:, 0:(max_token_idx+1), :], acts_k[:, 0:(max_token_idx+1), :]
        print(f"For max token index {max_token_idx} the new shapes of acts_l and acts_k are {acts_l_stripped.shape} and {acts_k_stripped.shape}")

    return acts_l_stripped, acts_k_stripped






### Given the labels and gate routing matrices for those samples, constructs a list of which labels were routed to each expert
def get_expert_labels(gates,labels):
    ### Output is a list of lists with non-uniform shape [# experts routed to, # samples routed to the given expert]

    ### Transpose gates to get [experts, samples] so we can map which samples are used in each expert
    # [82,28] --> [28,82]
    gates_T = gates.T
    # print(gates[:3])

    # gates_map = gates_T > 0.000000001
    gates_map = gates_T.ne(0.0)
    # print(gates_map[:,:2])

    ### Make a mask of which gates had any samples routed to them for filtering purposes
    mask = gates_map.any(dim=1)
    filtered = gates_map[mask]

    ### Since sample counts are non-uniform, we make a list of lists for each expert's routed sample labels
    exp_labels = []
    for i in range(filtered.shape[0]):
        assert filtered[i].any() == True, f"Expert with no samples was not properly filtered in get_expert_labels. Filtered is {filtered}"
        exp_temp = labels[filtered[i]]
        exp_labels.append(exp_temp)
        
    ### Returns a list of lists containing the integer labels of each routed sample per expert
    return exp_labels    



#!# Need to try this with 3 dimensional inputs and dim=2, but for now flattening should have the same results
def torch_cos_sim(a,b,dim=1):
    ### Takes two activation tensors of shape [tokens, embed] or [samples, embed] depending on preprocessing

    torch_cossim = nn.functional.cosine_similarity(a,b,dim=dim, eps=1e-6)
    return torch_cossim


def torch_euc_dist(a,b):
    ### Takes two tensors a and b of shape [token, embed] or [samples, embed]
    dist = a - b
    ### Take L2 norm of resulting distance vectors
    magnitude = torch.linalg.norm(dist, dim=-1)

    ### Note: I am intending this to give per-sample or per-token magnitudes, which can either be averaged or used for correlation calculation
    return magnitude


# ### I realize this is the same as the euclidean distance function, but I am keeping it as the intention is to take it between input/output
# ###    rather than between two subnetworks as is the case for euc_dist
# def gpt_energy(h_in,h_out):
#     delta = h_out - h_in
#     energy = torch.norm(delta, dim=-1)      # per token energy
#     energy = energy.mean()                  # per layer / batch mean
#     return energy




def calculate_energy(logits):
    ### Gets the log-sum-exp energy averaged over all samples
    ### Logits are shape (# images, # classes)
    ### Assume batches have been concatenated
    total_energy, num_samples = 0, 0

    for batch in range(len(logits)):
        num_samples += logits[batch].shape[0]
        ### Get energy for each sample
        for idx in range(logits[batch].shape[0]):
            energy = 0
            ### Sum exponents of logits for given sample
            for logit in range(logits[batch].shape[1]):
                energy += torch.exp(logits[batch][idx,logit])
            energy = -torch.log(energy)            
        total_energy += energy


    return (total_energy / num_samples)





#!# Numpy implementation is in backup notebook
### Calculate the connectivity between a given pair of layers
def torch_calc_conn(parent_acts, child_acts, labels, unique_samples=False):
    p_op = copy.deepcopy(parent_acts)
    c_op = copy.deepcopy(child_acts)


    if unique_samples == False:

        # print("Unique samples false")
        classes = torch.unique(labels)
        corr_by_class = []
        for c in classes:
            ### If data was class conditioned, we just access the appropriate class entry, otherwise we need the marginal sample subset by label
            p_marg = p_op[labels==c]
            c_marg = c_op[labels==c]
        
            # p_marg = torch.reshape(p_marg, (-1, p_marg.shape[-1]))
            # c_marg = torch.reshape(c_marg, (-1, c_marg.shape[-1]))

            cat_features = torch.cat([p_marg, c_marg], dim=1).permute(1,0)

            ### If statement needed to filter out cases where classes may have insufficient sample counts
            if p_marg.shape[0] > 1 and c_marg.shape[0] > 1:
                ### Parents is a 2D list of all of the connectivities of parents and children embed features for a single class (with token dim either flattened or folded)
                coefs = torch.corrcoef(cat_features)

                num_parents = p_marg.shape[-1]
                corrs_by_parent = coefs[:num_parents, num_parents:]
                # print("Coefs shape ", coefs.shape, ", num_parents shape: ", num_parents, " corrs_by_parent shape: ", corrs_by_parent.shape)

                ### We take the absolute value because we only care about the STRENGTH of the correlation, not the sign
                corrs_by_parent = torch.abs(corrs_by_parent)
                corr_by_class.append(corrs_by_parent.mean())
            else:
                print("Insufficient samples for class ", c)
        return torch.tensor(corr_by_class)
    
    ### If the samples are all unique labels (for example if classes were averaged already) then we forego class conditioning
    else:

        cat_features = torch.cat([p_op, c_op], dim=1).permute(1,0)

        ### Parents is a 2D list of all of the connectivities of parents and children embed features for a single class (with token dim either flattened or folded)
        coefs = torch.corrcoef(cat_features)

        num_parents = p_op.shape[-1]
        corrs_by_parent = coefs[:num_parents, num_parents:]
        # print("Coefs shape ", coefs.shape, ", num_parents shape: ", num_parents, " corrs_by_parent shape: ", corrs_by_parent.shape)

        ### We take the absolute value because we only care about the STRENGTH of the correlation, not the sign
        corrs_by_parent = torch.abs(corrs_by_parent)
    
        return corrs_by_parent.unsqueeze(0)






### Run the appropriate processing and metric functions and return the resulting scalar value
def get_metric(acts_l, acts_k, branch, labels, args, eot_indices=None):
    ### Activation tensors should always have shape [# samples, # tokens, embed_dim]

    if args.condition_by_class == True:
        acts_l = class_condition(acts_l, labels, branch)
        acts_k = class_condition(acts_k, labels, branch)

    if args.remove_padding == True:
        if branch == "text":
            acts_l, acts_k = remove_padding(acts_l, acts_k, eot_indices)

    if args.subsample_tokens == True:
        if branch == "text":
            acts_l = subsample_eot_token(acts_l, eot_indices)
            acts_k = subsample_eot_token(acts_k, eot_indices)
        else:
            acts_l = subsample_cls_token(acts_l)
            acts_k = subsample_cls_token(acts_k)

    final_metric = None
    if args.metric == "corr":
        assert len(acts_l.shape) == 3 and len(acts_k.shape) == 3, "unexpected activation tensor shapes in preprocessing for correlation"

        ### Expand and flatten the labels to reflect the flattening of samples x tokens in acts_l and acts_k
        labels = labels.view(-1,1).expand(acts_l.shape[0], acts_l.shape[1]).reshape(-1)

        ### Treats the samples x tokens as "observations"
        acts_l = acts_l.reshape(-1, acts_l.shape[-1])
        acts_k = acts_k.reshape(-1, acts_k.shape[-1])

        unique_samples = False

        ### Returns the correlation by class as a tensor
        temp_metric = torch_calc_conn(acts_l, acts_k, labels, unique_samples)
        
    #!# Currently not implemented as being averaged by class, but classes should be roughly balanced for the time being
    elif args.metric == "cos":
        assert len(acts_l.shape) == 3 and len(acts_k.shape) == 3, "unexpected activation tensor shapes in preprocessing for cosine similarity"

        ### Flatten the samples x tokens
        acts_l = acts_l.reshape(-1, acts_l.shape[-1])
        acts_k = acts_k.reshape(-1, acts_k.shape[-1])

        temp_metric = torch_cos_sim(acts_l, acts_k, dim=1)

    elif args.metric == "dist":
        temp_metric = torch_euc_dist(acts_l, acts_k)

    ### Gets the average value over all samples and tokens
    final_metric = torch.mean(temp_metric)

    return final_metric







# ### Run the appropriate processing and metric functions and return the resulting scalar value
# def get_metric(acts_l, acts_k, branch, labels, args_dict, eot_indices=None, logits=None):
#     ### Activation tensors should always have shape [# samples, # tokens, embed_dim]

#     if args_dict.get("condition_by_class") == True:
#         acts_l = class_condition(acts_l, labels, branch)
#         acts_k = class_condition(acts_k, labels, branch)

#     if args_dict.get("remove_padding") == True:
#         if branch == "text":
#             acts_l, acts_k = remove_padding(acts_l, acts_k, eot_indices)

#     if args_dict.get("subsample_tokens") == True:
#         if branch == "text":
#             acts_l = subsample_eot_token(acts_l, eot_indices)
#             acts_k = subsample_eot_token(acts_k, eot_indices)
#         else:
#             acts_l = subsample_cls_token(acts_l)
#             acts_k = subsample_cls_token(acts_k)

#     metric = None
#     if args_dict.get("metric") == "energy":
#         metric = calculate_energy(logits)

#     else:
#         if args_dict.get("metric") == "corr":
#             assert len(acts_l.shape) == 3 and len(acts_k.shape) == 3, "unexpected activation tensor shapes in preprocessing for correlation"

#             ### Expand and flatten the labels to reflect the flattening of samples x tokens in acts_l and acts_k
#             labels = labels.view(-1,1).expand(acts_l.shape[0], acts_l.shape[1]).reshape(-1)
    
#             ### Treats the samples x tokens as "observations"
#             acts_l = acts_l.reshape(-1, acts_l.shape[-1])
#             acts_k = acts_k.reshape(-1, acts_k.shape[-1])

#             unique_samples = False

#             ### Returns the correlation by class as a tensor
#             temp_metric = torch_calc_conn(acts_l, acts_k, labels, unique_samples)
            
#         #!# Currently not implemented as being averaged by class, but classes should be roughly balanced for the time being
#         elif args_dict.get("metric") == "cos":
#             assert len(acts_l.shape) == 3 and len(acts_k.shape) == 3, "unexpected activation tensor shapes in preprocessing for cosine similarity"

#             ### Flatten the samples x tokens
#             acts_l = acts_l.reshape(-1, acts_l.shape[-1])
#             acts_k = acts_k.reshape(-1, acts_k.shape[-1])

#             temp_metric = torch_cos_sim(acts_l, acts_k, dim=1)

#         elif args_dict.get("metric") == "dist":
#             temp_metric = torch_euc_dist(acts_l, acts_k)

#         ### Gets the average value over all samples and tokens
#         metric = torch.mean(temp_metric)

#     return metric


















# ### Run the appropriate processing and metric functions and return the resulting scalar value
# #!# Original implementation where the parsed args are directly passed in, for compatability with metric_calculations.py
# def get_metric_argdict(acts_l, acts_k, labels, args_dict, eot_indices=None, logits=None):
#     ### Activation tensors should always have shape [# samples, # tokens, embed_dim]
#     if args_dict.condition_by_class == True:
#         acts_l = class_condition(acts_l, labels, args_dict.branch)
#         acts_k = class_condition(acts_k, labels, args_dict.branch)

#     if args_dict.remove_padding == True:
#         if args_dict.branch == "text":
#             acts_l, acts_k = remove_padding(acts_l, acts_k, eot_indices)



#     if args_dict.fold_samples == True:
#         acts_l = fold_samples(acts_l)
#         acts_k = fold_samples(acts_k)
#     elif args_dict.fold_tokens == True:
#         acts_l = fold_tokens(acts_l)
#         acts_k = fold_tokens(acts_k)
#     elif args_dict.fold_embed == True:
#         acts_l = fold_embed(acts_l)
#         acts_k = fold_embed(acts_k)
#     elif args_dict.subsample_tokens == True:
#         if args_dict.branch == "text":
#             acts_l = subsample_eot_token(acts_l, eot_indices)
#             acts_k = subsample_eot_token(acts_k, eot_indices)
#         else:
#             acts_l = subsample_cls_token(acts_l)
#             acts_k = subsample_cls_token(acts_k)
#     # elif args_dict.metric == "corr"

#     metric = None


#     if args_dict.metric == "energy":
#         metric = calculate_energy(logits)

#     else:
#         if args_dict.metric == "corr":
#             assert len(acts_l.shape) == 3 and len(acts_k.shape) == 3, "unexpected activation tensor shapes in preprocessing for correlation"
#             assert args_dict.fold_embed == False, "invalid folding option for correlation: fold_embed"
#             # assert (args_dict.branch == "text" and args_dict.fold_tokens == True) is False, "Can\'t fold tokens for text correlation due to individual samples"

#             #!# Note: Removed this since we currently have no situations where we would group dimensions differently. Left it in as a reminder in case that changes
#             # # ### Prepare the 3d tensor to be the appropritate 2d format of [observations, values] (typically corresponding to [samples, neuron_activations])
#             # # if args_dict.fold_samples or args_dict.fold_tokens or args_dict.subsample_tokens:
#             # #     ### Treats the samples x tokens as "observations"
#             #     acts_l = acts_l.reshape(acts_l.shape[0], -1)

#             ### Expand and flatten the labels to reflect the flattening of samples x tokens in acts_l and acts_k
#             labels = labels.view(-1,1).expand(acts_l.shape[0], acts_l.shape[1]).reshape(-1)
    
#             ### Treats the samples x tokens as "observations"
#             acts_l = acts_l.reshape(-1, acts_l.shape[-1])
#             acts_k = acts_k.reshape(-1, acts_k.shape[-1])
#             # print("Resulting acts_l shape: ", acts_l.shape, " and labels shape: ", labels.shape)

#             unique_samples = False
#             # if (args_dict.branch == "text") and (args_dict.fold_tokens == True or args_dict.subsample_tokens == True):
#             #     unique_samples = True

#             ### Returns the correlation by class as a tensor
#             temp_metric = torch_calc_conn(acts_l, acts_k, labels, unique_samples)
            
#         #!# Currently not implemented as being averaged by class, but classes should be roughly balanced for the time being
#         elif args_dict.metric == "cos":
#             assert len(acts_l.shape) == 3 and len(acts_k.shape) == 3, "unexpected activation tensor shapes in preprocessing for cosine similarity"

#             ### Flatten the samples x tokens
#             acts_l = acts_l.reshape(-1, acts_l.shape[-1])
#             acts_k = acts_k.reshape(-1, acts_k.shape[-1])

#             temp_metric = torch_cos_sim(acts_l, acts_k, dim=1)

#         elif args_dict.metric == "dist":
#             temp_metric = torch_euc_dist(acts_l, acts_k)

#         ### Gets the average value over all samples and tokens
#         metric = torch.mean(temp_metric)

#     return metric








# #!# Numpy implementation is in backup notebook
# ### Calculate the connectivity between a given pair of layers
# def torch_calc_conn(parent_acts, child_acts, labels, class_conditioned=False, fold_token_dim=False):
#     p_op = copy.deepcopy(parent_acts)
#     c_op = copy.deepcopy(child_acts)

#     #!# Two options, either dim 0 is class-averaged single elements per class, or all class-conditioned samples.
#     #!#    In the first case we can loop over the entries, but for the second case we need to get the marginal sets of samples based on the labels

#     if class_conditioned == True:
#         classes = torch.arange(p_op.shape[0])
#     else:
#         classes = torch.unique(labels)

#     corr_by_class = []
#     for c in classes:
#         ### If data was class conditioned, we just access the appropriate class entry, otherwise we need the marginal sample subset by label
#         if class_conditioned == True:
#             p_marg = torch.unsqueeze(p_op[c], dim=0)
#             c_marg = torch.unsqueeze(c_op[c], dim=0)

#         else:
#             p_marg = p_op[labels==c]
#             c_marg = c_op[labels==c]

#         ### Either fold the token dimension by averaging all tokens, or flatten it with the class/sample dimension
#         if fold_token_dim == True:
#             p_marg = fold_tokens(p_marg)
#             c_marg = fold_tokens(c_marg)
#         else:
#             p_marg = torch.reshape(p_marg, (-1, p_marg.shape[-1]))
#             c_marg = torch.reshape(c_marg, (-1, c_marg.shape[-1]))
#         # print("Parent shape: ", p_marg.shape, " Children shape: ", c_marg.shape)

#         cat_features = torch.cat([p_marg, c_marg], dim=1).permute(1,0)

#         ### If statement needed to filter out cases where classes may have insufficient sample counts
#         if p_marg.shape[0] > 1 and c_marg.shape[0] > 1:
#             ### Parents is a 2D list of all of the connectivities of parents and children embed features for a single class (with token dim either flattened or folded)
#             coefs = torch.corrcoef(cat_features)

#             num_parents = p_marg.shape[-1]
#             corrs_by_parent = coefs[:num_parents, num_parents:]
#             # print("Coefs shape ", coefs.shape, ", num_parents shape: ", num_parents, " corrs_by_parent shape: ", corrs_by_parent.shape)

#             ### We take the absolute value because we only care about the STRENGTH of the correlation, not the sign
#             corrs_by_parent = torch.abs(corrs_by_parent)
#             corr_by_class.append(corrs_by_parent.mean())
#         else:
#             print("Insufficient samples for class ", c)



#     return torch.tensor(corr_by_class)



