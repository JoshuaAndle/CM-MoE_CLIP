import numpy as np
import torch
import torch_optimizer
from torch.nn import Module
from torch import optim
from torch.optim import lr_scheduler
import copy


acts = {}


def set_hook(modality, block, adapter, name):
    def hook_fn(module, input, output):
        acts[modality][block][adapter][name] = output.detach().cpu()
    return hook_fn


def set_all_layers(clip_model, hook_handles, subnet, only_subnet_adapters):    

    acts['visual'] = {}
    for block_idx, block in enumerate(clip_model.visual.transformer.resblocks):
        acts['visual'][block_idx] = {}
        for adapter_idx, adapter in enumerate(block.adaptmlp_list):
            if only_subnet_adapters and adapter_idx not in block.frozen_experts[subnet]:
                continue
            else:
                acts['visual'][block_idx][adapter_idx] = {}
                hook_handles.append(adapter.down_proj.register_forward_hook(set_hook('visual', block_idx, adapter_idx, "down_proj")))
                hook_handles.append(adapter.up_proj.register_forward_hook(set_hook('visual', block_idx, adapter_idx, "up_proj")))



def store_acts(clip_model, all_acts, all_labels, y_label, step):
    # all_labels = torch.cat((all_labels, y_label.detach().cpu()), dim=0)
    if step == 0:
        # for modal in ['text', 'visual']:
        for modal in ['visual']:
            for block_idx, block in enumerate(clip_model.visual.transformer.resblocks):
                ### Access the SparseDispatcher of the resblock to check how samples were routed
                dispatcher = block.dispatcher

                ### Determine which batch samples were routed to each expert in the block
                #!# Note: This assumes that frozen_experts and the stored adapter_idx keys have the same order, which is true as written
                split_idxs = torch.split(dispatcher._batch_index, dispatcher._part_sizes, dim=0)
                split_lens = []
                for sample_tensor in split_idxs:
                    split_lens.append(sample_tensor.shape[0])
                print("Split idx lengths: ", split_lens)

                for adapter_idx in acts[modal][block_idx].keys():
                    for layer in ['down_proj', 'up_proj']:
                        ### Note: this assumes linear layer outputs, as it is used with the adapter layers
                        all_acts[modal][block_idx][adapter_idx][layer] = acts[modal][block_idx][adapter_idx][layer]

                    all_labels[modal][block_idx][adapter_idx] = y_label[split_idxs[adapter_idx].cpu()]
                    # print("Adapter {}: {}".format(adapter_idx, split_idxs[adapter_idx]))






    else: 
        # for modal in ['text', 'visual']:
        for modal in ['visual']:
            for block_idx, block in enumerate(clip_model.visual.transformer.resblocks):
                ### Access the SparseDispatcher of the resblock to check how samples were routed
                dispatcher = block.dispatcher

                ### Determine which batch samples were routed to each expert in the block
                #!# Note: This assumes that frozen_experts and the stored adapter_idx keys have the same order, which is true as written
                split_idxs = torch.split(dispatcher._batch_index, dispatcher._part_sizes, dim=0)

                for adapter_idx in acts[modal][block_idx].keys():
                    for layer in ['down_proj', 'up_proj']:
                        all_acts[modal][block_idx][adapter_idx][layer] = torch.cat((all_acts[modal][block_idx][adapter_idx][layer],
                                                                            acts[modal][block_idx][adapter_idx][layer]), dim=0)

                    all_labels[modal][block_idx][adapter_idx] = torch.cat((all_labels[modal][block_idx][adapter_idx],
                                                                        y_label[split_idxs[adapter_idx].cpu()]), dim=0)
    return all_acts, all_labels



### Takes a MoE clip model and returns all activations for a set of data
def activations(clip_model, text_tokens, data_loader, subnet=None, only_subnet_adapters=True):
    print("\n\n")
    handles     = []

    set_all_layers(clip_model, handles, subnet, only_subnet_adapters)

    ### Make a copy of the empty acts dict to store values for all batches
    all_acts = copy.deepcopy(acts)
    all_labels = copy.deepcopy(acts)




    ### Note: The stacking will cause an error if the dataloader only has one batch, seemingly. Not currently an issue, but leaving this just in case
    with torch.no_grad():
        for step, data in enumerate(data_loader):
            x_input, y_label = data
            clip_model(x_input.cuda(), text_tokens)

            all_acts, all_labels = store_acts(clip_model, all_acts, all_labels, y_label, step)


    # first_expert = list(all_acts['visual'][0].keys())[0]
    # print("Resulting shape of one key: ", all_acts['visual'][0][first_expert]['down_proj'].shape)


    # for block in all_acts['visual'].keys():
    #     print("Adapter keys for visual block {}: {}".format(block, all_acts['visual'][block].keys()))


    # for adapter in all_labels['visual'][0].keys():
    #     print("Labels for adapter {} in block 0: {}".format(adapter, all_labels['visual'][0][adapter]))


    for handle in handles:
        handle.remove()    

    return all_acts, all_labels








### Get the average connectivity for all tokens, averaged across classes and neurons
def calc_conn_by_token(parent_acts, child_acts, adapter_labels, fold_tokens=False):
    #!# Note: Can't use fold tokens if we want the individual token connectivities
    raw_conns = calc_conn(parent_acts, child_acts, adapter_labels, fold_tokens=False)
    raw_conns = np.array(raw_conns).mean(axis=(0,2))

    return raw_conns


### Get the average connectivity for all neurons, averaged across classes and neurons
def calc_conn_by_neuron(parent_acts, child_acts, adapter_labels, fold_tokens=False):
    raw_conns = calc_conn(parent_acts, child_acts, adapter_labels, fold_tokens)

    if fold_tokens:
        raw_conns = np.array(raw_conns).mean(axis=(0))
    else:
        raw_conns = np.array(raw_conns).mean(axis=(0,1))


    return raw_conns


#!# Note: This may not be a good metric for clustering in MoE as there is no guarantee that all labels are present in each adapter
### Get the average connectivity for all classes, averaged across tokens and neurons
def calc_conn_by_class(parent_acts, child_acts, adapter_labels, fold_tokens=False):
    raw_conns = calc_conn(parent_acts, child_acts, adapter_labels, fold_tokens)
    if fold_tokens:
        raw_conns = np.array(raw_conns).mean(axis=(1))
    else:
        raw_conns = np.array(raw_conns).mean(axis=(1,2))

    return raw_conns


#*# for now removing the class-conditioning of correlation since the sparse dispatcher makes matching labels to activations much less straightforward
### Will look into re-adding, but that requires tracking the gates over all batches and transformer blocks during forward passes

### Calculate the connectivity between a given pair of layers
def calc_conn(parent_acts, child_acts, adapter_labels, fold_tokens=False):

    p_op = copy.deepcopy(parent_acts) 
    c_op = copy.deepcopy(child_acts)

    parent_aves = []
    p_op = p_op.numpy()
    c_op = c_op.numpy()
    
    ### Normalize activations for each class prior to getting covariance
    for label in list(np.unique(adapter_labels.numpy())):
    
        parent_mask = np.ones(p_op.shape,dtype=bool)
        child_mask = np.ones(c_op.shape,dtype=bool)

        parent_mask[adapter_labels != label] = False
        child_mask[adapter_labels != label] = False
        
        p_op[parent_mask] -= np.mean(p_op[parent_mask])
        p_op[parent_mask] /= np.std(p_op[parent_mask])

        c_op[child_mask] -= np.mean(c_op[child_mask])
        c_op[child_mask] /= np.std(c_op[child_mask])


    """
    Code for averaging conns by parent prior by layer
    """
    parent_class_aves = []
    parents_by_class = []
    parents_aves = []
    conn_aves = []
    parents = []
    
    ### 3d array of average correlations with shape [#classes, #tokens, #parent neurons]
    corr_by_class = []

    if fold_tokens == True:
        for label in list(np.unique(adapter_labels.numpy())):
            ### For faster computation, get covariance over all tokens of the given class
            p_marg = p_op[adapter_labels==label,:,:].reshape(-1, p_op.shape[2])
            c_marg = c_op[adapter_labels==label,:,:].reshape(-1, c_op.shape[2])
            
            ### Parents is a 2D list of all of the connectivities of parents and children for a single class
            coefs = np.corrcoef(p_marg, c_marg, rowvar=False).astype(np.float32)
            parents = []
            ### Loop over the cross correlation matrix for the rows corresponding to the parent layer's filters
            for j in range(0, len(p_marg[0])):
                parents.append(coefs[j, len(p_marg[0]):])

            ### We take the absolute value because we only care about the STRENGTH of the correlation, not the sign
            parents = np.abs(np.asarray(parents))
            corr_by_class.append(parents)


    else:
        for label in list(np.unique(adapter_labels.numpy())):
            token_conns = []
            ### For each token, get the covariance among activations conditioned on image labels
            for i in range(p_op.shape[1]):
                p_marg = p_op[adapter_labels==label,i,:]
                c_marg = c_op[adapter_labels==label,i,:]
                
                # print("Shape of parent and child acts for token {}: {} - {}".format(i, p_marg.shape, c_marg.shape))
                ### Parents is a 2D list of all of the connectivities of parents and children for a single class
                coefs = np.corrcoef(p_marg, c_marg, rowvar=False).astype(np.float32)
                parents = []
                ### Loop over the cross correlation matrix for the rows corresponding to the parent layer's filters
                for j in range(0, len(p_marg[0])):
                    parents.append(coefs[j, len(p_marg[0]):])

                ### We take the absolute value because we only care about the STRENGTH of the correlation, not the sign
                parents = np.abs(np.asarray(parents))
                token_conns.append(parents)
            
            corr_by_class.append(token_conns)        
        
    return corr_by_class
    




### For a given set of MoE-CLIP model activations, get the connectivity for all adapters
def get_conns(acts_dict, labels_dict, conn_type="raw", fold_tokens=False):
    conns_dict = {}
    for modal in ['visual']:
        conns_dict[modal] = {}
        for block_idx in acts[modal].keys():
            conns_dict[modal][block_idx] = {}
            for adapter_idx in acts[modal][block_idx].keys():

                adapter_labels = labels_dict[modal][block_idx][adapter_idx]
                parent_acts = acts_dict[modal][block_idx][adapter_idx]['down_proj']
                child_acts = acts_dict[modal][block_idx][adapter_idx]['up_proj']

                if conn_type == "raw":
                    conns_dict[modal][block_idx][adapter_idx] = calc_conn(parent_acts, child_acts, adapter_labels, fold_tokens)



    return conns_dict

































