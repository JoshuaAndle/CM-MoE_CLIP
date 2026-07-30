from typing import Tuple, Union, Optional
from collections import OrderedDict, Counter

import os
import copy
import numpy as np
import time

from collections import OrderedDict
from math import sqrt

import torch
from torch import nn
import torch.nn.functional as F
from .adapter import Adapter
from torch.distributions.normal import Normal
from collections import Counter


from .sparse_dispatcher import SparseDispatcher
from .adapter import Adapter
from .lora import MultiheadAttention as LoRAMultiheadAttention
from models.DIKI_layers import MultiheadAttention_DIKI



global_subnet_id = None       # Dictates which experts are used for training in CMMoE 
global_val_subnet_id = None   # Dictates which experts are used for eval in CMMoE

global_log_dir = None         # Specifies saving location for activation storage in CMMoE
global_task_id = None         # Specifies current task ID for activation storage in CMMoE
global_acts_dict = {}         # Master dict for collecting stored activations in CMMoE
global_sample_count = 0       # Used to handle indexing for storage of acts across batches


def prepare_acts_dict(task_key: int, blocks: list[int], subnets: list[int], num_samples: int, num_classes:int, store_layers: list[str] = []):
    global global_acts_dict


    acts_dict = {"image": {}, "text": {}}
    for modal in ["image", "text"]:
        for subnet in subnets:
            acts_dict[modal][subnet] = {}
            for block in blocks:
                acts_dict[modal][subnet][block] = {}

                for layer in store_layers:
                    ### Since experts are more flexible in number, we just handle them by appending within forward() if used
                    if layer not in ["expert_down_acts", "expert_up_acts"]:
                        #!# For now we're just hardcoding the embed dim
                        if modal == "image":
                            acts_dict[modal][subnet][block][layer] = torch.zeros((num_samples, 1, 768)) 
                        else:
                            acts_dict[modal][subnet][block][layer] = torch.zeros((num_classes, 1, 512)) 


    global_acts_dict = acts_dict    


def get_acts_dict():
    # global global_acts_dict
    # print(f"Gettings acts dict with type: {type(global_acts_dict)} and keys {global_acts_dict.keys()}", flush=True)
    return global_acts_dict





def set_sample_count(count: int):
    global global_sample_count
    global_sample_count = count


def get_sample_count():
    # global global_sample_count
    return global_sample_count






### Task integer denoting what task is being performed. Solely for debugging purposes of saving activations for subnetworks applied to other tasks
def peft_set_task(task_id, is_train=True):
    if is_train:
        global global_task_id 
        global_task_id = task_id


#*# I think that the val and train subnet IDs are redundant. We should never need them to differ, and they should be consolidated
### Subnet int denoting which routers to use
def peft_set_subnet(subnet_id, is_train=True):
    if is_train:
        global global_subnet_id 
        global_subnet_id = subnet_id
    else:
        global global_val_subnet_id
        global_val_subnet_id = subnet_id


def peft_set_log_dir(log_dir):
    global global_log_dir
    global_log_dir = log_dir



















class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(
                OrderedDict([("-1", nn.AvgPool2d(stride)),
                             ("0",  nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                             ("1", nn.BatchNorm2d(planes * self.expansion))]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):

    def __init__(self,
                 spacial_dim: int,
                 embed_dim: int,
                 num_heads: int,
                 output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim**2 + 1, embed_dim) / embed_dim**0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x,
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False)

        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self,
                 layers,
                 output_dim,
                 heads,
                 input_resolution=224,
                 width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3,
                               width // 2,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2,
                               width // 2,
                               kernel_size=3,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2,
                               width,
                               kernel_size=3,
                               padding=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)






















class ResidualAttentionBlock_DIKI(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, add_prompt=False,
                 text_layer=False, i=0, design_details=None):
        super().__init__()
        # print("Add prompt is: ", add_prompt)
        if add_prompt:
            if text_layer:
                self.attn = MultiheadAttention_DIKI(d_model, n_head, prefix_pool_size=design_details["pool_size"], 
                                                    prefix_len=design_details["language_ctx"])
            else:
                self.attn = MultiheadAttention_DIKI(d_model, n_head, prefix_pool_size=design_details["pool_size"], 
                                                    prefix_len=design_details["vision_ctx"])
        else:
            self.attn = MultiheadAttention_DIKI(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        # Only add learnable tokens if flag is set True
        # For the first iteration i, we should not add the learnable parameters
        # as it is already been taken care of in the very start, for both text
        # and the visual branch
        self.text_layer = text_layer
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, prompt_ids=None, batch_weight=None):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask, prompt_ids=prompt_ids, batch_weight=batch_weight)[0]

    def forward(self, x: torch.Tensor, prompt_ids=None, batch_weight=None):
        x = x + self.attention(self.ln_1(x), prompt_ids, batch_weight)
        x = x + self.mlp(self.ln_2(x))
        return x










class ResidualAttentionBlock(nn.Module):

    def __init__(self,
                 d_model: int,
                 n_head: int,
                 attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict([("c_fc", nn.Linear(d_model, d_model * 4)),
                         ("gelu", QuickGELU()),
                         ("c_proj", nn.Linear(d_model * 4, d_model))]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x





class ResidualAttentionBlock_LoRA(ResidualAttentionBlock):

    def __init__(self,
                 d_model: int,
                 n_head: int,
                 attn_mask: torch.Tensor = None,
                 design_details: dict = {}):
        super().__init__(d_model, n_head, attn_mask)

        self.lora_alpha = design_details.get('lora_alpha', 1)
        self.lora_r = design_details.get('lora_r', 4)

        self.attn = LoRAMultiheadAttention(d_model,
                                           n_head,
                                           lora_alpha=self.lora_alpha,
                                           r=self.lora_r)


class ResidualAttentionBlock_Adapter(ResidualAttentionBlock):

    def __init__(self,
                 d_model: int,
                 n_head: int,
                 attn_mask: torch.Tensor = None,
                 design_details: dict = {}):
        super().__init__(d_model, n_head, attn_mask)

        self.ffn_num = design_details.get('ffn_num', 64)

        # adapter
        self.adaptmlp = Adapter(
            d_model=d_model,
            dropout=0.1,
            bottleneck=self.ffn_num,
            init_option='lora',
            adapter_scalar=0.1,
            adapter_layernorm_option='none',
        )

    def forward(self, x: torch.Tensor):
        x = x + self.adaptmlp(self.attention(self.ln_1(x.clone())))
        x = x + self.adaptmlp(self.mlp(self.ln_2(x.clone())))
        return x


class ResidualAttentionBlock_MoA(ResidualAttentionBlock):

    def __init__(self,
                 d_model: int,
                 n_head: int,
                 attn_mask: torch.Tensor = None,
                 modal=None,
                 block_idx: int = -1,
                 design_details: dict = {},):
        super().__init__(d_model, n_head, attn_mask)


        # args = design_details.get('args')

        self.block_idx = block_idx

        self.store_layers = []


        self.top_k = design_details.get('top_k', 2)
        self.ffn_num = design_details.get('ffn_num', 64)
        self.noisy_gating = design_details.get('noisy_gating', True)

        ### Initial number of experts, and number added/frozen for each task
        self.experts_num = design_details.get('experts_num', 4)
        self.experts_per_task = design_details.get('experts_per_task', 4)
        
        self.softmax = nn.Softmax(1)
        self.softplus = nn.Softplus()
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))

        self.modal = modal
        self.choose_map = torch.zeros([self.experts_num])


        self.d_model=d_model


        self.frozen_experts = {"all":[]}
        self.adaptmlp_list = nn.ModuleList()
        self.router_list = nn.ParameterList()
        self.w_noise_list = nn.ParameterList()
        self.subnet_to_idx = {}    

        ### This is a super hacky way to print and store activation values for one block during only the first batch, placeholder code
        self.forward_verbose = False
        self.store_acts = False

        for i in range(self.experts_num):  #  Expert number
            self.adaptmlp = Adapter(d_model=d_model, dropout=0.1, bottleneck=self.ffn_num,
                                    init_option='lora',
                                    adapter_scalar=0.1,
                                    adapter_layernorm_option='none',
                                    )
            self.adaptmlp_list.append(self.adaptmlp)



    def reset_choosemap(self):
        """
        Resets the choosemap counts for experts
        """
        self.choose_map = torch.zeros([len(self.adaptmlp_list)])


    
    def set_activation_saving(self, store_setting=False):
        ### Queues the block to store activations on the next Forward pass
        self.store_acts = store_setting


    ### Add a new router and adapters when initializing a new subnetwork
    def init_subnet(self, subnet):
        self.init_expert(self.experts_per_task)
        self.router_list.append(nn.Parameter(torch.zeros(self.d_model, len(self.adaptmlp_list)), requires_grad=True))
        self.w_noise_list.append(nn.Parameter(torch.zeros(self.d_model, len(self.adaptmlp_list)), requires_grad=True))
        self.frozen_experts[subnet] = []

        ### Add the new subnet and sort them by value
        current_subnets = list(self.subnet_to_idx.keys())
        new_subnets = list(set(current_subnets + [subnet]))

        ### Get the list idx values for each subnetwork to update mapping dictionary
        new_mappings = {}
        for i in range(len(new_subnets)):
            new_mappings[new_subnets[i]] = i

        self.subnet_to_idx = new_mappings
        if self.block_idx == 0 and self.modal == "image":
            print("New subnet to idx mappings: ", self.subnet_to_idx)


    def init_expert(self, num_experts_added=1):
        ### Add new experts to accomodate learning new task
        for i in range(num_experts_added):
            self.adaptmlp = Adapter(d_model=self.d_model, dropout=0.1, bottleneck=self.ffn_num,
                                        init_option='lora',
                                        adapter_scalar=0.1,
                                        adapter_layernorm_option='none',
                                        )
            self.adaptmlp_list.append(self.adaptmlp)
            self.experts_num += 1

        ### Extend existing routers to account for new experts
        #!# Note that extending routers this way resets gradient data. This is fairly negligible, and for now we dont go back to retrain existing routers
        for i in range(len(self.router_list)):
            original_router = self.router_list[i].data 
            original_noise  = self.w_noise_list[i].data

            extended_router = nn.Parameter(torch.zeros(self.d_model, len(self.adaptmlp_list)), requires_grad=True)
            extended_noise = nn.Parameter(torch.zeros(self.d_model, len(self.adaptmlp_list)), requires_grad=True)

            extended_router.data[:, :original_router.shape[1]] = original_router.data
            extended_noise.data[:, :original_noise.shape[1]] = original_noise.data

            self.router_list[i] = extended_router
            self.w_noise_list[i] = extended_noise


        #!# Should not do this here, just extend. It's muddling function responsibility too much
        ### Reset choose maps, both to extend and to consider only counts for current subnetwork
        self.choose_map = torch.cat((self.choose_map, torch.zeros((num_experts_added))))





    def merge_subnets(self, merged_subnets, all_merged_experts, experts_to_remove, verbose=False):
        """
        Given multiple subnetworks to be merged and a list of lists for the experts belonging to each
        Remove all subnetwork routers except for the first in merged_subnets, which will be finetuned to cover all merged tasks
        Remove experts belonging to all bust the first subnetwork in merged_subnets, which will be finetuned to cover the role for all merged tasks
        Need to downsize remaining routers to reflect expert removal
        Returns the expert mappings so they can be applied within the method class as well
        """
        if verbose and self.block_idx == 0 and self.modal == "image":
            print("Merging Subnetworks in adapter layer")
            print("Pre-merge layer has {} adapters and {} routers. Removing experts: {}".format(len(self.adaptmlp_list), len(self.router_list), experts_to_remove))
            print("Pre-merge size of router: ", self.router_list[0].data.size())
            print("Pre-merge frozen experts: ", self.frozen_experts)




        """
        Need:
        Mappings for subnets to idx before removal 
            - To index the existing lists that are contiguous and remove the right subnet entries
        Mappings for experts after removal:
            - To update frozen expert values accordingly

        All subnets, Remaining subnets, and removed subnets
        All experts, remaining experts, and removed experts
    
        All subnet idxs being removed
        All expert idxs being removed

        """

        merged_subnets.sort()
        ### Get the list of all subnetworks and which are being kept or removed
        all_subnets = self.subnet_to_idx.keys()
        subnets_to_remove = merged_subnets[1:]
        remaining_subnets = [s for s in all_subnets if s not in subnets_to_remove]

        all_merged_subnet_idxs = [self.subnet_to_idx[s] for s in merged_subnets]
        subnet_idxs_to_keep = [self.subnet_to_idx[s] for s in all_subnets if s not in subnets_to_remove]        
        subnet_idxs_to_remove = [self.subnet_to_idx[s] for s in subnets_to_remove]

        subnet_mappings_before_removal = self.subnet_to_idx  # Prepare mapping dicts for subnetworks pre-removal


        ### Get the list of all experts and which are being kept or removed
        all_experts = range(len(self.adaptmlp_list))
        remaining_experts = [all_experts[i] for i in all_experts if i not in experts_to_remove]
        remaining_merged_experts = [exp for exp in all_merged_experts if exp not in experts_to_remove]

        expert_mappings = {} # Remaps remaining experts to a contiguous set of indices
        idx = 0
        for i in all_experts:
            if i not in experts_to_remove:
                expert_mappings[i] = idx
                idx += 1




        ### To ensure that the remaining target subnetwork routes to all of the pooled experts, we composite together
        ###    the weights from all of the routers of merged subnetworks
        merge_target = self.subnet_to_idx[merged_subnets[0]]
        pooled_router = self.router_list[merge_target].data
        pooled_noise  = self.w_noise_list[merge_target].data
        for idx in subnet_idxs_to_remove:
            pooled_router += self.router_list[idx].data
            pooled_noise += self.w_noise_list[idx].data

        self.router_list[merge_target] = nn.Parameter(pooled_router, requires_grad=True)
        self.w_noise_list[merge_target] = nn.Parameter(pooled_noise, requires_grad=True)

        ### Remove the routers and noises for merged subnetworks, leaving the target subnetwork
        self.router_list = nn.ParameterList([self.router_list[i] for i in subnet_idxs_to_keep])
        self.w_noise_list = nn.ParameterList([self.w_noise_list[i] for i in subnet_idxs_to_keep])


        ### Downsize the remaining elements to account for removal of experts deemed redundant
        for idx in range(len(self.router_list)):
            reduced_data = self.router_list[idx].data[:,remaining_experts]
            self.router_list[idx] = nn.Parameter(reduced_data, requires_grad=True)

            reduced_noise_data = self.w_noise_list[idx].data[:,remaining_experts]
            self.w_noise_list[idx] = nn.Parameter(reduced_noise_data, requires_grad=True)


        new_mappings = {} # Remaps remaining subnetworks to a contiguous set of indices
        for i in range(len(remaining_subnets)):
            new_mappings[remaining_subnets[i]] = i
        self.subnet_to_idx = new_mappings






        ### Unlist frozen experts for subnetworks being removed
        for s in subnets_to_remove:
            del self.frozen_experts[s]

        ### Reassign the experts belonging to all pooled subnetworks to the remaining subnetwork in the cluster
        self.frozen_experts[merged_subnets[0]] = remaining_merged_experts

        ### Remap remaining subnet experts to be contiguous
        for k in self.frozen_experts.keys():
            remapped_experts = []
            for e in self.frozen_experts[k]:
                if e not in experts_to_remove:
                    remapped_experts.append(expert_mappings[e]) # Remap any experts that aren't being removed
            self.frozen_experts[k] = remapped_experts


        ### Remove redundant experts
        self.adaptmlp_list = nn.ModuleList([self.adaptmlp_list[e] for e in all_experts if e not in experts_to_remove])

        ### For now we are opting to reset choosemaps upon merging as it significantly impacts expected relative fitness of experts
        ### Note: May consider only resetting merged experts or only removing entries without resetting values of kept experts
        self.experts_num -= len(experts_to_remove)
        self.choose_map = torch.tensor([self.choose_map[e] for e in all_experts if e not in experts_to_remove])



        if verbose and self.block_idx == 0 and self.modal == "image":
            print("Post-merge layer has {} adapters and {} routers. Removing experts: {}".format(len(self.adaptmlp_list), len(self.router_list), experts_to_remove))
            print("Post-merge size of router: ", self.router_list[0].data.size())
            print("Resulting frozen experts: ", self.frozen_experts)
            print("Resulting experts_num: ", self.experts_num)
            print("Resulting expert mappings: ", expert_mappings)



        return expert_mappings



    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.top_k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        normal = Normal(self.mean, self.std)

        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, w_gate, w_noise, subnet, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """

        ### Applies the selected routing matrix
        clean_logits = x @ w_gate.to(x)
        if self.noisy_gating and train:
            raw_noise_stddev = x @ w_noise.to(x)
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        ### Mask the router to avoid routing to experts frozen for other subnetworks
        frozen_mask = torch.zeros(logits.shape).to(x)
        ### If there are frozen experts dedicated to the subnet, only route to them
        if len(self.frozen_experts[subnet]) > 0:
            frozen = [i for i in range(len(self.adaptmlp_list)) if i not in self.frozen_experts[subnet]]

        ### If no experts have been frozen yet for this subnet, use all unfrozen experts 
        else:
            frozen = [i for i in self.frozen_experts["all"]]

        frozen_mask[:, frozen] = float("-inf")
        logits = logits + frozen_mask
        

        ### Gets the top router logits for each sample in the batch, corresponding to which expert to pass the sample to
        top_logits, top_indices = logits.topk(min(self.top_k + 1, self.experts_num), dim=1)

        top_k_logits = top_logits[:, :self.top_k]
        top_k_indices = top_indices[:, :self.top_k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits)

        ### Apply the softmax logits to the corresponding selected top_k indices
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if torch.sum(gates[:, frozen]) > 0:
            print("Error: , sum of frozen gates has non-zero value of {}".format(torch.sum(gates[:, frozen])))

        if self.noisy_gating and self.top_k < self.experts_num and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load




    def forward(self, x: torch.Tensor, is_train=True, eot_indices=None):
        
        start_time = time.time()
        store_time_sum = 0

        batch_size = x.shape[1] # X has shape [Tokens, Batch, Embed]

        global global_acts_dict        
        if self.store_acts:
            # if self.modal == "text":
            #     print("Storing for block: ", self.block_idx, flush=True)
            store_time_start = time.time()
            ### We now just use a shared global dict for task act storage     
            # global global_sample_count
            # global global_val_subnet_id

            store_time_sum += time.time() - store_time_start


        if self.store_acts and "block_input" in self.store_layers:
            store_time_start = time.time()
            acts = x.permute(1,0,2).cpu()
            ### To reduce storage sizes of acts, we only store the CLS or EOT tokens that will be used for metric calculations
            if eot_indices is not None and self.modal == "text":
                acts = acts[torch.arange(acts.shape[0]), eot_indices, :].unsqueeze(1)
            elif self.modal == "image":
                acts = acts[torch.arange(acts.shape[0]), 0, :].unsqueeze(1)


            # if self.modal == "text":
            #     print(f"Shape of storage tensor: {global_acts_dict[self.modal][global_val_subnet_id][self.block_idx]["block_input"][global_sample_count:(global_sample_count+batch_size),:,:].shape} \
            #         Shape of activations: {acts.shape}, eot indices is: {eot_indices}")

            if self.modal == "image" or (self.modal == "text" and global_sample_count == 0):
                global_acts_dict[self.modal][global_val_subnet_id][self.block_idx]["block_input"][global_sample_count:(global_sample_count+batch_size),:,:] = acts



            # if self.block_idx == 11 and self.modal == "image":
            #     temp_acts = global_acts_dict[self.modal][global_val_subnet_id][self.block_idx]["block_input"]
            #     print("Current sample counter: ", global_sample_count,
            #             ", Shape of stored_acts: ", temp_acts.shape,
            #             ", Sum of stored acts: ", torch.sum(temp_acts[:(global_sample_count+batch_size),:,:]), 
            #             ", Sum of unstored acts: ", torch.sum(temp_acts[(global_sample_count+batch_size):,:,:])
            #     )

            store_time_sum += time.time() - store_time_start


        x = x + self.attention(self.ln_1(x))

        ### Original MoE-Adapter method states that it uses zero-shot CLIP when no tasks are in-distribution, so skip adapters
        if is_train == False and global_val_subnet_id == -1:
            x = x + self.mlp(self.ln_2(x))
            return x
        else:            

            x_re = x.permute(1, 0, 2)

            #!# Note: This is routing based on the first token of each sample. This is the cls token for vision, but
            ###    doesnt make as much sense for prompt-based text inputs. Will look into the authors reasoning later. 
            ###    May be fine for text as simple task-specific routing is likely fine, but it's not clear at the moment.
            if is_train == False and global_val_subnet_id != None:
                sub_idx = self.subnet_to_idx[global_val_subnet_id]
                gates, load = self.noisy_top_k_gating(x_re[:,0,:], is_train, self.router_list[sub_idx], self.w_noise_list[sub_idx], global_val_subnet_id)
            elif global_subnet_id != None:
                sub_idx = self.subnet_to_idx[global_subnet_id]
                gates, load = self.noisy_top_k_gating(x_re[:,0,:], is_train, self.router_list[sub_idx], self.w_noise_list[sub_idx], global_subnet_id)
            else:
                print("Incorrect setting in forward call for CLIP model global subnet IDs")

            if self.store_acts:
                store_time_start = time.time()

                if "gates" not in global_acts_dict[self.modal][global_val_subnet_id][self.block_idx].keys():
                    global_acts_dict[self.modal][global_val_subnet_id][self.block_idx]["gates"] = [gates]
                else:
                    global_acts_dict[self.modal][global_val_subnet_id][self.block_idx]["gates"].append(gates)

                if "frozen_experts" not in global_acts_dict[self.modal][global_val_subnet_id][self.block_idx].keys():
                    global_acts_dict[self.modal][global_val_subnet_id][self.block_idx]["frozen_experts"] = [self.frozen_experts[global_subnet_id]]
                else:
                    global_acts_dict[self.modal][global_val_subnet_id][self.block_idx]["frozen_experts"].append(self.frozen_experts[global_subnet_id])
                store_time_sum += time.time() - store_time_start




            importance = gates.sum(0)

            #!# Note: Original MoE-Adapter author never implemented a penalty for this loss
            #!# Not sure we want to either, as we ideally want to partition certain experts to specific subnetworks with less sharing
            loss = self.cv_squared(importance) + self.cv_squared(load)
            loss *= 1e-2 # # Todo

            nonzero_indices = torch.nonzero(gates)
            counter = Counter(nonzero_indices[:, 1].tolist())
            
            if self.forward_verbose and self.block_idx == 0:
                print("nonzero indices counter: ", counter)

            ### Tracking how many times each adapter was routed to
            for number, count in counter.items():                    
                self.choose_map[number] += count

            ### Note: This dispatches a subset of samples to their corresponding top-k selected experts from the gates
            dispatcher = SparseDispatcher(self.experts_num, gates)
            ### Assigning as class variable so we can fetch the routing decisions of each batch sample after forward pass
            self.dispatcher = dispatcher

            expert_inputs = dispatcher.dispatch(x_re.view(x.shape[1], -1))

            # if self.store_acts:
            #     # acts_dict[block_key]["expert_inputs"] = expert_inputs
            #     if "expert_inputs" not in acts_dict[block_key].keys():
            #         acts_dict[block_key]["expert_inputs"] = [expert_inputs]
            #     else:
            #         acts_dict[block_key]["expert_inputs"].append(expert_inputs)

            if self.store_acts and ("expert_down_acts" in self.store_layers or "expert_up_acts" in self.store_layers):

                #!# Changed to return tuples of (output, down_acts, up_acts)
                expert_acts = [self.adaptmlp_list[i](expert_inputs[i].view(expert_inputs[i].shape[0], x.shape[0],x.shape[2]).to(x), 
                                                        add_residual=False, 
                                                        verbose=False if i>0 else self.forward_verbose, collect_acts=True) 
                                                        for i in range(self.experts_num)]  # 11个experts 一个router

                expert_outputs =   [expert_acts[i][0] for i in range(self.experts_num)]

                store_time_start = time.time()
                expert_down_acts = [expert_acts[i][1].cpu() for i in range(self.experts_num)]
                expert_up_acts =   [expert_acts[i][2].cpu() for i in range(self.experts_num)]

                # ## Need to copy outputs since they are modified later during Dispatcher combination of outputs
                # if "expert_outputs" not in acts_dict[block_key].keys():
                #     acts_dict[block_key]["expert_outputs"] = [copy.deepcopy(expert_outputs)]
                # else:
                #     acts_dict[block_key]["expert_outputs"].append(copy.deepcopy(expert_outputs))
                if self.modal == "image" or (self.modal == "text" and global_sample_count == 0):

                    if "expert_down_acts" not in global_acts_dict[self.modal][global_val_subnet_id][self.block_idx].keys():
                        global_acts_dict[self.modal][global_val_subnet_id][self.block_idx]["expert_down_acts"] = [expert_down_acts]
                    else:
                        global_acts_dict[self.modal][global_val_subnet_id][self.block_idx]["expert_down_acts"].append(expert_down_acts)

                    if "expert_up_acts" not in global_acts_dict[self.modal][global_val_subnet_id][self.block_idx].keys():
                        global_acts_dict[self.modal][global_val_subnet_id][self.block_idx]["expert_up_acts"] = [expert_up_acts]
                    else:
                        global_acts_dict[self.modal][global_val_subnet_id][self.block_idx]["expert_up_acts"].append(expert_up_acts)
                store_time_sum += time.time() - store_time_start

            else:
                #!# Changed to return tuples of (output, down_acts, up_acts)
                expert_outputs = [self.adaptmlp_list[i](expert_inputs[i].view(expert_inputs[i].shape[0], x.shape[0],x.shape[2]).to(x), 
                                                        add_residual=False, 
                                                        verbose=False if i>0 else self.forward_verbose) 
                                                        for i in range(self.experts_num)]  # 11个experts 一个router





            i = 0
            while i < len(expert_outputs):
                if expert_outputs[i].shape[0] == 0 :
                    expert_outputs.pop(i)
                else:
                    expert_outputs[i] = expert_outputs[i].view(expert_outputs[i].shape[0],-1)
                    i += 1

            y = dispatcher.combine(expert_outputs)

            y = y.view(x.shape[1],x.shape[0],x.shape[2])
            mlp_output = self.mlp(self.ln_2(x))
            x = x + mlp_output + y.permute(1, 0, 2)


            if self.store_acts and "dispatcher_combined" in self.store_layers:
                store_time_start = time.time()
                if eot_indices is not None and self.modal == "text":
                    ### To reduce storage sizes of acts, we only store the CLS or EOT tokens that will be used for metric calculations
                    acts = y[torch.arange(y.shape[0]), eot_indices, :].unsqueeze(1).cpu()
                elif self.modal == "image":
                    acts = y[torch.arange(y.shape[0]), 0, :].unsqueeze(1).cpu()


                if self.modal == "image" or (self.modal == "text" and global_sample_count == 0):
                    global_acts_dict[self.modal][global_val_subnet_id][self.block_idx]["dispatcher_combined"][global_sample_count:(global_sample_count+batch_size),:,:] = acts

                store_time_sum += time.time() - store_time_start


            ### mlp output is in shape of x, which is (tokens, samples, embed)
            if self.store_acts and "mlp_output" in self.store_layers:
                store_time_start = time.time()
                acts = mlp_output.permute(1,0,2).cpu()
                if eot_indices is not None and self.modal == "text":
                    ### To reduce storage sizes of acts, we only store the CLS or EOT tokens that will be used for metric calculations
                    acts = acts[torch.arange(acts.shape[0]), eot_indices, :].unsqueeze(1)
                    # print("Shape of mlp output acts: ", acts.shape)
                elif self.modal == "image":
                    acts = acts[torch.arange(acts.shape[0]), 0, :].unsqueeze(1)


                if self.modal == "image" or (self.modal == "text" and global_sample_count == 0):
                    global_acts_dict[self.modal][global_val_subnet_id][self.block_idx]["mlp_output"][global_sample_count:(global_sample_count+batch_size),:,:] = acts

                store_time_sum += time.time() - store_time_start





            # acts_dict, acts_path = {}, None

            # acts_dir = os.path.join(global_log_dir, "activations", task_key)
            # os.makedirs(acts_dir, exist_ok=True)
            # acts_path = os.path.join(acts_dir, f"acts_dict.pt")

            # if self.store_acts:
            #     total_time = time.time() - start_time
            #     train_time = total_time - store_time_sum

            #     torch.save(global_acts_dict, acts_path)
            #     print(f"Forward pass train time {train_time} and store time {store_time_sum}")



            #!# Currently just implemented to limit activation collection to one batch for testing purposes
            self.forward_verbose = False
            self.store_acts = False

            return x



































# --------------------------------------------------------
# Implementation of the Dynamic Moe-Adapters, build in 
# the transformer block of Text/Image Encoder
# --------------------------------------------------------

from .dynamic_gate import DynamicGate
from .auto_encoder import AutoEncoder

# from ..src.finetune import iter


eval_task_id_list_text = [-1] * 12
eval_task_id_list_visual = [-1] * 12
eval_zero_shot = False
eval_best_discrepancy = []


def get_val_task_id_text(idx):
    global eval_task_id_list_text
    return eval_task_id_list_text[idx]


def update_val_task_id_text(new_value, idx):
    global eval_task_id_list_text
    eval_task_id_list_text[idx:] = [new_value] * (len(eval_task_id_list_text) - idx)


def get_val_task_id_visual(idx):
    global eval_task_id_list_visual
    return eval_task_id_list_visual[idx]


def update_val_task_id_visual(new_value, idx):
    global eval_task_id_list_visual
    eval_task_id_list_visual[idx:] = [new_value] * (len(eval_task_id_list_visual) - idx)


def get_eval_best_discrepancy():
    global eval_best_discrepancy
    return eval_best_discrepancy


def update_eval_best_discrepancy(new_value):
    global eval_best_discrepancy
    eval_best_discrepancy.append(new_value)
    

def get_eval_zero_shot():
    global eval_zero_shot
    return eval_zero_shot


def update_eval_zero_shot(new_value):
    global eval_zero_shot
    eval_zero_shot = new_value



































class DynResidualAttentionBlock(nn.Module):
    """
    Core algorithm of MoE-adapter++, including Dynamic MoE-Adapters, DEeC & LEAS
    initialised 1 time before training for each task
    """

    def __init__(self,
                 d_model: int,
                 n_head: int,
                 attn_mask: torch.Tensor = None,
                 adapter_flag=False,
                 LEAS_flag=False,
                 args=None,
                 text_or_image=None,
                 i=None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.ln_3 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.layer = i

        # task_id when training
        self.task_id = args.task_id
        self.task_num = args.task_num
        self.current_iteration = 0


        self.d_model = d_model
        self.softmax = nn.Softmax(1)
        self.softplus = nn.Softplus()  
        self.apply_moe = args.apply_moe  # True for training and eval

        self.is_train = args.is_train  # Whether or not in training mode
        # Flags required for training
        self.force_zero_shot = False
        self.force_eval_task_id = None
        if self.is_train is False:
            if args.repeat_train is False:
                self.init = True
            if args.force_val_task_id is not None:
                self.force_eval_task_id = args.force_val_task_id
            if args.force_zero_shot:
                self.force_zero_shot = True
            # build a list to track the acc of val_task_id
            if args.track_val_task_id[self.layer]:
                self.eval_acc_list = []
            if args.track_val_discrepancy[self.layer]:
                self.eval_discrepancy_list = []
            # self.eval_best_sim = 1.
            self.print_eval_batches = args.print_eval_batches

        # expert num
        self.max_expert_num = args.max_expert_num
        self.text_or_image = text_or_image
        if text_or_image == 'text':
            # print('text transformer')
            self.choose_map_text = torch.zeros([self.max_expert_num]) 
            self.data_len = 77 
        else:
            # print('image transformer')
            self.choose_map_image = torch.zeros([self.max_expert_num])
            if args.model == "ViT-B/16":
                self.data_len = 197
            elif args.model == "ViT-L/14@336px":
                self.data_len = 577

        ### ffn num is the lora dimension
        self.ffn_num = args.ffn_num
        self.autorouter = args.autorouter        # Autorouter is true for eval phase
        self.single_router = args.single_router  # False
        self.adapter_flag = adapter_flag         # True for last 6 vision encoders
        self.LEAS_layer_flag = LEAS_flag         # False for all encoders in ViT-B

        self.reconginition_layer = False
        self.last_discrepancy = None  # Record the last discrepancy value
        self.last_contrast = None 
        self.current_discrepancy_train = 0.0
        
        # augmented_zero_shot (not deploy)
        self.augmented_zero_shot = args.augmented_zero_shot
        self.no_tree_strategy = args.no_tree_strategy
        self.without_LEAS = args.without_LEAS

        # setting for mutil-LEAS
        ### False for all layers
        if self.LEAS_layer_flag:
            # discrepancy function
            self.discrepancy_weighted_vector = args.discrepancy_weighted_vector
            # Construct AE as LEAS, use frozen latent feature embedding as input
            self.global_expert_num = None  # current experts num
            # Determine freezing, activation and training of AEs in LEAS
            if text_or_image == 'text':
                self.global_expert_num = args.text_expert_num_list[i]
                self.frozen_expert_num = args.text_expert_num_list[i]
                self.hidden_dims = args.text_AE_hidden_dims
            else:
                self.global_expert_num = args.image_expert_num_list[i]
                self.frozen_expert_num = args.image_expert_num_list[i]
                self.hidden_dims = args.visual_AE_hidden_dims

            # Record the number of experts and routers and block unused experts.
            self.register_buffer("activated_experts_num", torch.tensor([0.0]))
            self.activated_experts_num[0] = float(self.global_expert_num)
            self.register_buffer("activated_router_num", torch.tensor([float(args.task_id)]))
            self.register_parameter('experts_mask',
                                    torch.nn.Parameter(torch.zeros(size=(self.max_expert_num,)),
                                                        requires_grad=False))
            self.experts_mask[:self.global_expert_num] = 1.0

            # expansion signal, i.e. Flag_e in paper
            self.expansion_flag = False
            if args.repeat_train is False:
                self.expansion_flag = True  # Initial tasks do not require expansion
            self.force_expansion = args.force_expansion_list[self.layer]

            # Building Auto-Encoder Sequences for LEAS / DeEC
            self.auto_encoder_list = nn.ModuleList()
            self.mse_loss = torch.tensor(0.)  # # Sum of MSE_loss for the whole layer
            self.z_score = [torch.tensor(0.)] * self.max_expert_num  # z_score for each ae
            # MSE_loss list corresponding to each AE during training
            # To calculate the mean and std value to recored selection preferences
            self.mse_loss_list = [torch.tensor(0.)] * self.max_expert_num  
            # MSE_loss corresponding to each AE during evaluation
            self.eval_mse_list = [torch.tensor(0.)] * self.max_expert_num
            self.use_LEAS_to_eval = args.use_LEAS_to_eval

            if self.use_LEAS_to_eval:
                self.mse_loss_avg_list = nn.ParameterList()
                self.mse_loss_std_list = nn.ParameterList()
                # record discrepancy when training (Practically no application)
                self.discrepancy_list = nn.ParameterList()
                # best_discrepancy_list when inference
                self.previous_discrepancy_list = [torch.tensor(0.)] * self.task_num
            
            for i in range(self.task_num):  # Task number
                # Record selection preferences for LEAS during training
                self.mse_loss_avg_list.append(
                    nn.Parameter(torch.zeros(self.max_expert_num), requires_grad=False))
                self.mse_loss_std_list.append(
                    nn.Parameter(torch.zeros(self.max_expert_num), requires_grad=False))
                self.discrepancy_list.append(
                    nn.Parameter(torch.zeros(self.task_num), requires_grad=False))

            # build AEs for LEAS
            for idx in range(self.max_expert_num):
                auto_encoder = AutoEncoder(
                    # input_dims=self.data_len * d_model,
                    args=args,
                    input_dims=d_model,
                    hidden_dims=self.hidden_dims,
                )
                self.auto_encoder_list.append(auto_encoder)




        # build moe adapter(experts)
        ### True for last 6 ViT-B Vision encoders
        if self.adapter_flag:
            self.global_expert_num = None
            self.reconginition_layer = False  
            # Confirmation of the calculation of the discrepancy
            self.discrepancy_weighted_vector = args.discrepancy_weighted_vector  

            # Thresholds required to construct expansion and zero_shot
            if text_or_image == 'text':
                self.expansion_threshold = args.expansion_threshold_text
                self.zero_shot_threshold = args.zero_shot_threshold_text
                self.global_expert_num = args.text_expert_num_list[i]
                self.frozen_expert_num = args.text_expert_num_list[i]
                # Locating the Reconginition Layer
                ### Note: index only returns the first occurence, so only the first matched layer is set
                if i == args.use_dyn_moe_layer_list_text.index(True):
                    print("Setting a text recognition layer")
                    self.reconginition_layer = True

            else:  # visual transformer
                self.expansion_threshold = args.expansion_threshold_image
                self.zero_shot_threshold = args.zero_shot_threshold_image
                self.global_expert_num = args.image_expert_num_list[i]
                self.frozen_expert_num = args.image_expert_num_list[i]
                # Locating the Reconginition Layer
                if i == args.use_dyn_moe_layer_list_visual.index(True):
                    print("Setting a visual recognition layer")
                    self.reconginition_layer = True
                
            # Record the number of experts and routers and block unused experts.
            self.register_buffer("activated_experts_num", torch.tensor([0.0]))
            self.activated_experts_num[0] = float(self.global_expert_num)
            self.register_buffer("activated_router_num", torch.tensor([float(args.task_id)]))
            self.register_parameter('experts_mask', torch.nn.Parameter(torch.zeros(size=(self.max_expert_num,)), requires_grad=False))
            ### Masks the existing experts for freezing
            self.experts_mask[:self.global_expert_num] = 1.0

            # experts & auto_encoder
            self.adaptmlp_list = nn.ModuleList()

            # Router for Dynamic MoE-Adapters
            self.noisy_gating = args.use_gate_noise  # Whether noise gate 
            self.noise_epsilon = args.gate_noise_epsilon        
            self.adaptive_moe_gate = DynamicGate(
                num_global_experts=self.global_expert_num,
                fp32_gate=False,
                args=args
            )

            # expansion signal, i.e. Flag_e in paper
            self.expansion_flag = False
            if args.repeat_train is False:
                self.expansion_flag = True  # Initial tasks do not require expansion
            self.force_expansion = args.force_expansion_list[self.layer] # Only forced on the visual recognition layer

            # Building Auto-Encoder Sequences for LEAS / DeEC
            self.auto_encoder_list = nn.ModuleList()
            self.mse_loss = torch.tensor(0.)  # # Sum of MSE_loss for the whole layer
            self.z_score = [torch.tensor(0.)] * self.max_expert_num  # z_score for each ae
            # MSE_loss list corresponding to each AE during training
            # To calculate the mean and std value to recored selection preferences
            self.mse_loss_list = [torch.tensor(0.)] * self.max_expert_num  
            # MSE_loss corresponding to each AE during evaluation
            self.eval_mse_list = [torch.tensor(0.)] * self.max_expert_num
            self.use_LEAS_to_eval = args.use_LEAS_to_eval # True during both training and evaluation

            ### True for both training and eval on all layers
            if self.apply_moe:
                if self.single_router is False:  # use expert
                    # build routers
                    self.router_list = nn.ParameterList()
                    self.w_noise_list = nn.ParameterList()
                    self.expert_activate_freq_list = nn.ParameterList()
                    if self.use_LEAS_to_eval:
                        self.mse_loss_avg_list = nn.ParameterList()
                        self.mse_loss_std_list = nn.ParameterList()
                        if self.reconginition_layer:
                            # Recording of discrepancy in training for automated thresholding
                            self.discrepancy_list = nn.ParameterList()
                            # when eval, best_discrepancy_list
                            self.previous_discrepancy_list = [torch.tensor(0.)] * self.task_num
                            self.zs_thre_list = [torch.tensor(0.)] * self.task_num
                    for i in range(self.task_num):  # Task number
                        self.router_list.append(
                            nn.Parameter(torch.zeros(d_model, self.max_expert_num), requires_grad=True))
                        self.w_noise_list.append(
                            nn.Parameter(torch.zeros(d_model, self.max_expert_num), requires_grad=True))
                        self.expert_activate_freq_list.append(
                            nn.Parameter(torch.zeros(self.max_expert_num), requires_grad=False))
                        # Record the mean value of mse_loss for all AEs in the recognition layer
                        if self.use_LEAS_to_eval:
                            self.mse_loss_avg_list.append(
                                nn.Parameter(torch.zeros(self.max_expert_num), requires_grad=False))
                            self.mse_loss_std_list.append(
                                nn.Parameter(torch.zeros(self.max_expert_num), requires_grad=False))
                            if self.reconginition_layer:
                                self.discrepancy_list.append(
                                    nn.Parameter(torch.zeros(self.task_num), requires_grad=False))

                    # build AE & experts
                    for idx in range(self.max_expert_num):
                        self.adaptmlp = Adapter(d_model=d_model,
                                                dropout=0.1,
                                                bottleneck=self.ffn_num,
                                                init_option='lora',
                                                adapter_scalar=0.1,
                                                adapter_layernorm_option='none',
                                                )  # default
                        self.adaptmlp_list.append(self.adaptmlp)
                        if self.text_or_image == "text":
                            hidden_dims = args.text_AE_hidden_dims
                        else:
                            hidden_dims = args.visual_AE_hidden_dims
                        auto_encoder = AutoEncoder(
                            # input_dims=self.data_len * d_model,
                            args=args,
                            input_dims=d_model,
                            hidden_dims=hidden_dims,
                        )
                        self.auto_encoder_list.append(auto_encoder)

                # else:  # one router for all task
                #     self.router1 = nn.Parameter(torch.zeros(d_model, self.max_expert_num), requires_grad=True)
                #     self.w_noise1 = nn.Parameter(torch.zeros(d_model, self.max_expert_num), requires_grad=True)
                #     self.expert_activate_freq_list = nn.ParameterList()
                #     if self.use_LEAS_to_eval:
                #         self.mse_loss_avg_list = nn.ParameterList()
                #         self.mse_loss_std_list = nn.ParameterList()
                #         if self.reconginition_layer:
                #             # Recording of discrepancy in training for automated thresholding
                #             self.discrepancy_list = nn.ParameterList()
                #             # when eval, best_discrepancy_list
                #             self.previous_discrepancy_list = [torch.tensor(0.)] * self.task_num
                #             self.zs_thre_list = [torch.tensor(0.)] * self.task_num
                            
                #     for i in range(self.task_num):  # Task number
                #         # Record expert activation frequency
                #         self.expert_activate_freq_list.append(
                #             nn.Parameter(torch.zeros(self.max_expert_num), requires_grad=False))
                #         # Record the mean value of mse_loss for all AEs in the recognition layer
                #         if self.use_LEAS_to_eval:
                #             self.mse_loss_avg_list.append(
                #                 nn.Parameter(torch.zeros(self.max_expert_num), requires_grad=False))
                #             self.mse_loss_std_list.append(
                #                 nn.Parameter(torch.zeros(self.max_expert_num), requires_grad=False))
                #             if self.reconginition_layer:
                #                 self.discrepancy_list.append(
                #                     nn.Parameter(torch.zeros(self.task_num), requires_grad=False))

                #     for idx in range(self.max_expert_num):
                #         self.adaptmlp = Adapter(d_model=d_model,
                #                                 dropout=0.1,
                #                                 bottleneck=self.ffn_num,
                #                                 init_option='lora',
                #                                 adapter_scalar=0.1,
                #                                 adapter_layernorm_option='none',
                #                                 )  # default
                #         self.adaptmlp_list.append(self.adaptmlp)
                #         if self.text_or_image == "text":
                #             hidden_dims = args.text_AE_hidden_dims
                #         else:
                #             hidden_dims = args.visual_AE_hidden_dims
                #         auto_encoder = AutoEncoder(
                #             args=args,
                #             input_dims=d_model,
                #             hidden_dims=hidden_dims,
                #         )
                #         self.auto_encoder_list.append(auto_encoder)

            else:
                self.adaptmlp = Adapter(d_model=d_model,
                                        dropout=0.1,
                                        bottleneck=self.ffn_num,
                                        init_option='lora',
                                        adapter_scalar=0.1,
                                        adapter_layernorm_option='none',
                                        )  # default
  


















    def set_iteration(self, iteration):
        self.current_iteration = iteration

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor, x_original: torch.Tensor):

        x = x + self.attention(self.ln_1(x)) 
        if self.adapter_flag is False and self.LEAS_layer_flag is False:
            x = x + self.mlp(self.ln_2(x))
            return x, x





        # calculate x_frozen by frozen CLIP
        with torch.no_grad():
            x_original = x_original + self.attention(self.ln_1(x_original))
            x_frozen = x_original.detach()
            x_frozen = x_frozen + self.mlp(self.ln_2(x_frozen))
        zero_shot = self.force_zero_shot

        # Only the LEAS layer is deployed, i.e. the mutil-LEAS setup
        if self.LEAS_layer_flag and zero_shot is False:
            # Confirmation of the number of experts currently active
            current_experts_num = self.global_expert_num
            if self.is_train is False and self.force_eval_task_id is None:  # eval
                self.previous_discrepancy_list = self.get_best_discrepancy_list(x)
            if self.is_train:
                self.get_block_mse_loss_and_z_score(x)
                if self.expansion_flag is False:
                    self.add_LEAS(self.global_expert_num)

        # Dynamic MoE-Adapters
        if self.adapter_flag and zero_shot is False:
            # Confirmation of the number of experts currently active
            current_experts_num = self.global_expert_num

            # When eval get task_id by LEAS
            if self.is_train is False:
                eval_task_id = -1

                # Detection process at the reconginition layer only
                if self.reconginition_layer:
                    # get eval_task_id
                    if self.is_train is False and self.force_eval_task_id is None:  # eval
                        # Use LEAS to get task_id when eval
                        zero_shot, eval_task_id, current_experts_num = self.eval_task_id_LEAS(x_original)  

                    elif self.is_train is False and self.force_eval_task_id is not None:  # eval_force
                        eval_task_id = self.force_eval_task_id
                    # Update eval task_id to all layers
                    if self.text_or_image == "text":
                        update_val_task_id_text(eval_task_id, self.layer)
                    else:
                        update_val_task_id_visual(eval_task_id, self.layer)
                        
                else:
                    # Update eval task_id to all layers
                    if self.text_or_image == "text":
                        eval_task_id = get_val_task_id_text(self.layer)
                    else:
                        eval_task_id = get_val_task_id_visual(self.layer)
                    # # Other layers get zero-shot flag
                    zero_shot = get_eval_zero_shot()
                    
                # Record the number of times the frozen CLIP (zero-shot) was activated
                if zero_shot:
                    self.eval_acc_list.append(-1)
                else:
                    self.eval_acc_list.append(eval_task_id)
            
            # LEAS is not used, and zero-shot is not used by default, 
            # but the LEAS inference results are still retained 
            # for the normal operation of the program
            if self.without_LEAS:
                zero_shot = False

            # MoE
            if self.apply_moe:
                
                # get zero-shot flag
                if self.autorouter and zero_shot:
                    if self.augmented_zero_shot is False:
                        return x_frozen, x_frozen  # zero-shot

                # New data is constantly being added, and in the case of repeat training, multiple expert
                if self.single_router is False:  
                    
                    # cls token
                    x_re = x.permute(1, 0, 2)[:, 0, :]

                    if self.is_train:  # train
                        # get mse_loss & z_score in DEeC
                        if self.no_tree_strategy:
                            self.get_block_mse_loss_and_z_score(x)
                        else:
                            self.get_block_mse_loss_and_z_score(x_original)
                            
                        # calculate discrepancy
                        if self.reconginition_layer:
                                # get discrepancy in training
                                self.get_discrepancy_train(x)

                        # get expansion signal
                        if self.expansion_flag is False:
                            if all(element > self.expansion_threshold for element in
                                    self.z_score[:self.global_expert_num]) or self.force_expansion:
                                self.add_expert(self.global_expert_num)
                                current_experts_num = self.global_expert_num
                        router = self.get_effective_experts(self.router_list[self.task_id], self.experts_mask)
                        w_noise = self.get_effective_experts(self.w_noise_list[self.task_id], self.experts_mask)
                        gates, load = self.adaptive_moe_gate(x_re, router, w_noise, self.global_expert_num)

                    else:  # eval in LEAS
                        if self.without_LEAS:
                            current_experts_num = self.global_expert_num
                        else:
                            current_experts_num = self.eval_update_expert_num(eval_task_id)
                        eval_experts_mask = self.eval_update_expert_mask(current_experts_num)
                        router = self.get_effective_experts(self.router_list[eval_task_id], eval_experts_mask)
                        w_noise = self.get_effective_experts(self.w_noise_list[eval_task_id], eval_experts_mask)
                        gates, load = self.adaptive_moe_gate(x_re, router, w_noise, current_experts_num)

                    # Number of statistical expert activations
                    nonzero_indices = torch.nonzero(gates) 
                    counter = Counter(nonzero_indices[:, 1].tolist())
                    for number, count in counter.items(): 
                        if self.text_or_image == 'text':
                            self.choose_map_text[number] = self.choose_map_text[number] + count
                        else:
                            self.choose_map_image[number] = self.choose_map_image[number] + count
                    
                    # Use router to distribute data according to the current number of experts
                    dispatcher = SparseDispatcher(int(current_experts_num), gates)
                    expert_inputs = dispatcher.dispatch(x.permute(1, 0, 2).view(x.shape[1], -1))
                    expert_outputs = [self.adaptmlp_list[i](expert_inputs[i]
                                                            .view(expert_inputs[i].shape[0], x.shape[0], x.shape[2])
                                                            .to(x), add_residual=False)
                                        for i in range(int(current_experts_num))]  

                    # reshape output
                    i = 0
                    while i < len(expert_outputs):
                        if expert_outputs[i].shape[0] == 0:  
                            expert_outputs.pop(i) 
                        else:
                            expert_outputs[i] = expert_outputs[i].view(expert_outputs[i].shape[0], -1)
                            i += 1

                else:  
                    # mutil-experts with 1 router
                    x_re = x.permute(1, 0, 2)[:, 0, :]  # cls token
                    
                    # training stage
                    if self.is_train: 
                        # get mse_loss & z_score in DEeC
                        if self.no_tree_strategy:
                            self.get_block_mse_loss_and_z_score(x)
                        else:
                            self.get_block_mse_loss_and_z_score(x_original)

                        # get expansion signal
                        if self.expansion_flag is False:
                            if all(element > self.expansion_threshold for element in
                                    self.z_score[:self.global_expert_num]) or self.force_expansion:
                                self.add_expert(self.global_expert_num)
                                current_experts_num = self.global_expert_num
                        router = self.get_effective_experts(self.router1, self.experts_mask)
                        w_noise = self.get_effective_experts(self.w_noise1, self.experts_mask)
                        gates, load = self.adaptive_moe_gate(x_re, router, w_noise, self.task_id)
                        
                        # Number of statistical expert activations
                        nonzero_indices = torch.nonzero(gates)  
                        counter = Counter(nonzero_indices[:, 1].tolist()) 
                        for number, count in counter.items(): 
                            if self.text_or_image == 'text':
                                self.choose_map_text[number] = self.choose_map_text[number] + count
                            else:
                                self.choose_map_image[number] = self.choose_map_image[number] + count
                                
                        # Use router to distribute data according to the current number of experts
                        dispatcher = SparseDispatcher(int(self.global_expert_num), gates)
                        expert_inputs = dispatcher.dispatch(x.permute(1, 0, 2).view(x.shape[1], -1)) 
                        expert_outputs = [self.adaptmlp_list[i](expert_inputs[i].view(expert_inputs[i].shape[0], x.shape[0], x.shape[2]).to(x), add_residual=False) for i in
                                            range(int(self.global_expert_num))] 
                    
                    # infreence stage
                    else:  
                        if self.without_LEAS:
                            current_experts_num = self.global_expert_num
                        else:
                            current_experts_num = self.eval_update_expert_num(eval_task_id)
                        eval_experts_mask = self.eval_update_expert_mask(current_experts_num)
                        router = self.get_effective_experts(self.router1, eval_experts_mask)
                        w_noise = self.get_effective_experts(self.w_noise1, eval_experts_mask)
                        
                        gates, load = self.adaptive_moe_gate(x_re, router, w_noise, current_experts_num)               
                        # Use router to distribute data according to the current number of experts
                        dispatcher = SparseDispatcher(int(current_experts_num), gates)
                        expert_inputs = dispatcher.dispatch(x.permute(1, 0, 2).view(x.shape[1], -1))  #
                        expert_outputs = [self.adaptmlp_list[i](expert_inputs[i].view(expert_inputs[i].shape[0], x.shape[0], x.shape[2]).to(x), add_residual=False) for i in
                                            range(int(current_experts_num))] 
                        
                    # reshape output
                    i = 0
                    while i < len(expert_outputs):
                        if expert_outputs[i].shape[0] == 0:
                            expert_outputs.pop(i)
                        else:
                            expert_outputs[i] = expert_outputs[i].view(expert_outputs[i].shape[0], -1)
                            i += 1
                            
                # Merge the output of each expert
                y = dispatcher.combine(expert_outputs) 
                y = y.view(x.shape[1], x.shape[0], x.shape[2])  # n x 39424 -> n x 77 x 512
                # Record the raw x_frozen of this layer for the input of the next layer of DEeC
                x = x + self.mlp(self.ln_2(x))  
                x = x + y.permute(1, 0, 2)

            else:  # one adapter, no Dynamic MoE-Adapters
                x_re = x.permute(1, 0, 2)
                adapt_x = self.adaptmlp(x_re, add_residual=False).permute(1, 0, 2)
                x = x + self.mlp(self.ln_2(x)) 
                x = x + adapt_x
        
        # original frozen CLIP, zero-shot
        else:  
            x = x + self.mlp(self.ln_2(x)) 
        return x, x_frozen

    def update_freq_activated_experts(self, new_tensor):
        """
        count the time & freq that experts are activated,
        for frozen & check the task ID
        Update the extra tensor e.g.
            new_tensor = torch.tensor([4.0, 5.0, 6.0])
            model.module.transformer.resblocks[i].update_extra_tensor(new_tensor)

        """
        self.expert_activate_freq_list[self.task_id] = new_tensor

    def add_expert(self, add_idx):
        self.experts_mask[add_idx] = 1.0
        self.global_expert_num += 1
        
        # If true, it has already been expanded once under the current task expert
        self.expansion_flag = True 
        # print(
        #     f"{self.text_or_image} layer {self.layer} add a new expert, now expert_num is {int(self.global_expert_num)}")
        print(
            f"In iter {self.current_iteration}, {self.text_or_image} layer {self.layer} add a new expert, now expert_num is {int(self.global_expert_num)}, trigger device: {self.activated_experts_num.device}")
        
        self.activated_experts_num += 1.0
        self.adaptive_moe_gate.update_expert_num(self.global_expert_num)
        
        # Refresh Expert Activation Status Record map
        if self.text_or_image == "text":
            self.choose_map_text = torch.zeros([self.max_expert_num])
        else:
            self.choose_map_image = torch.zeros([self.max_expert_num])
        if self.single_router:
            # Single router only unfreezes the column corresponding to the expert
            self.router1.register_hook(self.grad_hook)
            self.w_noise1.register_hook(self.grad_hook)
            pass

    # Define a hook to modify the gradient
    def grad_hook(self, grad):
        # Create an all-zero matrix with the same shape as grad
        mask = torch.zeros_like(grad)        
        # Keep only the gradient of the k-th column
        mask[:, self.global_expert_num - 1] = 1
        return grad * mask

    def add_LEAS(self, add_idx):
        self.experts_mask[add_idx] = 1.0
        self.global_expert_num += 1
        
        # If true, it has already been expanded once under the current task expert
        self.expansion_flag = True  
        print(
            f"{self.text_or_image} layer {self.layer} add a new LEAS, now LEAS_num is {int(self.global_expert_num)}")
        self.activated_experts_num += 1.0

    def get_block_mse_loss_and_z_score(self, x):
        # cls token
        x_rd = x.permute(1, 0, 2)[:, 0, :]
        self.eval_mse_list = [torch.zeros(size=[], device=x.device)] * self.max_expert_num 
        # update mse_loss & z_score in all AEs
        self.mse_loss = torch.zeros(size=[], device=x.device)
        self.z_score = [torch.zeros(size=[], device=x.device)] * self.max_expert_num
        for i in range(len(self.experts_mask)):
            if self.experts_mask[i] == 1.:
                # mse_loss & z-score in current layer
                mse_loss, z_score = self.auto_encoder_list[i](x_rd)
                self.mse_loss += mse_loss
                self.z_score[i] = z_score
                self.mse_loss_list[i] = mse_loss

                # inference stage 
                if self.is_train is False: 
                    if self.use_LEAS_to_eval:
                        self.eval_mse_list[i] = mse_loss.item()
                    else:
                        self.eval_mse_list[i] = (1 / mse_loss.item())

    def get_effective_experts(self, input, mask):
        """
        Depending on the value of the mask, the corresponding column of input is preserved.
        :param input: input tensor, shape [512, a].
        :param mask: mask tensor with length a only 1 or 0
        :return: tensor after selectively preserving columns.
        """
        # Selective column retention with mask
        effective_experts = input[:, mask.bool()]
        return effective_experts

    def update_mse_loss_avg_list(self, cut_off_rate_new, cut_off_rate_frozen=1.0):
        if self.adapter_flag or self.LEAS_layer_flag:
            new_idx = self.global_expert_num
            current_rd_avg_list = self.mse_loss_avg_list[self.task_id][:new_idx]
            # frozen
            for i in range(self.frozen_expert_num):
                current_mse_loss_avg = self.auto_encoder_list[i].get_avg_init(cut_off_rate_frozen)
                current_rd_avg_list[i] = current_mse_loss_avg

            new_mse_loss_avg = self.auto_encoder_list[new_idx - 1].get_avg_init(cut_off_rate_new)
            current_rd_avg_list[new_idx - 1] = new_mse_loss_avg
            # update
            self.mse_loss_avg_list[self.task_id][:new_idx] = current_rd_avg_list

    def update_mse_loss_std_list(self, cut_off_rate_new, cut_off_rate_frozen=1.0):
        if self.adapter_flag or self.LEAS_layer_flag:
            new_idx = self.global_expert_num
            current_rd_std_list = self.mse_loss_std_list[self.task_id][:new_idx]
            # frozen
            for i in range(self.frozen_expert_num):
                current_mse_loss_std = self.auto_encoder_list[i].get_std_init(cut_off_rate_frozen)
                current_rd_std_list[i] = current_mse_loss_std

            new_mse_loss_std = self.auto_encoder_list[new_idx - 1].get_std_init(cut_off_rate_new)
            current_rd_std_list[new_idx - 1] = new_mse_loss_std
            # update
            self.mse_loss_std_list[self.task_id][:new_idx] = current_rd_std_list

    def _get_ranking(self, lst):
        """
        Get the sorting information of a list
        """
        sorted_lst = sorted(enumerate(lst), key=lambda x: x[1])
        rank = [0] * len(lst)

        for rank_idx, (original_idx, value) in enumerate(sorted_lst):
            rank[original_idx] = rank_idx

        return rank
 
    @torch.no_grad()
    def get_discrepancy_train(self, x):
        """
        Get the discrepancy of each task in training
        """
        if self.task_id == 0:
            # print("Task ID is 0")
            if self.current_iteration == 0:
                # print("Current iteration is 0")

                contrast = torch.tensor([1.0] * self.global_expert_num, device=x.device)
                self.last_discrepancy = 0.0
                self.last_contrast = contrast
            else:
                # print("Current iteration is not 0")
                contrast = self.last_contrast
        else:
            # print("Task ID is not 0")
            if self.current_iteration == 0:
                # print("Current iteration is 0")
                contrast = torch.tensor([1.0] * self.global_expert_num, device=x.device)
                self.last_discrepancy = 0.0
                self.last_contrast = contrast
            else:
                # print("Current iteration is not 0")
                contrast = self.last_contrast
            
        if self.discrepancy_weighted_vector == "STD" or self.discrepancy_weighted_vector == "STD_norm":
            std_list = self.mse_loss_std_list
        eval_mse_loss = torch.tensor(self.mse_loss_list, device=x.device)
        eval_mse_loss = self._trim_zeros(eval_mse_loss)
        tensor = contrast  # 参考值
        # print("Layer ", self.layer, " Current global_expert_num: ", self.global_expert_num, " and contrast: ", tensor)
        i = self.task_id
        tensor = self._trim_zeros(tensor)
        current_discrepancy = 1000.
        if tensor is not None:
            tensor_eval = eval_mse_loss[:len(tensor)]
            contrast_tensor = torch.maximum(tensor, tensor_eval)
            if self.discrepancy_weighted_vector == "CON":
                current_discrepancy = self.discrepancy_con(tensor, tensor_eval, contrast_tensor)
            elif self.discrepancy_weighted_vector == "CON_norm":
                current_discrepancy = self.discrepancy_con_norm(tensor, tensor_eval, contrast_tensor)
            elif self.discrepancy_weighted_vector == "STD":
                current_discrepancy = self.discrepancy_std(tensor, tensor_eval, contrast_tensor, std_list[i][:len(tensor)])
            elif self.discrepancy_weighted_vector == "STD_norm":
                current_discrepancy = self.discrepancy_std_norm(tensor, tensor_eval, contrast_tensor, std_list[i][:len(tensor)])
            elif self.discrepancy_weighted_vector == "Non_factor":
                current_discrepancy = self.discrepancy_non(tensor, tensor_eval, contrast_tensor)
            else:
                current_discrepancy = torch.norm((tensor - tensor_eval) / (contrast_tensor * contrast_tensor), p=2) / sqrt(len(tensor))
            if current_discrepancy > self.last_discrepancy:
                self.last_discrepancy = current_discrepancy
                self.last_contrast = tensor_eval
            if self.task_id == 0:
                self.current_discrepancy_train = current_discrepancy / self.global_expert_num
            else:
                self.current_discrepancy_train = current_discrepancy         

    def eval_task_id_LEAS(self, x):
        """
        Get the task_id corresponding to the eval batch by its discrepancy
        """
        zero_shot = False
        update_eval_zero_shot(False)
        eval_task_id = -1
        length = 0
        std_list = None
        # get mse_loss & z_score
        self.get_block_mse_loss_and_z_score(x)
        best_discrepancy = 1000.
        contrast_list = self.mse_loss_avg_list
        if self.discrepancy_weighted_vector == "STD" or self.discrepancy_weighted_vector == "STD_norm":
            std_list = self.mse_loss_std_list
        eval_mse_loss = torch.tensor(self.eval_mse_list, device=x.device)
        eval_mse_loss = self._trim_zeros(eval_mse_loss)
        for i, tensor in enumerate(contrast_list):
            tensor = self._trim_zeros(tensor)
            
            current_discrepancy = 1000.
            if tensor is not None:
                tensor_eval = eval_mse_loss[:len(tensor)]
                contrast_tensor = torch.maximum(tensor, tensor_eval)
                if self.discrepancy_weighted_vector == "CON":
                    current_discrepancy = self.discrepancy_con(tensor, tensor_eval, contrast_tensor)
                elif self.discrepancy_weighted_vector == "CON_norm":
                    current_discrepancy = self.discrepancy_con_norm(tensor, tensor_eval, contrast_tensor)
                elif self.discrepancy_weighted_vector == "STD":
                    current_discrepancy = self.discrepancy_std(tensor, tensor_eval, contrast_tensor, std_list[i][:len(tensor)])
                elif self.discrepancy_weighted_vector == "STD_norm":
                    current_discrepancy = self.discrepancy_std_norm(tensor, tensor_eval, contrast_tensor, std_list[i][:len(tensor)])
                elif self.discrepancy_weighted_vector == "Non_factor":
                    current_discrepancy = self.discrepancy_non(tensor, tensor_eval, contrast_tensor)
                else:
                    current_discrepancy = torch.norm((tensor - tensor_eval) / (contrast_tensor * contrast_tensor), p=2) / sqrt(len(tensor))
                current_discrepancy = current_discrepancy + self.previous_discrepancy_list[i]
                
        
            # Update the maximum sum and the corresponding index
            if current_discrepancy < best_discrepancy:
                best_discrepancy = current_discrepancy
                eval_task_id = i

        if best_discrepancy < self.zero_shot_threshold:
            best_rd_list = contrast_list[eval_task_id]
            best_rd_list = self._trim_zeros(best_rd_list)
            length = len(best_rd_list)

        else:
            zero_shot = True
            update_eval_zero_shot(True)
            length = 0

        if self.print_eval_batches:
            print(best_discrepancy, eval_task_id, torch.argsort(eval_mse_loss))
        self.eval_discrepancy_list.append(best_discrepancy.item())
        return zero_shot, eval_task_id, length

    def discrepancy_con(self, tensor, tensor_eval, contrast_tensor):
        return torch.norm((tensor - tensor_eval) / (contrast_tensor * contrast_tensor), p=2) / sqrt(len(tensor))
    
    def discrepancy_non(self, tensor, tensor_eval, contrast_tensor):
        return torch.norm((tensor - tensor_eval) / (contrast_tensor), p=2) / sqrt(len(tensor))

    def discrepancy_con_norm(self, tensor, tensor_eval, contrast_tensor):
        reciprocal_tensor = 1.0 / contrast_tensor
        norm_tensor = reciprocal_tensor / reciprocal_tensor.sum()
        return torch.norm((tensor - tensor_eval) * norm_tensor / contrast_tensor, p=2) / sqrt(len(tensor))

    def discrepancy_std(self, tensor, tensor_eval, contrast_tensor, std):
        return torch.norm((tensor - tensor_eval) / (std * contrast_tensor), p=2) / sqrt(len(tensor)) / 1000

    def discrepancy_std_norm(self, tensor, tensor_eval, contrast_tensor, std):
        reciprocal_tensor = 1.0 / std
        norm_tensor = reciprocal_tensor / reciprocal_tensor.sum()
        return torch.norm((tensor - tensor_eval) * norm_tensor / contrast_tensor, p=2) / sqrt(len(tensor))

    def get_best_discrepancy_list(self, x):
        std_list = None
        # clear list
        discrepancy_list = [torch.tensor(0.).to(x)] * self.task_num
        # get mse_loss & z_score
        self.get_block_mse_loss_and_z_score(x)
        contrast_list = self.mse_loss_avg_list
        eval_mse_loss = torch.tensor(self.eval_mse_list, device=x.device)
        eval_mse_loss = self._trim_zeros(eval_mse_loss)
        if self.discrepancy_weighted_vector == "STD" or self.discrepancy_weighted_vector == "STD_norm":
            std_list = self.mse_loss_std_list
        for i, tensor in enumerate(contrast_list):
            tensor = self._trim_zeros(tensor)
            if tensor is not None:
                tensor_eval = eval_mse_loss[:len(tensor)]
                contrast_tensor = torch.maximum(tensor, tensor_eval)
                if self.discrepancy_weighted_vector == "CON":
                    current_discrepancy = self.discrepancy_con(tensor, tensor_eval, contrast_tensor)
                elif self.discrepancy_weighted_vector == "CON_norm":
                    current_discrepancy = self.discrepancy_con_norm(tensor, tensor_eval, contrast_tensor)
                elif self.discrepancy_weighted_vector == "STD":
                    current_discrepancy = self.discrepancy_std(tensor, tensor_eval, contrast_tensor, std_list[i][:len(tensor)])
                elif self.discrepancy_weighted_vector == "STD_norm":
                    current_discrepancy = self.discrepancy_std_norm(tensor, tensor_eval, contrast_tensor,
                                                                            std_list[i][:len(tensor)])
                else:
                    current_discrepancy = torch.norm((tensor - tensor_eval) / (contrast_tensor * contrast_tensor),
                                                    p=2) / sqrt(len(tensor))
                discrepancy_list[i] = current_discrepancy
        return discrepancy_list

    def _trim_zeros(self, tensor):
        """
        Intercepts the portion of the tensor that is not preceded by 0.

        Params:
        tensor (torch.Tensor): the input one-dimensional tensor

        Returns:
        torch.Tensor: the intercepted tensor
        """
        # Find the index of all non-zero elements
        non_zero_indices = torch.nonzero(tensor).squeeze()

        if len(non_zero_indices) > 0:
            first_zero_index = non_zero_indices[-1].item() + 1
            return tensor[:first_zero_index]
        else:
            return None

    def eval_update_expert_mask(self, current_experts_num):
        mask = self.experts_mask.clone()
        mask[current_experts_num:] = 0.
        return mask

    def eval_update_expert_num(self, eval_task_id):
        contrast_list = self.expert_activate_freq_list[eval_task_id]
        contrast_list = self._trim_zeros(contrast_list)
        return len(contrast_list)

    def get_convergence_value(self, tensor, method='slide_window', kernel_size=3, alpha=0.2):
        """
        Smooths the input 1D tensor, and returns the last value after smoothing.

        Parameters.
        tensor (torch.Tensor): the input one-dimensional tensor
        method (str): smoothing method, supports 'slide_window' and 'ewma'.
        kernel_size (int): window size of the moving average (only works with ‘moving_average’ method)
        alpha (float): smoothing factor (only available in 'ewma' method)

        Returns: torch.
        torch.Tensor: last value of the smoothed tensor
        """
        if method == 'slide_window':
            # Moving Average (using convolution)
            if kernel_size > len(tensor):
                raise ValueError("kernel_size must be smaller than or equal to the length of the tensor.")

            kernel = torch.ones(kernel_size) / kernel_size
            kernel = kernel.view(1, 1, -1)

            tensor_unsqueezed = tensor.view(1, 1, -1)  
            smoothed_tensor = F.conv1d(tensor_unsqueezed, kernel, padding=kernel_size // 2)
            smoothed_tensor = smoothed_tensor.view(-1)  

            return smoothed_tensor[-1]  

        elif method == 'ewma':
            smoothed_tensor = torch.zeros_like(tensor)
            smoothed_tensor[0] = tensor[0]  #

            for i in range(1, len(tensor)):
                smoothed_tensor[i] = alpha * tensor[i] + (1 - alpha) * smoothed_tensor[i - 1]

            return smoothed_tensor[-1] 
        
        

