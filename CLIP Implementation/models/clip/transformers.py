from typing import Tuple, Union, Optional
from collections import OrderedDict, Counter

import os
import copy
import numpy as np
import time

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


from .peft import ResidualAttentionBlock_MoA, ResidualAttentionBlock_Adapter, ResidualAttentionBlock_DIKI, ResidualAttentionBlock_LoRA, ResidualAttentionBlock, LayerNorm



class Transformer(nn.Module):

    def __init__(self,
                 width: int,
                 layers: int,
                 heads: int,
                 attn_mask: torch.Tensor = None,
                 design_details: dict = {},
                 modal='text'):
        super().__init__()

        args = design_details.get('args')

        self.width = width
        self.layers = layers
        self.modal = modal
        self.res_type = design_details.get('peft_method', 'vanilla')
        peft_flag = design_details.get('peft_encoder','none') in ['both', modal]
        if modal == "text":
            self.peft_blocks = design_details.get('adapter_blocks_text', [True*12])
        else:
            self.peft_blocks = design_details.get('adapter_blocks_image', [True*12])

        if self.res_type == 'moe' and peft_flag:
            self.resblocks = nn.ModuleList([])
            for idx in range(layers):
                if self.peft_blocks[idx] == True:
                    self.resblocks.append(ResidualAttentionBlock_MoA(width, heads, attn_mask, modal, idx, design_details))
                else:
                    print(f"Adding vanilla resblock for idx {idx} in modal {modal}")
                    # self.resblocks.append(ResidualAttentionBlock_MoA(width, heads, attn_mask, modal, idx, design_details))
                    self.resblocks.append(ResidualAttentionBlock(width, heads, attn_mask))

        elif self.res_type == 'adapter' and peft_flag:
            self.resblocks = nn.Sequential(*[ResidualAttentionBlock_Adapter(width, heads, attn_mask, design_details) for _ in range(layers)])
        elif self.res_type == 'lora' and peft_flag:
            self.resblocks = nn.Sequential(*[ResidualAttentionBlock_LoRA(width, heads, attn_mask, design_details) for _ in range(layers)])
        else:
            self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])





    def forward(self, x: torch.Tensor, is_train=True, eot_indices=None, prompt_ids=None, batch_weight=None):
        if self.res_type == "moe":
            for i, block in enumerate(self.resblocks):
                if self.peft_blocks[i] == True:
                    x = block(x, is_train, eot_indices)
                else:
                    x = block(x)
            return x
        else:
            return self.resblocks(x)






class VisualTransformer(nn.Module):

    def __init__(self,
                 input_resolution: int,
                 patch_size: int,
                 width: int,
                 layers: int,
                 heads: int,
                 output_dim: int,
                 modal=None,
                 design_details: dict = {}):
        super().__init__()

        # args = design_details.get('args')

        self.input_resolution = input_resolution
        self.output_dim = output_dim
        # Added so this info is available. should not change anything.
        self.patch_size = patch_size
        self.width = width
        self.layers = layers
        self.heads = heads

        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=width,
                               kernel_size=patch_size,
                               stride=patch_size,
                               bias=False)

        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size)**2 + 1, width))
        self.ln_pre = LayerNorm(width)

        #*# Needs to add adapter flag and text_or_image
        self.transformer = Transformer(width,
                                       layers,
                                       heads,
                                       modal=modal,
                                       design_details=design_details)



        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor, is_train=True):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), 
                        x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND [197, 64, 768]

        x = self.transformer(x, is_train)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x










class Transformer_DIKI(nn.Module):

    def __init__(self,
                 width: int,
                 layers: int,
                 heads: int,
                 attn_mask: torch.Tensor = None,
                 prompts_needed=0,
                 text_layer=False,
                 design_details: dict = {},
                 modal='text'):
        super().__init__()

        # args = design_details.get('args')

        self.width = width
        self.layers = layers
        self.modal = modal
        self.res_type = design_details.get('peft_method', 'vanilla')
        peft_flag = design_details.get('peft_encoder','none') in ['both', modal]
        # print("prompts needed: ", prompts_needed)
        # if "pool_size" in design_details.keys():
        #     print("Transformer DIKI design details pool size: ", design_details["pool_size"])
        if self.res_type == 'moe' and peft_flag:
            self.resblocks = nn.ModuleList([ResidualAttentionBlock_MoA(width, heads, attn_mask, modal, idx, design_details) for idx in range(layers)])
        elif self.res_type == 'diki' and peft_flag:
            self.resblocks = nn.Sequential(*[ResidualAttentionBlock_DIKI(width, heads, attn_mask, True,
                                                                        text_layer, i, design_details) if prompts_needed > i
                                            else ResidualAttentionBlock_DIKI(width, heads, attn_mask, False,
                                                                            text_layer, i, design_details)
                                            for i in range(layers)])        
        elif self.res_type == 'adapter' and peft_flag:
            self.resblocks = nn.Sequential(*[ResidualAttentionBlock_Adapter(width, heads, attn_mask, design_details) for _ in range(layers)])
        elif self.res_type == 'lora' and peft_flag:
            self.resblocks = nn.Sequential(*[ResidualAttentionBlock_LoRA(width, heads, attn_mask, design_details) for _ in range(layers)])
        else:
            self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])





    def forward(self, x: torch.Tensor, prompt_ids=None, batch_weight=None):
        if self.res_type == "diki":
            for module in self.resblocks._modules.values():
                x = module(x, prompt_ids, batch_weight)
            return x
        else:
            return self.resblocks(x)






class VisualTransformer_DIKI(nn.Module):

    def __init__(self,
                 input_resolution: int,
                 patch_size: int,
                 width: int,
                 layers: int,
                 heads: int,
                 output_dim: int,
                 modal=None,
                 design_details: dict = {}):
        super().__init__()

        # args = design_details.get('args')

        self.input_resolution = input_resolution
        self.output_dim = output_dim
        # Added so this info is available. should not change anything.
        self.patch_size = patch_size
        self.width = width
        self.layers = layers
        self.heads = heads

        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=width,
                               kernel_size=patch_size,
                               stride=patch_size,
                               bias=False)

        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size)**2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.prompt_till_layer_visual = design_details["vision_depth"]

        #*# Needs to add adapter flag and text_or_image
        self.transformer = Transformer_DIKI(width,
                                           layers,
                                           heads,
                                           prompts_needed=self.prompt_till_layer_visual,
                                           design_details=design_details)



        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor, prompt_ids=None, batch_weight=None):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), 
                        x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND [197, 64, 768]

        x = self.transformer(x, prompt_ids, batch_weight)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x




















from .peft import DynResidualAttentionBlock, update_val_task_id_text, update_val_task_id_visual


class Dyn_MoE_Transformer(nn.Module):
    """
    To match the structure of MoE-Adapter++,
    change the output of the transformer to two outputs
    """
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, adapter_flag=True,
                 args=None, text_or_image=None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.adapter_flag = adapter_flag
        self.dyn_moe = args.dyn_moe
        self.layer_lock = args.force_layer_lock
        
        # if true, only use the discrepancy for last recognition layer 
        ### Not used
        self.mutil_LEAS_lock = args.mutil_LEAS_lock 
        
        #!# For eval only calls, use leas to eval is true, but use_LEAS_list is false for all encoders
        self.use_LEAS_to_eval = args.use_LEAS_to_eval
        if text_or_image == "text":
            self.layer_adapter_flag_list = args.use_dyn_moe_layer_list_text
            self.LEAS_flag_list = args.use_LEAS_list_text
        else:
            self.layer_adapter_flag_list = args.use_dyn_moe_layer_list_visual
            self.LEAS_flag_list = args.use_LEAS_list_visual


        if self.dyn_moe:
            self.resblocks = nn.Sequential(
                *[DynResidualAttentionBlock(
                    d_model=width,
                    n_head=heads,
                    attn_mask=attn_mask,
                    adapter_flag=self.layer_adapter_flag_list[i],
                    LEAS_flag=self.LEAS_flag_list[i],
                    args=args,
                    text_or_image=text_or_image,
                    i=i) for i in
                  range(layers)])
        # else:
        #     self.resblocks = nn.Sequential(
        #         *[ResidualAttentionBlock(width, heads, attn_mask, True, args, text_or_image, i) for i in
        #           range(layers)])
        #     # self.resblocks = nn.Sequential(
        #         # *[ResidualAttentionBlock(width, heads, attn_mask, self.layer_adapter_flag_list[i], args, text_or_image, i) for i in
        #         #   range(layers)])







    def forward(self, x: torch.Tensor):

        if self.dyn_moe:
            #!# This seems to directly conflict with the evaluation code which sets the val_task_id in dyn_moe_encode_text
            # Refresh the task_id of each batch.
            update_val_task_id_visual(-1, 0)
            update_val_task_id_text(-1, 0)
            x_original = x.clone()
            for i in range(self.layers):
                if self.resblocks[i].reconginition_layer and self.mutil_LEAS_lock is False:
                    self.trans_previous_discrepancy_to_reconginition_layer()
                x, x_original = self.resblocks[i](x, x_original)
            output = x
            output_original = x_original

            ### Layer lock is False for experiments with ViT-B
            # if self.layer_lock:    
            # # Lock all previous layers that have not had an expert added to them and only allow the later layers to expand
            #     for i in range(self.layers - 1):
            #         if self.resblocks[i].expansion_flag is False and self.resblocks[i+1].expansion_flag:
            #             for j in range(i + 1):
            #                 self.resblocks[j].expansion_flag = True  
            #             print(f"lock layers before the {self.resblocks[i].text_or_image} layer {i}")
        else:
            output = self.resblocks(x)
            output_original = output
        return output, output_original




    def trans_previous_discrepancy_to_reconginition_layer(self):
        result = []
        previous_list = []
        for i in range(self.layers):
            if self.resblocks[i].LEAS_layer_flag:
                previous_list.append(self.resblocks[i].previous_discrepancy_list)
            if self.resblocks[i].reconginition_layer and len(previous_list) != 0:
                # Iterate over each position and add together the tensor at the corresponding position
                for idx in range(len(previous_list[0])):
                    sum_tensor = sum(lst[idx] for lst in previous_list)
                    result.append(sum_tensor)
                self.resblocks[i].previous_discrepancy_list = result










class Dyn_MoE_VisualTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,
                 args, text_or_image=None):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        # Added so this info is available. should not change anything.
        self.patch_size = patch_size
        self.width = width
        self.layers = layers
        self.heads = heads
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Dyn_MoE_Transformer(width, layers, heads, adapter_flag=True, args=args, text_or_image=text_or_image)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x, x_original = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x_original = x_original.permute(1, 0, 2)  # LND -> NLD
        
        x = self.ln_post(x[:, 0, :])
        x_original = self.ln_post(x_original[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj
            x_original = x_original @ self.proj

        return x, x_original

