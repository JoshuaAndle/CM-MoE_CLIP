import os.path as osp
import os
import json
import statistics

import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import warnings
from typing import Callable, List, Optional, Tuple, Union

from torch import Tensor
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.parameter import Parameter
from torch.nn.modules import Module





# from models.clip import clip
from models.clip.tokenizer import SimpleTokenizer as _Tokenizer

from torch.distributions.multivariate_normal import MultivariateNormal

_tokenizer = _Tokenizer()


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, indices, batch_weight=None):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, indices, batch_weight)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


#!# The original promptprocessor was set up to handle all tasks' classes with corresponding templates
### We have reworked it to reflect the current exposed classes. This is to keep in line with other methods constraints
###   requiring classification against all known classes instead of just the current task
class PromptProcessor(nn.Module):
    def __init__(self, cfg, texts, text_tokens, clip_model):
        super().__init__()

        dtype = clip_model.dtype
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.input_size[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"


        self.n_cls = len(texts)
        self.class_ids_per_task = [self.n_cls]
        self.class_ids_per_task = [texts]
        self.cur_n_cls = self.n_cls

        self.classnames = texts
        self.classnames = [name.replace("_", " ") for name in self.classnames]
        self.all_name_lens = [len(_tokenizer.encode(name)) for name in self.classnames]
        # print("Setting up promptprocessor text_tokens type: ", type(text_tokens))
        self.register_buffer("all_tokenized_prompts", text_tokens)
        with torch.no_grad():
            # self.all_embedding = clip_model.token_embedding(self.all_tokenized_prompts).type(clip_model.dtype)
            self.register_buffer("all_embedding", clip_model.token_embedding(self.all_tokenized_prompts).type(clip_model.dtype))
            # self.register_buffer("all_embedding", clip_model.token_embedding(text_tokens).type(clip_model.dtype))
        # init with all classes, but will be updated before training and testing
        self.register_buffer("token_prefix", self.all_embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", self.all_embedding[:, 1:, :])  # CLS, EOS
        self.register_buffer("tokenized_prompts", self.all_tokenized_prompts.clone())


    def forward(self, indices):
        ### This is always 1 in the provided code whether or not a task-id is provided in the forward pass
        batch_size = indices.size(0)
        if batch_size > 1:
            print("PromptProcessor Batch Size in forward pass is: ", batch_size)
        prefix = self.token_prefix.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # [bs, n_cls, 1, ctx_dim]
        suffix = self.token_suffix.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # [bs, n_cls, ..., ctx_dim]
        prompts = torch.cat([prefix, suffix], dim=2)  # [bs, n_cls, 77, ctx_dim]
        prompts = prompts.view(batch_size*self.cur_n_cls, prompts.size(2), prompts.size(3))  # [bs*n_cls, 77, ctx_dim]
        tokenized_prompts = self.tokenized_prompts.unsqueeze(0).repeat(batch_size, 1, 1).view(batch_size*self.cur_n_cls, -1)  # [bs*n_cls, 77, tkn_dim]
        return prompts, tokenized_prompts
    


    ### Modified to directly hand in preprocessed classes and token names for compatibility with existing code
    #!# Swapped from taking new task ID to swap between sets of classes to taking direct inputs
    def update_classnames(self, texts, text_tokens, clip_model):
        ### Update variables
        self.n_cls = len(texts)
        self.class_ids_per_task = [self.n_cls]
        self.class_ids_per_task = [texts]
        self.cur_n_cls = self.n_cls

        self.classnames = texts
        self.classnames = [name.replace("_", " ") for name in self.classnames]
        self.all_name_lens = [len(_tokenizer.encode(name)) for name in self.classnames]

        ### Update buffers
        # self.all_tokenized_prompts = text_tokens.detach().cpu()
        self.all_tokenized_prompts = text_tokens
        self.all_embedding = clip_model.token_embedding(self.all_tokenized_prompts).type(clip_model.dtype)
        self.token_prefix = self.all_embedding[:, :1, :]
        self.token_suffix = self.all_embedding[:, 1:, :]
        self.tokenized_prompts = self.all_tokenized_prompts.clone()
        self.name_lens = self.all_name_lens
        # self.cur_n_cls = len(class_idx)


class CustomCLIP(nn.Module):
    def __init__(self, cfg, texts, text_tokens, clip_model, clip_model_ori=None):
        super().__init__()
        self.prompt_processor = PromptProcessor(cfg, texts, text_tokens, clip_model)
        self.image_encoder = clip_model.visual
        self.image_encoder_ori = clip_model_ori.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.vis_dim = clip_model.visual.output_dim


        self.pool_size = cfg.n_tasks
        self.visual_prompt = cfg.prompt_depth_vision > 0
        self.batchwise_prompt = cfg.batchwise_prompt
        self.token_embedding = clip_model.token_embedding

        self.register_buffer("means", torch.empty(self.pool_size, self.vis_dim, dtype=torch.float))
        self.register_buffer("covars", torch.empty(self.pool_size, self.vis_dim, self.vis_dim, dtype=torch.float))
        self.register_buffer("task_learnt", torch.tensor(0, dtype=torch.int))


        self.running_mean = 0.0
        self.running_covars = 0.0


    ### task_ids is a list of per-sample IDs provided by their batch parser
    def forward(self, image, task_ids=None):
        res = {}
        batch_weight = None
        text_batch_weight = None

        with torch.no_grad():
            image_features = self.image_encoder_ori(image.type(self.dtype))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            res["image_features"] = image_features.detach()
        
        if task_ids is not None:
            task_ids = task_ids.type(torch.int).to(image.device)
            assert (task_ids == task_ids[0]).all()
            indices = task_ids[0:1]
            indices = indices.unsqueeze(1)  # size [1, 1]
        else:
            dists = [MultivariateNormal(self.means[i], self.covars[i]) for i in range(self.task_learnt.item())]
            log_probs = torch.vstack([dist.log_prob(image_features) for dist in dists]).t()   # [bs, cur_learnt_task_num]
            topk, indices = log_probs.topk(k=1, dim=1)  # [bs, selected_prompt_num]
            exp_part = topk.squeeze(1)/512-1.0
            batch_weight = torch.sigmoid(exp_part)  # [bs]
            text_batch_weight = batch_weight.mean(dim=0, keepdim=True).repeat(self.prompt_processor.cur_n_cls)
            res["text_batch_weight"] = text_batch_weight[0].item()
            res["raw_indices"] = indices
            if self.batchwise_prompt:
                prompt_id, id_counts = torch.unique(indices, return_counts=True, sorted=True)
                _, major_idx = torch.topk(id_counts, k=1)
                indices = prompt_id[major_idx]
                indices = indices.unsqueeze(0)  # [1, selected_prompt_num]
        
        res["indices"] = indices

        prompts, tokenized_prompts = self.prompt_processor(indices)  # [bs*n_cls, 77, ctx_dim]
        text_features = self.text_encoder(prompts, tokenized_prompts, indices, text_batch_weight)  # [bs*n_cls, model_dim]
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        if self.visual_prompt:
            # print("Using image encoder with grad", flush=True)
            image_features = self.image_encoder(image.type(self.dtype), indices, batch_weight)  # [bs, model_dim]
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        logit_scale = self.logit_scale.exp()
        if indices.size(0) == 1:
            logits = logit_scale * image_features @ text_features.t()  # [bs, n_cls]
        #!# My understanding from DIKI's code is that they deprecated any cases of # indices being greater than 1
        else:
            text_features_resize = text_features.view(image_features.size(0), -1, text_features.size(1))  # [bs, n_cls, model_dim]
            image_features_resize = image_features.unsqueeze(1)  # [bs, 1, model_dim]
            logits = logit_scale * image_features_resize @ text_features_resize.permute(0, 2, 1)  # [bs, 1, n_cls]
            logits = logits.squeeze(1)  # [bs, n_cls]
        res["outputs"] = logits
        
        return res

    
    def update_classnames(self, texts, text_tokens):
        self.prompt_processor.update_classnames(texts, text_tokens, self)






def _in_projection_packed(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w: Tensor,
    b: Optional[Tensor] = None,
) -> List[Tensor]:
    E = q.size(-1)
    if k is v:
        if q is k:
            # self-attention
            proj = F.linear(q, w, b)
            # reshape to 3, E and not E, 3 is deliberate for better memory coalescing and keeping same order as chunk()
            proj = proj.unflatten(-1, (3, E)).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
            return proj[0], proj[1], proj[2]
        else:
            # encoder-decoder attention
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            q_proj = F.linear(q, w_q, b_q)
            kv_proj = F.linear(k, w_kv, b_kv)
            # reshape to 2, E and not E, 2 is deliberate for better memory coalescing and keeping same order as chunk()
            kv_proj = kv_proj.unflatten(-1, (2, E)).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
            return (q_proj, kv_proj[0], kv_proj[1])
    else:
        w_q, w_k, w_v = w.chunk(3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.chunk(3)
        return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v)


def multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Optional[Tensor],
    in_proj_bias: Optional[Tensor],
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    training: bool = True,
    need_weights: bool = False,
    attn_mask: Optional[Tensor] = None,
    is_causal: bool = False,
    prefix: Tensor = None,
    batch_weight: Tensor = None
) -> Tuple[Tensor, Optional[Tensor]]:

    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape  # for visual: [197, bs, 768]; for text: [77, n_cls, 512]
    src_len, _, _ = key.shape
    if prefix is not None:
        _, _, prefix_len, _ = prefix.shape  # [bs, 2, prefix_len, embed_dim]

    assert embed_dim == embed_dim_to_check, \
        f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

    # compute in-projection
    q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
    if prefix is not None:
        prefix_k = prefix[:, 0, ...].transpose(0, 1).contiguous()  # [prefix_len, bs, embed_dim]
        prefix_v = prefix[:, 1, ...].transpose(0, 1).contiguous()  # [prefix_len, bs, embed_dim]

    # prep attention mask
    attn_mask = F._canonical_mask(
        mask=attn_mask,
        mask_name="attn_mask",
        other_type=None,
        other_name="",
        target_type=q.dtype,
        check_other=False,
    )

    if attn_mask is not None:
        # ensure attn_mask's dim is 3
        if attn_mask.dim() == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
        else:
            raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

    # reshape q, k, v for multihead attention and make em batch first
    q = q.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)  # for visual: [bs*12, 197, 64]; for text: [n_cls*8, 77, 64]
    k = k.view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    v = v.view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    if prefix is not None:
        prefix_k = prefix_k.view(prefix_k.size(0), bsz * num_heads, head_dim).transpose(0, 1)  #for visual: [bs*12, prefix_len, 64]; for text: [n_cls*8, prefix_len, 64]
        prefix_v = prefix_v.view(prefix_v.size(0), bsz * num_heads, head_dim).transpose(0, 1)

    # (deep breath) calculate attention and out projection
    if attn_mask is not None:
        if attn_mask.size(0) == 1 and attn_mask.dim() == 3:
            attn_mask = attn_mask.unsqueeze(0)  # only for text: [1, 1, 77, 77]
        else:
            attn_mask = attn_mask.view(bsz, num_heads, -1, src_len)

    q = q.view(bsz, num_heads, tgt_len, head_dim)  # for visual: [bs, 12, 197, 64]; for text: [n_cls, 8, 77, 64]
    k = k.view(bsz, num_heads, src_len, head_dim)
    v = v.view(bsz, num_heads, src_len, head_dim)
    if prefix is not None:
        prefix_k = prefix_k.view(bsz, num_heads, prefix_len, head_dim)  #for visual: [bs, 12, prefix_len, 64]; for text: [n_cls, 8, prefix_len, 64]
        prefix_v = prefix_v.view(bsz, num_heads, prefix_len, head_dim)

    attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal)  # for visual: [bs, 12, 197, 64]; for text: [n_cls, 8, 77, 64]
    attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)  # for visual: [197*bs, 768]; for text: [77*n_cls, 512]
    if prefix is not None:
        attn_output_prefix = F.scaled_dot_product_attention(q, prefix_k, prefix_v, None, dropout_p, is_causal)  # for visual: [bs, 12, 197, 64]; for text: [n_cls, 8, 77, 64]
        if batch_weight is not None:
            attn_output_prefix = attn_output_prefix * batch_weight.view(bsz, 1, 1, 1)
        attn_output_prefix = attn_output_prefix.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)  # for visual: [197*bs, 768]; for text: [77*n_cls, 512]
        attn_output += attn_output_prefix

    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)  # for visual: [197*bs, 768]; for text: [77*n_cls, 512]
    attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))  # for visual: [197, bs, 768]; for text: [77, n_cls, 512]
    return attn_output, None


class MultiheadAttention_DIKI(Module):
    
    def __init__(self, embed_dim, num_heads, device=None, dtype=None, prefix_pool_size=0, prefix_len=0) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        
        self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
        self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=True, **factory_kwargs)

        xavier_uniform_(self.in_proj_weight)
        constant_(self.in_proj_bias, 0.)
        constant_(self.out_proj.bias, 0.)
        # print("MHA pool size: ", prefix_pool_size)

        self.add_preifx = False
        if prefix_pool_size > 0:
            self.add_preifx = True
            prefix_shape = (prefix_pool_size, 2, prefix_len, embed_dim)
            prefix_pool = torch.zeros(prefix_shape, dtype=torch.float16)
            torch.nn.init.uniform_(prefix_pool[:, 0], -1, 1)
            self.prefix_pool = Parameter(prefix_pool)
            # print("Added prefix pool")
        
    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            need_weights: bool = False,
            attn_mask: Optional[Tensor] = None,
            prompt_ids: Tensor = None,
            batch_weight: Tensor = None
            ) -> Tuple[Tensor, Optional[Tensor]]:

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )

        prefix = None
        if self.add_preifx:
            assert prompt_ids.size(1) == 1, "Only single prefix for one sample is supported."
            prompt_ids = prompt_ids.squeeze(1)
            if prompt_ids.size(0) == 1:
                prompt_ids = prompt_ids.repeat(query.size(1))
            prefix = self.prefix_pool[prompt_ids]  # [bs, 2, prefix_len, embed_dim]

        attn_output, attn_output_weights = multi_head_attention_forward(
            query, key, value, self.embed_dim, self.num_heads,
            self.in_proj_weight, self.in_proj_bias,
            None, None, False,
            0., self.out_proj.weight, self.out_proj.bias,
            training=self.training,
            need_weights=need_weights,
            attn_mask=attn_mask, 
            prefix=prefix, 
            batch_weight=batch_weight)
        return attn_output, attn_output_weights