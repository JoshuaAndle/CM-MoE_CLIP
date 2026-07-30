from typing import Union, List

import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# from tqdm import tqdm
import logging

import torchvision.models as models
import torch.nn as nn
# import clip.clip as clip
from torch.utils.data import DataLoader


from utils.train_utils import torch_save, torch_load, exp_lr_scheduler

from .clip import clip_loader
from .clip.tokenizer import SimpleTokenizer as _Tokenizer
from .DIKI_layers import CustomCLIP



_tokenizer = _Tokenizer()




class DIKI(nn.Module):

    def __init__(self, args, model_name, device='cpu', log_dir=None):
        super(DIKI, self).__init__()
        self.device = device
        self.log_dir = log_dir
        self.model_name = model_name
        self.args = args

        ### MoE-Adapters++ uses its own args setup for preparing the clip model, so we just pass in the args
        self.design_details = {
            'args': args,
            "peft_encoder": "both",
            'peft_method': "diki",
            "vision_depth": args.prompt_depth_vision,
            "language_depth": args.prompt_depth_text, 
            "vision_ctx": args.n_ctx_vision,
            "language_ctx": args.n_ctx_text,
            "pool_size": args.n_tasks
        }
        self.design_details_ori = {
            'args': args,
            "peft_encoder": "both",
            'peft_method': "diki",
            "vision_depth": 0,
            "language_depth": 0, 
            "vision_ctx": 0,
            "language_ctx": 0
        }

        #!# We need to defer clip setup until the start of the first task
        self.clip_model = clip_loader.load(self.model_name, device=self.device, jit=False, design_details=self.design_details, log_dir=self.log_dir)
        self.clip_model_ori = clip_loader.load(self.model_name, device=self.device, jit=False, design_details=self.design_details_ori, log_dir=self.log_dir)
        self.clip_model_diki = None


        self.texts = []
        # self.text_tokens = torch.tensor([])
        self.current_class_names = []
        # self.current_class_tokens = None


        ### Using the original prompt since the image size should be sufficiently large for decent image clarity
        # self.prompt_template = "a bad photo of a {}."
        self.prompt_template = "a photo of a {}."
        self.text_tokens = self.labels_tokenize(["placeholder"])



    ### These should be redundant since the parent class is nn.module, but just to be safe we pass them explicitly to the trainable CLIP model
    def eval(self):
        self.clip_model.eval()

    def train(self):
        self.clip_model.train()





    # def load_model(self, cfg, task_id, load_file=None):
    def load_model(self, load_file):
        state_dict = torch.load(load_file, map_location="cpu")

        print(f"Loading weights from {load_file}")
        # set strict=False
        self.clip_model_diki.load_state_dict(state_dict, strict=False)

        return [i for i in state_dict.keys()]




    ### Prepare custom model for DIKI, load previous weights if available, and freeze backbone
    def setup_diki_clip_model(self):
        self.clip_model_diki = CustomCLIP(self.args, self.texts, self.text_tokens, self.clip_model, self.clip_model_ori)

        if self.args.load:
            self.load_model(self.args.load)
        
        print("Turning off gradients in both the image and the text encoder")
        names_to_update = ["prompt_key", "prefix_pool"] # not sure where prompt_key is? Prefix pool is in the custom MHA class

        for name, param in self.clip_model_diki.named_parameters():
            update_flag = False
            if "prefix" in name.lower():
                print(name, param.shape)
            for name_to_update in names_to_update:
                if name_to_update in name:
                    update_flag = True
            if not update_flag:
                param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.clip_model_diki.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        para_log = f"Parameters to be updated: {enabled}"
        print(para_log)




    ### Set up model for current task with updated args
    def setup_task_clip_model(self):
        model = clip_loader.load(self.model_name, device=self.device, jit=False, design_details=self.design_details, log_dir=self.log_dir)
        model_ori = clip_loader.load(self.model_name, device=self.device, jit=False, design_details=self.design_details_ori, log_dir=self.log_dir)

        if self.device is not None:
            model = model.to(self.device)
            model_ori = model_ori.to(self.device)
        self.clip_model = model
        self.clip_model_ori = model_ori



    ### We implement the tokenization function from CLIP directly to store the eot tokens as a text-modal equivalent to cls tokens
    def labels_tokenize(self, labels: Union[str, List[str]], context_length: int = 77) -> torch.LongTensor:
        """
        Returns the tokenized representation of given input string(s)
        Parameters
        ----------
        labels : Union[str, List[str]]
            An input string or a list of labels to tokenize
        context_length : int
            The context length to use; all CLIP models use 77 as the context length
        Returns
        -------
        A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
        """
        if isinstance(labels, str):
            labels = [labels]


        texts = [self.prompt_template.format(c) for c in labels]
        self.texts = texts
        # print("\nTokenization input: ", texts)

        sot_token = _tokenizer.encoder["<start_of_text>"]
        eot_token = _tokenizer.encoder["<end_of_text>"]
        all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token]
                      for text in texts]

        ### Record locations of eot tokens for debugging purposes
        self.eot_indices = [len(all_tokens[idx]) - 1 for idx in range(len(all_tokens))]

        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:  # Truncate
                ### Changing from the initial implementation to ensure eot_token is not removed
                # tokens = tokens[:context_length]
                tokens = tokens[:context_length-1] + [eot_token]
            result[i, :len(tokens)] = torch.tensor(tokens)

        # self.current_class_tokens = result

        return result.to(self.device)

    # def update_class_names(self, new_class_names):
    #     _num = 0
    #     for c in new_class_names:
    #         if c not in self.current_class_names:
    #             self.current_class_names.append(c)
    #             _num += 1
    #     if _num > 0:
    #         self.text_tokens = self.labels_tokenize(self.current_class_names)
    #     return self.text_tokens


    def reset_class_names(self, new_class_names):
        if new_class_names != self.current_class_names:
            self.current_class_names = new_class_names
            self.text_tokens = self.labels_tokenize(self.current_class_names)
            return True # Indicating whether tokenized labels were modified
        return False

    def forward(self, image=None, task_ids=None, apply_softmax=False):
        return self.clip_model_diki(image, task_ids)



        # return self.clip_model(image, text_tokens, val_text_task_id=val_text_task_id)



        # logits_per_image, _, image_features, text_features = self.clip_model(image, text_tokens)
        # if apply_softmax:
        #     probs = logits_per_image.softmax(dim=-1)
        # else:
        #     probs = logits_per_image
        # return probs, image_features, text_features
        # # raise ValueError
