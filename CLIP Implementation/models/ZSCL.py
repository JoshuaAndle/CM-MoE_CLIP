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




_tokenizer = _Tokenizer()






class ZSCL(nn.Module):

    def __init__(self, args, model_name, device='cpu', log_dir=None):
        super(ZSCL, self).__init__()
        self.device = device
        self.log_dir = log_dir
        self.model_name = model_name
        self.args = args

        self.design_details = {
            'args': args
        }

        #!# We need to defer clip setup until the start of the first task
        self.clip_model = clip_loader.load(self.model_name, device=self.device, jit=False, design_details=self.design_details, log_dir=self.log_dir)
        # self.clip_model = None
        self.clip_model_ref = clip_loader.load(self.model_name, device=self.device, jit=False, design_details=self.design_details, log_dir=self.log_dir)
        self.clip_model_l2 = None
        self.clip_model_we = None

        self.text_tokens = None
        self.current_class_names = []
        # self.current_class_tokens = None


        ### Using the original prompt since the image size should be sufficiently large for decent image clarity
        # self.prompt_template = "a bad photo of a {}."
        self.prompt_template = "a photo of a {}."

    ### These should be redundant since the parent class is nn.module, but just to be safe we pass them explicitly to the trainable CLIP model
    def eval(self):
        self.clip_model.eval()

    def train(self):
        self.clip_model.train()

    ### Set a copy of the current model state for regularization and distillation
    def set_l2_model(self):
        self.clip_model_l2 = copy.deepcopy(self.clip_model)
        self.clip_model_l2.cpu()
        # self.clip_model_l2.cuda()

    ### Set a copy of the current model state for regularization and distillation
    def set_we_model(self):
        self.clip_model_we = copy.deepcopy(self.clip_model)
        self.clip_model_we.cuda()


    # def freeze_clip(self, useful_params):
    #     for k, v in self.clip_model.named_parameters():  # frozen params
    #         # if not any(exclude_str in k for exclude_str in exclude_list):
    #         #     v.requires_grad = False
    #         if any(k.startswith(s) for s in useful_params):
    #             v.requires_grad = True
    #         else:
    #             v.requires_grad = False







    ### Set up model for current task with updated args
    def setup_task_clip_model(self):

        model = clip_loader.load(self.model_name, device=self.device, jit=False, design_details=self.design_details, log_dir=self.log_dir)

        #  train_preprocess is_train=True val_preprocess is_train=False
        # if self.args.load is not None and self.args.repeat_train is True:  # continual learning
        if self.args.load is not None:  # continual learning
            print("Loading state dict from ", self.args.load)
            checkpoint = torch.load(self.args.load)
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint["state_dict"], strict=False)

            if len(missing_keys) > 0 or len(unexpected_keys) > 0:
                print("Missing keys:", missing_keys)
                print("Unexpected keys:", unexpected_keys)
                # print("check missing keys")
            # print("Checkpoint loaded from:", save_path)
            # with open(save_path, 'rb') as f:
            #     classifier = pickle.load(f)

        if self.device is not None:
            model = model.to(self.device)
        self.clip_model = model



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

    def forward(self, image=None, text_tokens=None, apply_softmax=True, val_text_task_id=None, use_we=False):
        # if text_tokens is None:
        #     text_tokens = self.text_tokens

        if use_we == True:
            model = self.clip_model_we
        else:
            model = self.clip_model

        if image is None or text_tokens is None:
            return model(image, text_tokens, val_text_task_id=val_text_task_id)
        else:
            logits_per_image, _, image_features, text_features = model(image, text_tokens, val_text_task_id=val_text_task_id)
            if apply_softmax:
                probs = logits_per_image.softmax(dim=-1)
            else:
                probs = logits_per_image
            return probs, image_features, text_features
            # raise ValueError
