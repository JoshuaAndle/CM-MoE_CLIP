from typing import Union, List

import os
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






class CMMoE(nn.Module):

    def __init__(self, args, model_name, device='cpu', log_dir=None):
        super(CMMoE, self).__init__()
        self.device = device
        self.log_dir = log_dir

        #!# peft_method is lora for lora_clip and adapter for adapter_clip
        design_details = {
            'args': args,
            'peft_method': 'moe',
            'peft_encoder': args.peft_encoder,
            'adapter_blocks_text': args.adapter_blocks_text,
            'adapter_blocks_image': args.adapter_blocks_image,
            'ffn_num': 64,
            'lora_alpha': 1,
            'lora_r': 4,
            'top_k': args.router_topk,
            'experts_per_task': args.experts_per_subnet
        }

        self.clip_model = clip_loader.load(model_name, device=self.device, jit=False, design_details=design_details, log_dir=self.log_dir)

        self.text_tokens = None
        self.current_class_names = []
        # self.current_class_tokens = None

        self.eot_indices = []

        ### Using the original prompt since the image size should be sufficiently large for decent image clarity
        # self.prompt_template = "a bad photo of a {}."
        self.prompt_template = "a photo of a {}."

    ### These should be redundant since the parent class is nn.module, but just to be safe we pass them explicitly to the trainable CLIP model
    def eval(self):
        self.clip_model.eval()

    def train(self):
        self.clip_model.train()



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

    def update_class_names(self, new_class_names):
        # _num = 0
        # for c in new_class_names:
        #     if c not in self.current_class_names:
        #         self.current_class_names.append(c)
        #         _num += 1
        # if _num > 0:
        #     self.text_tokens = self.labels_tokenize(self.current_class_names)
        # return self.text_tokens
        self.reset_class_names(new_class_names)

    def reset_class_names(self, new_class_names):
        if new_class_names != self.current_class_names:
            self.current_class_names = new_class_names
            self.text_tokens = self.labels_tokenize(self.current_class_names)
            return True # Indicating whether tokenized labels were modified
        return False

    def forward(self, image, text_tokens=None, is_train=True, val_subnet_id=None, apply_softmax=True, eot_indices=None):
        # if text_tokens is None:
        #     text_tokens = self.text_tokens
        if image is None:
            return self.clip_model.encode_text(text_tokens, is_train, val_subnet_id)
        if text_tokens is None:
            return self.clip_model.encode_image(image, is_train, val_subnet_id)


        logits_per_image, _, image_features, text_features = self.clip_model(image, text_tokens, is_train, val_subnet_id)
        if apply_softmax:
            probs = logits_per_image.softmax(dim=-1)
        else:
            probs = logits_per_image
        return probs, image_features, text_features
