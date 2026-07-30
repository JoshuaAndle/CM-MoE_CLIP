# Wraping for the OxfordIIITPet dataset

import os
import json
from typing import Callable, Optional
from collections import defaultdict

from torch.utils.data import Dataset
from torchvision.datasets import Places365 as tv_Places365
from torchvision.transforms import transforms



class SplitPlaces365(tv_Places365):

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        splits: int = 5,
        split_num: int = 0
    ) -> None:

        if train:
            split = "train-standard"
        else:
            split = "val"

        super().__init__(
            root, split, small=True, download=False,
            transform=transforms.ToTensor() if transform is None else transform,
            target_transform=target_transform)
        

        # if train:
        filename_dict, labels = defaultdict(list), []

        files, labels = zip(*self.imgs)
        files, labels = list(files), list(labels)

        #!# Currently limiting the number of samples per class to 1000 to reduce computational costs for experiments (before splitting)
        for idx in range(len(labels)):
            if len(filename_dict[labels[idx]]) < 1000:
                filename_dict[labels[idx]].append(files[idx])

        counts = {}
        for key,val in filename_dict.items():
            counts[key] = len(val)

        splits_dict = {}
        for s in range(splits):
            split_files =  []
            split_labels = []

            for key, val in filename_dict.items():
                split_len = counts[key] // splits
                start, end = split_len*s, split_len*(s+1)
                split_files.extend(val[start:end])
                split_labels.extend((end-start)* [key])
            splits_dict[s] = {'files': list(zip(split_files, split_labels)), "labels": split_labels}

        self.imgs = splits_dict[split_num]['files']
        self.targets = splits_dict[split_num]['labels']


        self.classes_names = [self.classes[idx].split('/')[-1] for idx in range(len(self.classes))]


