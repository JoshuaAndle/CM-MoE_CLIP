# Wraping for the OxfordIIITPet dataset

import os
import json
from typing import Callable, Optional
from collections import defaultdict

from torch.utils.data import Dataset
from torchvision.datasets import Food101 as tv_Food101
from torchvision.transforms import transforms


class SplitFood101(tv_Food101):

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
            split = "train"
        else:
            split = "test"

        super().__init__(
            root, split, 
            transforms.ToTensor() if transform is None else transform,
            target_transform, download=False)

        ### Train set is split across all classes for distribution shift tasks
        # if train:
        filename_dict, labels = defaultdict(list), []

        for idx in range(len(self._labels)):
            # if self._labels[idx] not in filename_dict.keys():
            #     filename_dict[self._labels[idx]] = []
            filename_dict[self._labels[idx]].append(self._image_files[idx])

        counts = {}
        for key,val in filename_dict.items():
            counts[key] = len(val)

        splits_dict = {}
        for s in range(splits):
            split_files = []
            split_labels = []
            for key, val in sorted(filename_dict.items()):
                split_len = counts[key] // splits
                start, end = split_len*s, split_len*(s+1)
                # split_files[key] = val[start:end]
                split_files.extend(val[start:end])
                split_labels.extend((end-start)*[key])
            splits_dict[s] = {"files": split_files, "labels": split_labels}       



        self._image_files = splits_dict[split_num]["files"]
        self._labels = splits_dict[split_num]["labels"]
        ### Make aliases for referencing by MultiDataset class
        self.targets = self._labels
        self.classes_names = self.classes