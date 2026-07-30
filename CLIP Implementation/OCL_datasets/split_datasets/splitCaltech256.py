# Wraping for the OxfordIIITPet dataset

import os
import json
from typing import Callable, Optional
from collections import defaultdict

from torch.utils.data import Dataset
from torchvision.datasets import Caltech256 as tv_Caltech256
from torchvision.transforms import transforms

### Note: Torchvision Caltech256 sorts indices, so seeding shouldn't matter here for splits
class SplitCaltech256(tv_Caltech256):

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

        super().__init__(
            root,
            transforms.ToTensor() if transform is None else transform,
            target_transform, download=False
            )

        ### Caltech256 has no test split, so we're holding out a manual split for testing. The test split is the final split.
        if train == False:
            split_num = splits
        splits += 1

        # print("Self y and self index: ", self.y, "\n\n\n", self.index)
        
        label_to_samples = defaultdict(list)
        for label, sample_num in zip(self.y, self.index):
            label_to_samples[label].append(sample_num)
        # print(label_to_samples)

        counts = {}
        for label,indices in label_to_samples.items():
            counts[label] = len(indices)

        splits_dict = {}
        for s in range(splits):
            split_files = {}
            for label,indices in sorted(label_to_samples.items()):
                split_len = counts[label] // splits
                start, end = split_len*s, split_len*(s+1)
                split_files[label] = indices[start:end]
            splits_dict[s] = split_files       

        # for s in range(splits):
        #     for i,(k,v) in enumerate(sorted(splits_dict[s].items())):
        #         if i > 4:
        #             break
                # print(k, " ", len(v), " of " , counts[k], " ", v[0], " ", type(v[0])) 

        splits_indices, splits_y = defaultdict(list), defaultdict(list) 
        for s in range(splits):
            split_indices, split_y = [], []
            ### Gets the subset of included indices for each label for the given dataset split
            for k,v in splits_dict[s].items():
                num_samples = len(v)
                split_y.extend(num_samples * [k])
                split_indices.extend(v)
            splits_indices[s] = split_indices 
            splits_y[s] = split_y 
        # print(splits_y[0])
        # print(splits_indices[0])
        self.index = splits_indices[split_num]
        self.y = splits_y[split_num]

        #!# Might need to modify the get_item function and some others to directly replace .y and .index, but this should work for what we're doing
        self.data = self.index
        self.targets = self.y
        self.classes_names = [self.categories[idx].split('.')[1] for idx in range(len(self.categories))]
