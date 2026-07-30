# Wraping for the SHVN dataset

from typing import Callable, Optional

import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST as MNIST_tv
from torchvision.transforms import transforms
import numpy as np
from PIL import Image


"""
Implements splits of MNIST for CL tasks. 
Child classes act as wrappers that just pass in the appropriate class subsets for indexing the full MNIST dataset
Could be simplified by just passing classes into splitMNIST directly, but current dataset loading approach doesn't allow for custom arguments being passed
"""



class split_MNIST(MNIST_tv):
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

        #!# This is hardcoded for now since our computing cluster cant download datasets for some reason
        download=False
        print("Root is: ", root)
        super().__init__(root, train, transforms.ToTensor() if transform is None else transform, target_transform, download)
        # self.dataset = MNIST

        total_classes = 10
        classes_per_split = total_classes//splits

        assert(split_num>=0 and split_num<splits)

        class_labels = list(range(total_classes))
        task_classes = class_labels[(classes_per_split*split_num):(classes_per_split*(split_num+1))]

        self.data = torch.stack([self.data[i] for i, label in enumerate(self.targets) if label in task_classes], dim=0)
        ### Stack greyscale data along an RGB channel and convert to numpy array for consistency with other datasets
        self.data = torch.stack([self.data,self.data,self.data], dim=-1).numpy()

        self.targets = [self.targets[i].item() for i, label in enumerate(self.targets) if label in task_classes]

        self.classes = [self.classes[i] for i in task_classes]
        self.classes_names = self.classes

        print("Shape of MNIST split data: ", self.data.shape)
        # labels_mask = torch.zeros(dataset.targets.shape)
        # for cls in self.classes:
        #     temp_mask = (labels == cls)
        #     labels_mask = torch.max(labels_mask, temp_mask.long())
        
        # ### Filter the full dataset to only contain the specified classes
        # self.dataset.data = self.dataset.data[labels_mask]
        # self.dataset.targets = self.dataset.targets[labels_mask]

        # self.targets = []
        # for cls in self.dataset.targets:
        #     self.targets.append(int(cls))

    #!# Modifying torchvisions' __getitem__ function since we store data as numpy arrays in datasets
    def __getitem__(self, index: int):
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target



    # def __getitem__(self, index):
    #     image, label = self.dataset.__getitem__(index)
    #     return image.expand(3,-1,-1), label
        
    # def __len__(self):
    #     return len(self.dataset)



# class splitMNIST_01(splitMNIST):
#     def __init__(
#         self,
#         root: str,
#         train: bool = True,
#         transform: Optional[Callable] = None,
#         target_transform: Optional[Callable] = None,
#         download: bool = False,
#     ) -> None:

#         super().__init__(root, 
#                         train, 
#                         transforms.ToTensor() if transform is None else transform, 
#                         target_transform, 
#                         download, 
#                         classes = [0,1]
#                     )
        
# class splitMNIST_23(splitMNIST):
#     def __init__(
#         self,
#         root: str,
#         train: bool = True,
#         transform: Optional[Callable] = None,
#         target_transform: Optional[Callable] = None,
#         download: bool = False,
#     ) -> None:

#         super().__init__(root, 
#                         train, 
#                         transforms.ToTensor() if transform is None else transform, 
#                         target_transform, 
#                         download, 
#                         classes = [0,1]
#                     )
        
# class splitMNIST_45(splitMNIST):
#     def __init__(
#         self,
#         root: str,
#         train: bool = True,
#         transform: Optional[Callable] = None,
#         target_transform: Optional[Callable] = None,
#         download: bool = False,
#     ) -> None:

#         super().__init__(root, 
#                         train, 
#                         transforms.ToTensor() if transform is None else transform, 
#                         target_transform, 
#                         download, 
#                         classes = [0,1]
#                     )
        
