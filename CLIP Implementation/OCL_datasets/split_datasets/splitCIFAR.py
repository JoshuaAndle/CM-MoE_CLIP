# Wraping for the CIFAR dataset

from typing import Callable, Optional

from torchvision.datasets import CIFAR10 as CIFAR10_tv
from torchvision.datasets import CIFAR100 as CIFAR100_tv
from torchvision.transforms import transforms
import numpy as np

class split_CIFAR10(CIFAR10_tv):

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
        """
        splits: How many equal-sized splits to divide the dataset into. Splits contain consecutive classes.
        split_num: Which split to return a dataset for.
        """

        super().__init__(
            root, train,
            transforms.ToTensor() if transform is None else transform,
            target_transform, download)

        ### Split the dataset into the given number of splits, then take one given task split based on split_num
        total_classes = 10
        classes_per_split = total_classes//splits

        assert(split_num>=0 and split_num<splits)

        class_labels = list(range(total_classes))
        task_classes = class_labels[(classes_per_split*split_num):(classes_per_split*(split_num+1))]

        self.data = np.asarray([self.data[i] for i, label in enumerate(self.targets) if label in task_classes])
        self.targets = [self.targets[i] for i, label in enumerate(self.targets) if label in task_classes]

        self.classes = [self.classes[i] for i in task_classes]
        self.classes_names = self.classes


class split_CIFAR100(CIFAR100_tv):

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
            root, train,
            transforms.ToTensor() if transform is None else transform,
            target_transform, download)

        total_classes = 10
        classes_per_split = total_classes//splits

        assert(split_num>=0 and split_num<splits)

        class_labels = list(range(total_classes))
        task_classes = class_labels[(classes_per_split*split_num):(classes_per_split*(split_num+1))]

        self.data = np.asarray([self.data[i] for i, label in enumerate(self.targets) if label in task_classes])
        self.targets = [self.targets[i] for i, label in enumerate(self.targets) if label in task_classes]

        self.classes = [self.classes[i] for i in task_classes]
        self.classes_names = self.classes