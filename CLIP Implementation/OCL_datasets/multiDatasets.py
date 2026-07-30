# Multi datasets for continual learning
# All datasets needs to be in the same format.
# have targets and classes within the dataset.

from typing import Callable, Optional, Iterable
from torch.utils.data import Dataset


class multiDatasets(Dataset):

    def __init__(
        self,
        datasets: Iterable[Dataset],
        root: Optional[str] = None,
        train: Optional[bool] = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: Optional[bool] = False,
        preprocessed: bool = False,
    ) -> None:

        super().__init__()
        self.datasets = []
        self.dataset_lengths = []
        self.classes = []

        class_count = 0
        self.class_count_by_dataset = []
        if preprocessed:
            # print("Setting up multiDataset with preprocessed datasets.")
            for dataset in datasets:
                if not isinstance(dataset, Dataset):
                    raise TypeError("dataset should be a Dataset object")
                # print("Dataset number of classes: ", len(dataset.classes_names))
                # print("Dataset type of targets: ", type(dataset.targets))
                self.datasets.append(dataset)
                self.dataset_lengths.append(len(self.datasets[-1]))
                class_count += len(self.datasets[-1].classes_names)
                self.class_count_by_dataset.append(len(self.datasets[-1].classes_names))

        else:
            for dataset in datasets:
                if not isinstance(dataset, Dataset):
                    raise TypeError("dataset should be a Dataset object")
                self.datasets.append(dataset(root, train, transform, target_transform, download))
                self.dataset_lengths.append(len(self.datasets[-1]))
                class_count += len(self.datasets[-1].classes_names)
                self.class_count_by_dataset.append(len(self.datasets[-1].classes_names))
        
        # self.classes = [str(i) for i in range(len(self.classes))]
        self.classes_names = []
        self.classes_names_by_task = {}
        self.task_classes = {}
        # self.task_class_names = {}
        self.targets = []
        self.sample_task_ids = []
        self.classes_remap_dict = {}

        #!# Note: For distribution-shift tasks I'm allowing remapped_targets to have reoccuring classes represented by multiple different integers across tasks
        ###     This SHOULD be fine, since they will map to the same string label for observed class names, and the distinct integers may allow for better debugging, 
        ###     but need to keep an eye on it to make sure it works as intended
        for t, dataset in enumerate(self.datasets):
            # task_classes, task_class_names = [], []
            task_classes = []
            running_class_count = sum(self.class_count_by_dataset[:t])
            # print("Running Class Count for task ", t, ": ", running_class_count)
            self.classes_names += dataset.classes_names
            self.classes_names_by_task[t] = dataset.classes_names
            # self.task_class_names[t] = dataset.classes_names

            #!# Added remapping since it would not handle cases where different datasets have overlapping labels 
            ###    (this was designed for single-dataset splits where labels are unique)
            self.classes_remap_dict[t] = {}

            dataset_targets = sorted(set(dataset.targets))
            # print(dataset_targets)
            for i, class_label in enumerate(dataset_targets):
                self.classes_remap_dict[t][class_label] = i + running_class_count

            self.task_classes[t] = list(self.classes_remap_dict[t].values())

            remapped_targets = []
            remapped_targets = [self.classes_remap_dict[t][label_key] for label_key in dataset.targets]
            
            # for cls in dataset.targets:
                # self.targets.append(int(cls) + sum(self.classes[:i]))
                # self.sample_task_ids.append(i)
            ### Add all the targets for dataset t, offset by the amount of classes in preceding tasks' datasets
            # for i in dataset.targets:
            # for i in remapped_targets:
            self.targets.extend(remapped_targets)
            self.sample_task_ids.extend([t]*len(remapped_targets))
                
        # print("Task classes: ", self.task_classes)
        # print("Dataset class names: ", self.classes_names)
        # print("Unique multidataset target labels: ", set(self.targets))
        # print("Remap dict: ", self.classes_remap_dict)


    def __getitem__(self, index):
        target = self.targets[index]
        #!# Don't think this is a performance concern but could use a dict to map sample IDs to task IDs for faster lookup
        ### Set up this way rather than a single list/tensor as dataset image sizes may be incompatible
        for i, dataset in enumerate(self.datasets):
            if index < self.dataset_lengths[i]:
                ### Allow the dataset class to apply any transforms, then return the image and offset target label
                image, _ = dataset[index]
                return image, target
                # return dataset[index], target
            index -= self.dataset_lengths[i]
        raise ValueError(f"Sample index {index} not found in multidataset!")

    def __len__(self):
        return len(self.targets)
