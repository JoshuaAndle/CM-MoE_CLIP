import datetime
import gc
import time

from sklearn.metrics import confusion_matrix
import torch
import logging

from tqdm import tqdm

from methods._trainer import _Trainer

logger = logging.getLogger()

### Zero shot CLIP for Online Evaluation
class ContinualCLIP(_Trainer):

    def __init__(self, args, **kwargs):
        super(ContinualCLIP, self).__init__(args, **kwargs)
        self.visible_classes = self.args.visible_classes




    def online_step(self, images, labels, idx):
        self.add_new_class(labels)
        # self.model.update_class_names(self.exposed_classes_names)

        # # zero-shot, don't need to train
        # for j in range(len(labels)):
        #     labels[j] = self.exposed_classes.index(labels[j].item())

        # images = images.to(self.device)
        # labels = labels.to(self.device)
        # images = self.train_transform(images)

        # logit = self.model(images)
        # preds = torch.argmax(logit, dim=-1)
        # _, preds = logit.topk(self.topk, 1, True, True)
        # _correct = torch.sum(preds == labels.unsqueeze(1)).item()
        # _num_data = labels.size(0)

        # del (images, labels)
        # gc.collect()
        # return 0, _correct / _num_data
        del (images, labels)
        gc.collect()
        return -1, -1



    def report_training(self, sample_num, train_loss, train_acc, step=""):
        super().report_training(sample_num, train_loss, train_acc)
        # pass

    def report_test(self, sample_num, avg_loss, avg_acc, step=""):
        super().report_test(sample_num, train_loss, train_acc, step)
        # pass

        
    def online_train(self, data):
        pass

    def online_before_task(self, task_id):
        pass

    def online_after_task(self, task_id):
        pass


    def offline_evaluate(self, test_task):
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        label, pred_list = [], []

        self.model.eval()
        with torch.no_grad():
            for data in self.test_dataloader:
                x, y = data

                ### Resets the batch exposed classes
                self.add_new_class(y, mode="test")

                if self.visible_classes == 'batch':
                    test_class_list = self.batch_exposed_classes
                    test_class_name_list = self.batch_exposed_classes_names
                    self.model.reset_class_names(test_class_name_list)
                else:
                    test_class_list = self.exposed_classes
                    test_class_name_list = self.exposed_classes_names
                    self.model.reset_class_names(test_class_name_list)


                ### Makes a contiguous set of labels to match the order of class names input into the text encoder
                for j in range(len(y)):
                    label = y[j].item()
                    label_name = self.test_dataset.classes_names[label]
                    y[j] = test_class_name_list.index(label_name)


                x = x.to(self.device)
                y = y.to(self.device)

                logit, _, _ = self.model(x)
                pred = torch.argmax(logit, dim=-1)
                _, preds = logit.topk(self.topk, 1, True, True)
                total_correct += torch.sum(preds == y.unsqueeze(1)).item()
                total_num_data += y.size(0)

                # xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(y, pred)
                # correct_l += correct_xlabel_cnt.detach().cpu()
                # num_data_l += xlabel_cnt.detach().cpu()

        ave_acc = total_correct / total_num_data

        return ave_acc














    def add_new_batch_class(self, labels, mode="train"):
        batch_exposed_classes = []

        for label in labels:
            if label.item() not in self.batch_exposed_classes:
                self.batch_exposed_classes.append(label.item())

        self.batch_exposed_classes.sort()        
        self.batch_exposed_classes_names = []



        dataset = self.train_dataset if mode == "train" else self.test_dataset

        for i in self.batch_exposed_classes:
            if dataset.classes_names[i] not in self.batch_exposed_classes_names:
                self.batch_exposed_classes_names.append(dataset.classes_names[i])
        
        # ### Changed to allow multiple labels to map to the same class name for datasets with reoccuring classes under distribution shift
        # if mode == "train":
        #     for i in self.batch_exposed_classes:
        #         if self.train_dataset.classes_names[i] not in self.batch_exposed_classes_names:
        #             self.batch_exposed_classes_names.append(self.train_dataset.classes_names[i])
        # else:
        #     for i in self.batch_exposed_classes:
        #         if self.test_dataset.classes_names[i] not in self.batch_exposed_classes_names:
        #             self.batch_exposed_classes_names.append(self.test_dataset.classes_names[i])


        # #!# Needs to check if duplicate labels map to the same class
        # self.batch_exposed_classes_names = [self.train_dataset.classes_names[i]
        #                                         for i in self.batch_exposed_classes]



    def add_new_class(self, labels, mode="train"):
        _old_num = len(self.exposed_classes)
        super().add_new_class(labels)

        self.batch_exposed_classes = []
        self.batch_exposed_classes_names = []
        self.add_new_batch_class(labels, mode)
