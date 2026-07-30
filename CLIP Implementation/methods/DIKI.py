import datetime
import gc
import time
import logging

from methods._trainer import _Trainer

import copy
import os
import numpy as np
# import clip.clip as clip
import torch
from torch import nn
import torch.nn.functional as F






train_iteration = None
logger = logging.getLogger()



class DIKI(_Trainer):

    def __init__(self, args, **kwargs):
        super(DIKI, self).__init__(args, **kwargs)
        self.visible_classes = self.args.visible_classes
        self.task_id = -1
        self.known_tasks = []

        # self.useful_params = None
        # self.loss_interval  = -1
        self.num_batches = -1
        self.total_iterations = -1
        self.batch_counter = 0







    ### Update the args to reflect the original .sh training loop format throughout tasks and eval
    def update_method_args(self, is_train=True):

        self.args.is_train = is_train
        self.args.task_id = self.task_id
        self.args.save = os.path.join(self.log_dir, str(self.task_id))
        self.args.task_num = self.args.n_tasks

        if self.task_id not in self.known_tasks:
            self.known_tasks.append(self.task_id)

        if self.task_id > 0:
            if is_train:
                self.args.load = os.path.join(self.log_dir, str(self.task_id-1), "state_dict.pt")

        if is_train == True:
            self.args.eval_only = False
        else:
            self.args.eval_only = True

        return





    def online_before_task(self, task_id):
        self.task_id = task_id
        
        ### Update arguments for current task and propagate changes to model
        self.update_method_args()
        self.model.design_details["args"] = self.args
        self.model.args = self.args

        ### Update the state of the model (make a new copy for current task and load state)
        self.model.setup_task_clip_model()
        self.model.setup_diki_clip_model()



        # self.loss_interval = self.args.loss_interval
        self.num_batches = len(self.train_dataloader)
        self.batch_counter = 0
        self.total_iterations = self.args.online_iter * self.num_batches
        print("New num batches: ", self.num_batches)

        ### Remake the optimizer for the new training task and task clip model
        self.reset_opt()
        self.model = self.model.cuda()
        self.logit_scale = self.model.clip_model.logit_scale




        #!# Need to modify this to be done in a running fashion, for now I will implement it cheating with the full loader
        ###   in order to see how much we lose accuracy by updating it to be "properly" online

        all_image_features = torch.empty([0, self.model.clip_model_diki.vis_dim], dtype=self.model.clip_model_diki.dtype, device=self.device)
        with torch.no_grad():
            for sample in self.train_dataloader:
                x, _, _ = sample
                # parse_sample(sample, is_train=False, task_id=task_id, cfg=cfg)
                image_features = self.model.clip_model_diki.image_encoder_ori(x.type(self.model.clip_model_diki.dtype).to(self.device))
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                all_image_features = torch.cat([all_image_features, image_features.detach()], dim=0)

        all_image_features = all_image_features.type(torch.float)  # to avoid precision problems
        mean = all_image_features.mean(dim=0)
        delta = (all_image_features - mean.unsqueeze(0))
        covar = delta.t() @ delta / (all_image_features.size(0) - 1)
        covar +=  torch.eye(covar.size(0), device=covar.device, dtype=torch.float)*1e-7  # to avoid precision problems
        self.model.clip_model_diki.means[task_id] = mean
        self.model.clip_model_diki.covars[task_id] = covar
        self.model.clip_model_diki.task_learnt += 1
            











    def online_step(self, images, labels, idx):

        self.add_new_class(labels)

        _loss, _acc, _iter = 0.0, 0.0, 0
        for _ in range(int(self.online_iter)):
            loss, acc = self.online_train([images, copy.deepcopy(labels)])

            #!# I am unsure currently if it would be best to change this to only allow accuracy updates on first iteration, to strictly track evaluation of unseen streamed samples
            _loss += loss
            _acc += acc
            _iter += 1

        self.batch_counter += 1


        return _loss / _iter, _acc / _iter





    def online_train(self, data):
        self.model.train()
        total_loss, total_correct, total_num_data = 0.0, 0.0, 0.0


        ### Universal setup for managing exposed classes and ensuring image and text logits match properly
        if self.visible_classes == 'batch':
            train_class_list = self.batch_exposed_classes
            train_class_name_list = self.batch_exposed_classes_names
            self.model.reset_class_names(train_class_name_list)
        else:
            train_class_list = self.exposed_classes
            train_class_name_list = self.exposed_classes_names
            names_added = self.model.reset_class_names(train_class_name_list)
            if names_added == True:
                self.model.clip_model_diki.update_classnames(self.model.texts, self.model.text_tokens)


        x, y = data

        ### Makes a contiguous set of labels to match the order of class names input into the text encoder
        for j in range(len(y)):
            label = y[j].item()
            label_name = self.train_dataset.classes_names[label]
            y[j] = train_class_name_list.index(label_name)

        x,y = x.to(self.device), y.to(self.device)
        text_tokens = self.model.text_tokens


        task_ids = torch.IntTensor([self.task_id]).repeat(x.size(0))

        ### Train clip model adapters
        with torch.amp.autocast('cuda', enabled=self.use_amp):  
            res = self.model(x, task_ids)
            outputs = res["outputs"] # outputs are the pre-softmax logits_per_image





        loss_main = F.cross_entropy(outputs, y)
        loss = loss_main
        # print(loss)
        # print(loss.requires_grad)
        # print(loss.grad_fn)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.scheduler.step(self.batch_counter)



        ### Prediction accuracy calculation
        _, preds = outputs.topk(self.topk, 1, True, True)

        total_correct += torch.sum(preds == y.unsqueeze(1)).item()
        total_num_data += y.size(0)
        total_loss += loss.item()

        return total_loss, total_correct / total_num_data







    def online_after_task(self, task_id):

        self.save_model()


        ### Reload the model under evaluation settings prior to eval phase
        self.task_id = task_id
        
        ### Update arguments for current task and propagate changes to model
        self.update_method_args(is_train=False)
        self.model.design_details["args"] = self.args
        self.model.args = self.args









    def offline_evaluate(self, test_task_id):
        """
        Returns the average test accuracy for a given task.
        If args.unknown_test_task_id is set to True, will predict the task ID and use the appropriate subnetwork        
        """
        print("Running offline eval for test task: ", test_task_id)

        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        label, pred_list = [], []
        self.args.eval_acc_task_id = test_task_id

        self.model.eval()
    
        with torch.no_grad():
            for i, data in enumerate(self.test_dataloader):
                x, y = data

                if self.debug and (i + 1) * self.temp_batchsize >= 1000:
                    # print("Temp test batch size ", self.temp_batchsize, " images shape: ", images.shape)
                    break
                
                self.add_new_class(y, mode="test")

                if self.visible_classes == 'batch':
                    test_class_list = self.batch_exposed_classes
                    test_class_name_list = self.batch_exposed_classes_names
                    self.model.reset_class_names(test_class_name_list)
                else:
                    test_class_list = self.exposed_classes
                    test_class_name_list = self.exposed_classes_names
                    names_added = self.model.reset_class_names(test_class_name_list)
                    if names_added == True:
                        self.model.clip_model_diki.update_classnames(self.model.texts, self.model.text_tokens)


                for j in range(len(y)):
                    label = y[j].item()
                    label_name = self.test_dataset.classes_names[label]
                    y[j] = test_class_name_list.index(label_name)

                x,y = x.to(self.device), y.to(self.device)

                text_tokens = self.model.text_tokens
                
                # task_ids = torch.IntTensor([self.task_id]).repeat(x.size(0))


                res = self.model(x)
                outputs = res["outputs"] # outputs are the pre-softmax logits_per_image



                _, preds = outputs.topk(self.topk, 1, True, True)
                total_correct += torch.sum(preds == y.unsqueeze(1)).item()
                total_num_data += y.size(0)

        total_acc = total_correct / total_num_data

        return total_acc












    def add_new_batch_class(self, labels, mode="train"):
        batch_exposed_classes = []

        for label in labels:
            if label.item() not in self.batch_exposed_classes:
                self.batch_exposed_classes.append(label.item())

        self.batch_exposed_classes.sort()        
        self.batch_exposed_classes_names = []




        dataset = self.train_dataset if mode == "train" else self.test_dataset

        # print("Dataset classes names: ", dataset.classes_names, flush=True)
        for i in self.batch_exposed_classes:
            if dataset.classes_names[i] not in self.batch_exposed_classes_names:
                self.batch_exposed_classes_names.append(dataset.classes_names[i])
        
        
    def add_new_class(self, labels, mode="train"):
        # print("Adding new classes")
        _old_num = len(self.exposed_classes)
        super().add_new_class(labels)

        self.batch_exposed_classes = []
        self.batch_exposed_classes_names = []
        self.add_new_batch_class(labels, mode)


    def report_training(self, sample_num, train_loss, train_acc, step=""):
        super().report_training(sample_num, train_loss, train_acc)
        # pass

    def report_test(self, sample_num, avg_loss, avg_acc, step=""):
        super().report_test(sample_num, train_loss, train_acc, step)
        # pass




    def reset_opt(self):

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.scheduler = build_cosine_scheduler(self.optimizer, lr=self.args.lr, total_step=self.total_iterations)

    def save_model(self):
        save_dict = {}
        for name, para in self.model.clip_model_diki.named_parameters():
            if para.requires_grad:
                save_dict[name] = para
        for name, para in self.model.clip_model_diki.named_buffers():  # for gaussian parameters
            if "means" in name or "covars" in name or "task_learnt" in name:
                save_dict[name] = para
        save_dir = self.args.save
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(save_dict, os.path.join(save_dir, f'state_dict.pt'))










def cosine_schedule_warmup(total_step, value, final_value=0, warmup_step=0, warmup_value=0):
    if warmup_step > 0:
        warmup_schedule = np.linspace(warmup_value, value, warmup_step+2)[1:-1]
    else:
        warmup_schedule = np.array([])
    steps = np.arange(total_step - warmup_step)
    schedule = final_value + 0.5 * (value-final_value) * (1+np.cos(np.pi * steps / len(steps)))
    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == total_step
    return schedule

class build_cosine_scheduler:
    def __init__(self, optimizer, lr, total_step, lr_warmup_step=0):
        init_lr = 0
        final_lr = lr * 1e-3
        self.lrs = cosine_schedule_warmup(total_step, lr, final_lr, lr_warmup_step, init_lr)
        self.optimizer = optimizer

    def step(self,idx):
        lr = self.lrs[idx]
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"]= lr
        self.lr=lr












