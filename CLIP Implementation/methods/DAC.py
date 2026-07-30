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


from .DAC_utils.lrbmutil import clip_loss,compute_orthogonal_loss,load_lora_weights
from .DAC_utils.lora_utils import mark_only_lora_as_trainable, get_lora_parameters, apply_lora_all, save_lora_all, load_lora_all, merge_lora_back_to_original_all


# load_lora_weights
# apply_lora_all
# load_lora_all
# merge_lora_back_to_original_all
# mark_only_lora_as_trainable







train_iteration = None
logger = logging.getLogger()




class DAC(_Trainer):

    def __init__(self, args, **kwargs):
        super(DAC, self).__init__(args, **kwargs)
        self.visible_classes = self.args.visible_classes
        self.task_id = -1
        self.known_tasks = []

        # self.useful_params = None
        # self.loss_interval  = -1
        self.num_batches = -1
        self.total_iterations = -1
        self.batch_counter = 0

        self.list_lora_layers = None




    def get_lora_paths(self):
        paths = None

        for task in range(self.task_id):
            if paths is None:
                paths = os.path.join(self.log_dir, str(task), "lora.pt")
            else:
                paths = paths + "," + os.path.join(self.log_dir, str(task), "lora.pt")

        print("Loading from paths: ", paths)
        return paths







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
                self.args.lora_paths = self.get_lora_paths()
                self.args.load = os.path.join(self.log_dir, str(self.task_id-1), "lora.pt")

        if is_train == True:
            self.args.eval_only = False
        else:
            self.args.eval_only = True

        return




    def setup_lora_model(self):
        ### Load pretrained clip model then load lora weights
        self.model.setup_task_clip_model()

        self.loading = None
        if self.args.load:
            print("Loading LoRA")
            lora_paths_list = self.args.lora_paths.split(',') # paths is a comma-separated str of all past task save paths
            self.loading=load_lora_weights(lora_paths_list)

            index=0
            max_len=len(lora_paths_list)
            for lora_path in lora_paths_list:
                list_lora_layers = apply_lora_all(self.args, self.model.clip_model)
                load_lora_all(self.args, list_lora_layers, lora_path)
                self.model.clip_model = merge_lora_back_to_original_all(self.args, self.model.clip_model, list_lora_layers, index, max_len)
                index += 1
    

        print("[Training mode] lora")
        self.list_lora_layers = apply_lora_all(self.args, self.model.clip_model) # Adds LoRA modules to the vanilla CLIP model
            
        self.model = self.model.cuda()
        mark_only_lora_as_trainable(self.model.clip_model) # Freeze backbone
        total_params_size = sum(p.numel() * p.element_size() for p in self.model.clip_model.parameters() if p.requires_grad)

        print('The number of Total Trainable Parameters------------------:', sum(p.numel() for p in self.model.clip_model.parameters() if p.requires_grad))
        print(f"Total Trainable Parameters Memory Size: {total_params_size / 1024 / 1024:.2f} MB")



    def online_before_task(self, task_id):
        self.task_id = task_id
        
        ### Update arguments for current task and propagate changes to model
        self.update_method_args()
        self.model.design_details["args"] = self.args
        self.model.args = self.args

        ### Call setup_task_model to update the state of the model (make a new copy for current task)
        self.setup_lora_model()



        # self.loss_interval = self.args.loss_interval
        self.num_batches = len(self.train_dataloader)
        self.batch_counter = 0
        self.total_iterations = self.args.online_iter * self.num_batches
        print("New num batches: ", self.num_batches)

        ### Remake the optimizer for the new training task and task clip model
        self.reset_opt()
        self.model = self.model.cuda()
        self.logit_scale = self.model.clip_model.logit_scale

        # self.save_path = os.path.join(self.args.save, 'rank')
        os.makedirs(self.args.save, exist_ok=True)




    def online_step(self, images, labels, idx):

        self.add_new_class(labels)
        self.batch_counter += 1

        _loss, _acc, _iter = 0.0, 0.0, 0
        for _ in range(int(self.online_iter)):
            loss, acc = self.online_train([images, copy.deepcopy(labels)])

            #!# I am unsure currently if it would be best to change this to only allow accuracy updates on first iteration, to strictly track evaluation of unseen streamed samples
            _loss += loss
            _acc += acc
            _iter += 1


        return _loss / _iter, _acc / _iter





    def online_train(self, data):
        self.model.train()
        total_loss, total_correct, total_num_data, total_mem_loss, total_mem_correct, total_num_mem_data = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        print("Train step")

        ### Universal setup for managing exposed classes and ensuring image and text logits match properly
        if self.visible_classes == 'batch':
            train_class_list = self.batch_exposed_classes
            train_class_name_list = self.batch_exposed_classes_names
            self.model.reset_class_names(train_class_name_list)
        else:
            train_class_list = self.exposed_classes
            train_class_name_list = self.exposed_classes_names
            self.model.reset_class_names(train_class_name_list)

        x, y = data

        # batch_texts = []
        ### Makes a contiguous set of labels to match the order of class names input into the text encoder
        for j in range(len(y)):
            label = y[j].item()
            label_name = self.train_dataset.classes_names[label]
            y[j] = train_class_name_list.index(label_name)
            # batch_texts.append(label_name)

        # batch_text_tokens = self.model.labels_tokenize(batch_texts)

        x,y = x.to(self.device), y.to(self.device)
        text_tokens = self.model.text_tokens


        # self.scheduler(self.batch_counter)






        ### Train clip model adapters
        with torch.amp.autocast('cuda', enabled=self.use_amp):  

            # text_embeds  = self.model(None, text_tokens)
            text_embeds  = self.model(None, text_tokens)
            image_embeds  = self.model(x, None)
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        logits_per_image = self.logit_scale.exp() * image_embeds @ text_embeds.t()        
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * self.logit_scale.exp()
        # print(f"Shapes of logits {logits_per_text.shape} and y {y.shape}")
        logits_per_image = logits_per_image.softmax(dim=-1)
        ### Clip loss assumes # batch labels match the # batch names, which they do not for our setting, so we manually compute loss
        # loss = clip_loss(logits_per_text, y)
        # print("Shape of logits per text: ", logits_per_text.shape)
        # print("Shape of logits per image: ", logits_per_image.shape)
        # print("Shape of y: ", y.shape)


        # texts_loss = F.cross_entropy(logits_per_text, y)
        images_loss = F.cross_entropy(logits_per_image, y)
        # loss = (images_loss + texts_loss) / 2.0
        loss = images_loss


        ### Loss calculation for LoRA 
        o_loss = 0
        if self.loading:
            o_loss=compute_orthogonal_loss(self.model.clip_model, self.loading)
            
        loss=loss+o_loss*0.1
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        if self.batch_counter % 20 == 0:
            print("Loss:", loss.item()) 
            print("oLoss:", o_loss*0.1) 

  
        ### Prediction accuracy calculation
        _, preds = logits_per_image.topk(self.topk, 1, True, True)

        total_correct += torch.sum(preds == y.unsqueeze(1)).item()
        total_num_data += y.size(0)
        total_loss += loss.item()

        return total_loss, total_correct / total_num_data







    def online_after_task(self, task_id):

        save_lora_all(self.args, self.list_lora_layers)


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
                    self.model.reset_class_names(test_class_name_list)

                # batch_texts = []
                for j in range(len(y)):
                    label = y[j].item()
                    label_name = self.test_dataset.classes_names[label]
                    y[j] = test_class_name_list.index(label_name)
                    # batch_texts.append(label_name)

                # batch_text_tokens = self.model.labels_tokenize(batch_texts)

                x,y = x.to(self.device), y.to(self.device)

                text_tokens = self.model.text_tokens
                
                # ### Makes a list of class embeddings for each task (should be redundant though since the tasks are selected by the recognition layer)
                # zeroshot_weights = zeroshot_classifier(text_tokens, self.model.clip_model, self.args)        
                # class_embeddings = self.model(None,texts.unsqueeze(0))  # embed with text encoder
                class_embeddings = self.model(None,self.model.text_tokens)  # embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)


                # predict
                image_features = self.model(x, None)
                image_features /= image_features.norm(dim=-1, keepdim=True)

                logits = self.logit_scale.exp() * image_features @ class_embeddings.t()
                
                # # measure accuracy
                # acc1, acc5 = accuracy(logits, target, topk=(1, 5))
                # top1 += acc1
                # top5 += acc5
                # n += images.size(0)




                _, preds = logits.topk(self.topk, 1, True, True)
                total_correct += torch.sum(preds == y.unsqueeze(1)).item()
                total_num_data += y.size(0)

                # xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(y, pred)
                # correct_l += correct_xlabel_cnt.detach().cpu()
                # num_data_l += xlabel_cnt.detach().cpu()

                # label += y.tolist()
                # pred_list += pred.tolist()

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


        lora_params = get_lora_parameters(self.model.clip_model, bias='none')
        gate_params = []
        for name, param in self.model.clip_model.named_parameters():
            if 'lora_w' in name:
                gate_params.append(param)
        params = [
            {'params': lora_params, 'lr': self.args.lr, 'weight_decay': 1e-2, 'betas': (0.9, 0.999)},
        ]
        self.optimizer = torch.optim.AdamW(params)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.total_iterations, eta_min=1e-6)








# def save_model(args, model):
#     if args.save is not None:
#         # to_save_model = model.module
#         path = os.path.join(args.save, f"state_dict.pth")
#         utils.torch_save(model, path)









