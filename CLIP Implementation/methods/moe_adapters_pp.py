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
import torch.nn.functional as F
from torch import nn

# from .. import utils
# from .loss import total_loss
from utils.train_utils import dyn_cosine_lr
from models.clip.peft import get_val_task_id_visual, get_eval_zero_shot











train_iteration = None


logger = logging.getLogger()





### Zero shot CLIP for Online Evaluation
class moe_adapters_pp(_Trainer):

    def __init__(self, args, **kwargs):
        super(moe_adapters_pp, self).__init__(args, **kwargs)
        self.visible_classes = self.args.visible_classes
        self.task_id = -1
        self.known_tasks = []

        self.useful_params = None
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
            ### Doing this breaks discrepency list handling during save/load. Would need to overhaul that code to dynamically update value
            # self.args.task_num = len(self.known_tasks)

        if self.task_id > 0:
            self.args.repeat_train = True
            if is_train:
                self.args.load = os.path.join(self.log_dir, str(self.task_id-1), "state_dict.pth")

        if is_train == True:
            self.args.eval_only = False
            self.args.autorouter = False
            self.args.ls = 0.2

        else:
            self.args.eval_only = True
            self.args.autorouter = True
            self.args.ls = 0.0
            self.args.load = os.path.join(self.log_dir, str(self.task_id), "state_dict.pth") # Reload from completed task when doing post-task evaluation




        ############################################################################################################################
        ### Initialize and/or get list of all trainable layers for task
        ############################################################################################################################

        # check the model
        ### This is true for all tasks>0 during training
        if self.args.load is not None and self.args.repeat_train:  # continual learning && use dyn_moe
            ### Just gets the number of initialized experts in each encoder block
            self.text_expert_num, self.image_expert_num = get_experts_and_router_num(self.args.load, self.args)
            self.args.text_expert_num_list = self.text_expert_num
            self.args.image_expert_num_list = self.image_expert_num
            print('use Dynamic MoE_Adapters for continual learning')
            print('text_expert_num in all layers:', self.text_expert_num)
            print('image_expert_num in all layers:', self.image_expert_num)
            # useful_params = get_useful_params_continual(args)
            ### Get the trainable layers in the model, with at most 1 expert per dynmoe layer
            useful_params = get_only_one_useful_params_continual(self.args)
            # print('router_num:', router_num)
        else:  # first train && use dyn_moe
            ### Inits experts in any layers that are set as TRUE for either LEAS or dyn moe args
            self.text_expert_num = [self.args.init_expert_num if dyn or id_det else 0
                                for dyn, id_det in zip(self.args.use_dyn_moe_layer_list_text, self.args.use_LEAS_list_text)]
            self.image_expert_num = [self.args.init_expert_num if dyn or id_det else 0
                                for dyn, id_det in zip(self.args.use_dyn_moe_layer_list_visual, self.args.use_LEAS_list_visual)]

            # image_expert_num = [args.init_expert_num if value else 0 for value in args.use_dyn_moe_layer_list_visual]
            ### This would then just be the number of experts present in each encoder block
            self.args.text_expert_num_list = self.text_expert_num
            self.args.image_expert_num_list = self.image_expert_num
            print(f'use Dynamic MoE_Adapters for continual learning, init the model with {int(self.args.init_expert_num)} experts')
            print('[model state] init training')
            ### Get the trainable layers in the model, with all dynmoe experts
            useful_params = get_useful_params_init(self.args)


        return useful_params




    def online_before_task(self, task_id):
        self.task_id = task_id
        
        ### Update arguments for current task and propagate changes to model
        self.useful_params = self.update_method_args()
        self.model.design_details["args"] = self.args
        self.model.args = self.args

        ### Call setup_task_model to update the state of the model (make a new copy for current task)
        self.model.setup_task_clip_model()
        self.model.freeze_clip(self.useful_params)

        # self.loss_interval = self.args.loss_interval
        self.num_batches = len(self.train_dataloader)
        self.batch_counter = 0
        self.total_iterations = self.args.online_iter * self.num_batches
        print("New num batches: ", self.num_batches)

        ### Remake the optimizer for the new training task and task clip model
        self.reset_opt()
        self.model = self.model.cuda()
        self.logit_scale = self.model.clip_model.logit_scale





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

        ### Makes a contiguous set of labels to match the order of class names input into the text encoder
        for j in range(len(y)):
            label = y[j].item()
            label_name = self.train_dataset.classes_names[label]
            y[j] = train_class_name_list.index(label_name)

        x,y = x.to(self.device), y.to(self.device)
        text_tokens = self.model.text_tokens


        self.scheduler(self.batch_counter)






        ### Train clip model adapters
        with torch.amp.autocast('cuda', enabled=self.use_amp):  
            # logit, image_features, text_features = self.model(x, text_tokens)
            # loss = self.criterion(logit, y)

            # -- get text embedding --
            embeddings = self.model(None, text_tokens)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

            # -- get image embedding --
            out, _ = self.model(x, None)
            out = out / out.norm(dim=-1, keepdim=True)


        # -- cross entropy loss --
        logits_per_image = self.logit_scale.exp() * out @ embeddings.t()
        loss = F.cross_entropy(logits_per_image, y, label_smoothing=self.args.ls)  # ce_loss
        # get mse_loss
        mse_loss_image = dyn_get_mse_loss_image(self.model.clip_model, self.image_expert_num, self.args)
        mse_loss_text = dyn_get_mse_loss_text(self.model.clip_model, self.text_expert_num, self.args)
        mse_loss = mse_loss_text + mse_loss_image
        # get total_loss
        loss = calc_total_loss(loss, mse_loss, self.args)

        # update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        _, preds = logits_per_image.topk(self.topk, 1, True, True)

        total_loss += loss.item()
        total_correct += torch.sum(preds == y.unsqueeze(1)).item()
        total_num_data += y.size(0)

        _set_model_iter(self.model.clip_model, self.batch_counter, self.args)  # set model iteration


        return total_loss, total_correct / total_num_data












    def online_after_task(self, task_id):

        # save activated experts & models
        dyn_save_model(self.args, self.model.clip_model)


        ### Reload the model under evaluation settings prior to eval phase
        self.task_id = task_id
        
        ### Update arguments for current task and propagate changes to model
        self.useful_params = self.update_method_args(is_train=False)
        self.model.design_details["args"] = self.args
        self.model.args = self.args

        ### Call setup_task_model to update the state of the model (make a new copy for current task)
        self.model.setup_task_clip_model()
        self.model.freeze_clip(self.useful_params)

        # self.loss_interval = self.args.loss_interval
        # self.num_batches = len(self.train_dataloader)
        # self.batch_counter = 0
        # self.total_iterations = self.args.online_iter * self.num_batches

        ### Remake the optimizer for the new training task and task clip model
        # self.reset_opt()
        self.model = self.model.cuda()
        # self.logit_scale = self.model.clip_model.logit_scale
























































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

        input_key = "images"
        image_enc = None
        self.model.eval()
    
        # top1, top5 = dyn_zeroshot_eval(model, dataloader, zeroshot_weights, args)

        # print(f"Top-1 accuracy: {top1:.2f}")
        # print(f"Top-5 accuracy: {top5:.2f}")
        # get_eval_acc(model, args)

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


                for j in range(len(y)):
                    label = y[j].item()
                    label_name = self.test_dataset.classes_names[label]
                    y[j] = test_class_name_list.index(label_name)

                x,y = x.to(self.device), y.to(self.device)

                text_tokens = self.model.text_tokens
                
                ### Makes a list of class embeddings for each task (should be redundant though since the tasks are selected by the recognition layer)
                zeroshot_weights = zeroshot_classifier(text_tokens, self.model.clip_model, self.args)




                # predict
                image_features, image_features_original = self.model.clip_model.dyn_moe_encode_image(x)
                image_features /= image_features.norm(dim=-1, keepdim=True)


                if self.unknown_test_task_id == True:
                    predicted_task_id = get_val_task_id_visual(-1)
                else:
                    predicted_task_id = test_task_id
                    
                zero_shot_flag = get_eval_zero_shot()
                if zero_shot_flag:
                    logits = 100.0 * image_features @ zeroshot_weights[0]
                else:
                    ### Offset because idx 0 is zeroshot
                    logits = 100.0 * image_features @ zeroshot_weights[predicted_task_id + 1]
                
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
        params = [v for k, v in self.model.clip_model.named_parameters() 
                    if any(k.startswith(s) for s in self.useful_params) and ".auto_encoder_list." not in k]
        params_rd = [v for k, v in self.model.clip_model.named_parameters() 
                    if any(k.startswith(s) for s in self.useful_params) and ".auto_encoder_list." in k]
        params_name = [k for k, v in self.model.clip_model.named_parameters() 
                    if any(k.startswith(s) for s in self.useful_params)]
        # print('===========trainable params============\n', params_name)
        # print('===========trainable params============', params_name)

        # print trainable params' information
        total_params_size = sum(p.numel() * p.element_size() for p in self.model.clip_model.parameters() if p.requires_grad)
        print('The number of Total Trainable Parameters:', sum(p.numel() for p in self.model.clip_model.parameters() if p.requires_grad))
        print(f"Total Trainable Parameters Memory Size: {total_params_size / 1024 / 1024:.2f} MB")

        

        self.optimizer = torch.optim.AdamW([
            {'params': params, 'lr': self.args.lr, 'weight_decay': self.args.wd, 'betas': (0.9, self.args.beta2)},
            {'params': params_rd, 'lr': self.args.lr_ae, 'weight_decay': self.args.wd, 'betas': (0.9, self.args.beta2)}
        ])
        self.scheduler = dyn_cosine_lr(self.optimizer, [self.args.lr, self.args.lr_ae], self.args.warmup_length, self.total_iterations)




























# def MSE_loss(x, reconstructed_x):
#     """
#     The reconstruction loss on all the features fed to Adapters.
#     See paper: Self-Expansion of Pre-trained Models with Mixture of Adapters for Continual Learning
#     """
#     rd_loss = nn.MSELoss()
#     return rd_loss(reconstructed_x, x)


def calc_total_loss(ce_loss,
               mse_loss,
               args):
    total_loss = args.ce_weight * ce_loss
    total_loss += args.mse_weight * mse_loss
    return total_loss





@torch.no_grad()
def zeroshot_classifier(text_tokens, model, args):

    zeroshot_weights = []

    #!# This was the original code for this function, but task_id did not do anything in the text encoder
    # for task_id in range(args.task_num):
    # for task_id in range(-1, args.task_num):
    #     zeroshot_weights_i = []
    #     for texts in text_tokens:
    #         # texts = [self.model.prompt_template.format(c) for c in labels]
    #         # texts = clip.tokenize(texts).cuda()  # tokenize
    #         # print("Texts shape: ", texts.shape) #[77]
    #         if args.non_text == True:
    #             # if len(texts.shape) == 2:
    #                 # text = texts.unsqueeze(0)
    #             class_embeddings = model.dyn_moe_encode_text(texts.unsqueeze(0), -1)  # embed with text encoder
    #         else:
    #             class_embeddings = model.dyn_moe_encode_text(texts.unsqueeze(0), task_id)  # embed with text encoder
    #         class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
    #         class_embedding = class_embeddings.mean(dim=0)
    #         class_embedding /= class_embedding.norm()
    #         zeroshot_weights_i.append(class_embedding)
    #     zeroshot_weights_i = torch.stack(zeroshot_weights_i, dim=1).cuda()
    #     zeroshot_weights.append(zeroshot_weights_i)

    #!# Since this is used for eval only, the task ID determined by the network is deterministic and identical same for all loops. 
    ### Not sure if this was the original intent, but after confirming this behavior we swapped it to just use one set of embeddings to save time
    zeroshot_weights_i = []
    for texts in text_tokens:
        # texts = [self.model.prompt_template.format(c) for c in labels]
        # texts = clip.tokenize(texts).cuda()  # tokenize
        # print("Texts shape: ", texts.shape) #[77]
        if args.non_text == True:
            # if len(texts.shape) == 2:
                # text = texts.unsqueeze(0)
            class_embeddings = model.dyn_moe_encode_text(texts.unsqueeze(0), -1)  # embed with text encoder
        else:
            # class_embeddings = model.dyn_moe_encode_text(texts.unsqueeze(0), task_id)  # embed with text encoder
            class_embeddings = model.dyn_moe_encode_text(texts.unsqueeze(0), 0)  # embed with text encoder
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm()
        zeroshot_weights_i.append(class_embedding)
    zeroshot_weights_i = torch.stack(zeroshot_weights_i, dim=1).cuda()
    # print("ZS Weights shape: ", zeroshot_weights_i.shape)
    for task_id in range(-1, args.task_num):
        zeroshot_weights.append(zeroshot_weights_i)


    # print("Zeroshot weights length: ", len(zeroshot_weights))
    # print("Zeroshot weights: ", zeroshot_weights)
    return zeroshot_weights




"""
Load the checkpoint from prior task
For each encoder, checks the number of experts and adds 2 if none are initialized yet
Returns 0 for any encoders not set to use dyn_moe

"""
def get_experts_and_router_num(model_path, args):
    # 加载保存的模型权重
    checkpoint = torch.load(model_path)

    # 提取state_dict
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    text_expert_counts = []
    image_expert_counts = []

    ### text_layer is 12 for vitB
    for i in range(args.text_layer):
        ### Both are 0 (false) for all encoders in vitB TIL clip
        if args.use_dyn_moe_layer_list_text[i] or args.use_LEAS_list_text[i]:
            # Construct the keywords that need to be checked.
            text_adaptmlp_key = f"transformer.resblocks.{i}.activated_experts_num"
            # Extract matching parameter names
            text_matching_keys = [key for key in state_dict.keys() if
                                    text_adaptmlp_key in key and "visual" not in key]
            current_text_experts_num = state_dict[text_matching_keys[0]][0].item()
            text_expert_counts.append(int(current_text_experts_num))
            if current_text_experts_num == [1.0]:
                text_expert_counts.append(args.init_expert_num + 2)  # If no matching parameter is found, enter the number of experts + 2.
        else:
            text_expert_counts.append(0)
    for i in range(args.vision_layer):
        ### Seems like it uses dyn moe for the last 6 encoders for ViT-B TIL CLIP
        if args.use_dyn_moe_layer_list_visual[i] or args.use_LEAS_list_visual[i]:
            image_adaptmlp_key = f"visual.transformer.resblocks.{i}.activated_experts_num"
            image_matching_keys = [key for key in state_dict.keys() if image_adaptmlp_key in key]
            current_image_experts_num = state_dict[image_matching_keys[0]][0].item()
            image_expert_counts.append(int(current_image_experts_num))
            if current_image_experts_num == [1.0]:
                image_expert_counts.append(args.init_expert_num + 2)  # If no matching parameter is found, enter the number of experts + 2.

        else:
            image_expert_counts.append(0)
    return text_expert_counts, image_expert_counts









def _update_mse_loss_info_visual(model, args, i):
    if args.use_LEAS_to_eval:
        cut_off_rate_new = args.cut_off_rate_visual
        cut_off_rate_frozen = args.cut_off_rate_visual
        if args.repeat_train is False:  # init
            cut_off_rate_new = args.cut_off_rate_visual
        model.visual.transformer.resblocks[i].update_mse_loss_avg_list(
            cut_off_rate_new=cut_off_rate_new,
            cut_off_rate_frozen=cut_off_rate_frozen,
        )
        model.visual.transformer.resblocks[i].update_mse_loss_std_list(
            cut_off_rate_new=cut_off_rate_new,
            cut_off_rate_frozen=cut_off_rate_frozen,
        )

def _update_mse_loss_info_text(model, args, i):
    if args.use_LEAS_to_eval:
        cut_off_rate_new = args.cut_off_rate_text
        cut_off_rate_frozen = args.cut_off_rate_text
        if args.repeat_train is False:  # init
            cut_off_rate_new = args.cut_off_rate_text
        model.transformer.resblocks[i].update_mse_loss_avg_list(
            cut_off_rate_new=cut_off_rate_new,
            cut_off_rate_frozen=cut_off_rate_frozen,
        )
        model.transformer.resblocks[i].update_mse_loss_std_list(
            cut_off_rate_new=cut_off_rate_new,
            cut_off_rate_frozen=cut_off_rate_frozen,
        )




def dyn_get_mse_loss_text(model, text_experts_num, args):
    mse_loss = torch.tensor(0.0, device="cuda:0")
    for i in range(args.text_layer):
        if args.use_dyn_moe_layer_list_text[i] or args.use_LEAS_list_text[i]:
            if model.transformer.resblocks[i].expansion_flag:
                if args.repeat_train:
                    mse_loss += model.transformer.resblocks[i].mse_loss_list[int(text_experts_num[i])]
                else:  # init
                    mse_loss += model.transformer.resblocks[i].mse_loss
    return mse_loss


def dyn_get_mse_loss_image(model, image_experts_num, args):
    mse_loss = torch.tensor(0.0, device="cuda:0")
    for i in range(args.vision_layer):
        if args.use_dyn_moe_layer_list_visual[i] or args.use_LEAS_list_visual[i]:
            if model.visual.transformer.resblocks[i].expansion_flag:
                if args.repeat_train:
                    mse_loss += model.visual.transformer.resblocks[i].mse_loss_list[int(image_experts_num[i])]
                else:  # init
                    mse_loss += model.visual.transformer.resblocks[i].mse_loss
    return mse_loss


def get_useful_params_init(args):
    useful_list = []
    for i in range(args.text_layer):
        if args.use_dyn_moe_layer_list_text[i] or args.use_LEAS_list_text[i]:
            for j in range(args.init_expert_num):
                # text-expert
                useful_list.append(f"transformer.resblocks.{i}.auto_encoder_list.{j}.")
                useful_list.append(f"transformer.resblocks.{i}.adaptmlp_list.{j}.")
            if args.single_router:
                # text-router
                useful_list.append(f"transformer.resblocks.{i}.w_noise1")
                useful_list.append(f"transformer.resblocks.{i}.router1")
            else:
                # text-router
                useful_list.append(f"transformer.resblocks.{i}.w_noise_list.{args.task_id}")
                useful_list.append(f"transformer.resblocks.{i}.router_list.{args.task_id}")
    for i in range(args.vision_layer):  
        if args.use_dyn_moe_layer_list_visual[i] or args.use_LEAS_list_visual[i]:
            for j in range(args.init_expert_num):
                # visual-expert
                useful_list.append(f"visual.transformer.resblocks.{i}.auto_encoder_list.{j}.")
                useful_list.append(f"visual.transformer.resblocks.{i}.adaptmlp_list.{j}.")
            if args.single_router:
                # visual-router
                useful_list.append(f"visual.transformer.resblocks.{i}.w_noise1")
                useful_list.append(f"visual.transformer.resblocks.{i}.router1")  
            else:
                # visual-router
                useful_list.append(f"visual.transformer.resblocks.{i}.w_noise_list.{args.task_id}")
                useful_list.append(f"visual.transformer.resblocks.{i}.router_list.{args.task_id}")
                
    return useful_list



"""
Gets the "useful" layers for each encoder block. Useful seems to mean trainable
Gets only one expert adapter per block, unlike get_useful_params_init()
    - The reason for this is that it gets the (potential) index of the next expanded expert per block
For vitb TIL CLIP:
    use_dyn_moe_layer_list_text is False for all text encoders, so useful list only gets the text w_noise and routers.
    use_dyn_moe_layer_list_visual is true for the last 6 encoders so:
        Gets the wnoise and router
        Gets the used experts for dynmoe encoder blocks

"""
def get_only_one_useful_params_continual(args):
    useful_list = []
    for i in range(args.text_layer):
        if args.use_dyn_moe_layer_list_text[i] or args.use_LEAS_list_text[i]:
            text_idx = args.text_expert_num_list[i]
            # text-expert
            useful_list.append(f"transformer.resblocks.{i}.auto_encoder_list.{text_idx}.")
            useful_list.append(f"transformer.resblocks.{i}.adaptmlp_list.{text_idx}.")
        if args.single_router is False:    
            # text-router
            useful_list.append(f"transformer.resblocks.{i}.w_noise_list.{args.task_id}")
            useful_list.append(f"transformer.resblocks.{i}.router_list.{args.task_id}")
        else:
            # text-router
            useful_list.append(f"transformer.resblocks.{i}.w_noise1")
            useful_list.append(f"transformer.resblocks.{i}.router1")
            
    for i in range(args.vision_layer):
        if args.use_dyn_moe_layer_list_visual[i] or args.use_LEAS_list_visual[i]:
            image_idx = args.image_expert_num_list[i]
            # visual-expert
            useful_list.append(f"visual.transformer.resblocks.{i}.auto_encoder_list.{image_idx}.")
            useful_list.append(f"visual.transformer.resblocks.{i}.adaptmlp_list.{image_idx}.")
        if args.single_router is False:
            # visual-router
            useful_list.append(f"visual.transformer.resblocks.{i}.w_noise_list.{args.task_id}")
            useful_list.append(f"visual.transformer.resblocks.{i}.router_list.{args.task_id}")
        else:
            # visual-router
            useful_list.append(f"visual.transformer.resblocks.{i}.w_noise1")
            useful_list.append(f"visual.transformer.resblocks.{i}.router1")
            
    return useful_list

  
def _set_model_iter(model, iteration, args):
    for i in range(args.text_layer):
        model.transformer.resblocks[i].set_iteration(iteration)
    for i in range(args.vision_layer):
        model.visual.transformer.resblocks[i].set_iteration(iteration)





def dyn_save_model(args, model):
    # get freq of experts
    for i in range(args.text_layer):
        if args.use_dyn_moe_layer_list_text[i]:
            # if args.single_router is False:
            # text activated record
            text_choose_map = model.transformer.resblocks[i].choose_map_text
            text_rate = F.normalize(text_choose_map, p=1.0, dim=0)  # get rate
            model.transformer.resblocks[i].update_freq_activated_experts(text_rate)  # save to model
            # update avg & std of RD_loss
            _update_mse_loss_info_text(model, args, i)
            # update activated expert_nums and routers
            text_activated_expert = torch.sum(model.transformer.resblocks[i].experts_mask)
        else:
            text_activated_expert = 0
        if args.use_LEAS_list_text[i]:
            # update avg & std of RD_loss
            _update_mse_loss_info_text(model, args, i)
            
    for i in range(args.vision_layer):
        if args.use_dyn_moe_layer_list_visual[i]:
            # if args.single_router is False:
            # image activated record
            visual_choose_map = model.visual.transformer.resblocks[i].choose_map_image
            image_rate = F.normalize(visual_choose_map, p=1.0, dim=0)  # get rate
            model.visual.transformer.resblocks[i].update_freq_activated_experts(image_rate)  # save to model
            # update avg & std of RD_loss
            _update_mse_loss_info_visual(model, args, i)
            # update activated expert_nums and routers
            visual_activated_expert = torch.sum(model.visual.transformer.resblocks[i].experts_mask)
        else:
            visual_activated_expert = 0
        if args.use_LEAS_list_visual[i]:
            # update avg & std of RD_loss
            _update_mse_loss_info_visual(model, args, i)
        
        print(f"Layer {i}, text expert_num: {text_activated_expert}, visual expert_num: {visual_activated_expert}")
    # Saving model
    if args.save is not None:
        to_save_model = model
        os.makedirs(args.save, exist_ok=True)
        save_file = os.path.join(args.save, f"state_dict.pth")
        # utils.torch_save(to_save_model, save_file)
        torch.save({"state_dict": to_save_model.state_dict()}, save_file)







