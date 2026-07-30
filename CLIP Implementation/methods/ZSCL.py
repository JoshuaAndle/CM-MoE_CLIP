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

from OCL_datasets.imagenet_ref import ImageNetSUB, conceptual_captions
from models.clip import clip_loader




train_iteration = None
logger = logging.getLogger()





### Zero shot CLIP for Online Evaluation
class ZSCL(_Trainer):

    def __init__(self, args, **kwargs):
        super(ZSCL, self).__init__(args, **kwargs)
        self.visible_classes = self.args.visible_classes
        self.task_id = -1
        self.known_tasks = []

        # self.useful_params = None
        # self.loss_interval  = -1
        self.num_batches = -1
        self.total_iterations = -1
        self.batch_counter = 0

        self.we_n = 0







    def setup_refs(self):

        test_preprocess = clip_loader._transform(self.model.clip_model.visual.input_resolution, is_train=False)

        ### Set up the reference datasets which will be used during all tasks
        self.ref_dataset = ImageNetSUB(
                preprocess=self.test_transform,
                location=self.args.data_location,
                batch_size=self.args.batchsize,
            )
        self.ref_image_iter = iter(self.ref_dataset.train_loader)


        # ref_sentences_cls = getattr(datasets, args.ref_sentences)
        # print(f"[Ref Sentences] {args.ref_sentences}")
        self.ref_sentences = conceptual_captions(
            location=os.path.join(self.args.data_location, "ConceptualCaptions"),
            batch_size=self.args.batchsize,
        )
        # if args.ref_sentences == "conceptual_captions":
        self.ref_texts = self.ref_sentences.train_dataset.captions
        self.ref_texts = clip_loader.tokenize(self.ref_texts, force_eot=True).cuda()

        self.ref_text_iter = iter(self.ref_sentences.train_loader)







    ### Update the args to reflect the original .sh training loop format throughout tasks and eval
    def update_method_args(self, is_train=True):

        self.args.is_train = is_train
        self.args.task_id = self.task_id
        self.args.save = os.path.join(self.log_dir, str(self.task_id))
        self.args.task_num = self.args.n_tasks

        if self.task_id not in self.known_tasks:
            self.known_tasks.append(self.task_id)


        if is_train == True:
            if self.task_id > 0:
                self.args.load = os.path.join(self.log_dir, str(self.task_id-1), "state_dict.pth")

            self.args.eval_only = False
            self.args.ls = 0.2

        else:
            self.args.eval_only = True
            self.args.ls = 0.0
            self.args.load = os.path.join(self.log_dir, str(self.task_id), "state_dict.pth") # Reload from completed task when doing post-task evaluation

        
        return




    def online_before_task(self, task_id):
        self.task_id = task_id
        
        ### Update arguments for current task and propagate changes to model
        self.update_method_args()
        self.model.design_details["args"] = self.args
        self.model.args = self.args

        ### Call setup_task_model to update the state of the model (make a new copy for current task)
        self.model.setup_task_clip_model()
        # self.model.freeze_clip(self.useful_params)

        if self.args.we == True:
            self.model.set_we_model()
        if self.args.l2 > 0:
            self.model.set_l2_model()




        # self.loss_interval = self.args.loss_interval
        self.num_batches = len(self.train_dataloader)
        self.batch_counter = 0
        self.total_iterations = self.args.online_iter * self.num_batches
        print("New num batches: ", self.num_batches)

        ### Remake the optimizer for the new training task and task clip model
        self.reset_opt(self.model.clip_model)
        self.model = self.model.cuda()
        self.logit_scale = self.model.clip_model.logit_scale

        ### We defer reference initialization since the model is not available within ZSCL.__init__()
        if task_id == 0:
            self.setup_refs()



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


        ### Train clip model adapters
        with torch.amp.autocast('cuda', enabled=self.use_amp):  
            ### ZSCL evaluates with the we_model periodically, we move this to the start of each batch
            with torch.no_grad():
                logit_eval, _, _ = self.model(x, text_tokens, use_we=True)
                # logit_eval, _, _ = self.model(x, text_tokens)
            # -- get text embedding --
            embeddings = self.model(None, text_tokens)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

            # -- get image embedding --
            out = self.model(x, None)
            out = out / out.norm(dim=-1, keepdim=True)


        ### Prediction accuracy calculation
        logits_per_image = self.logit_scale.exp() * out @ embeddings.t()        
        _, preds = logit_eval.topk(self.topk, 1, True, True)

        total_correct += torch.sum(preds == y.unsqueeze(1)).item()
        total_num_data += y.size(0)

        del logit_eval
        del preds
        
        ### Loss handling for ZSCL steps
        loss = F.cross_entropy(logits_per_image, y, label_smoothing=self.args.ls)  # ce_loss
        # loss = F.cross_entropy(logits_per_image, y)  # ce_loss

        del out
        del embeddings
        del logits_per_image


        if self.args.l2 > 0:
            loss_l2 = l2_loss(self.model.clip_model, self.model.clip_model_l2)
            loss += self.args.l2 * loss_l2


        if self.args.ref_dataset:
            try:
                ref_batch = next(self.ref_image_iter)
            except:
                self.ref_image_iter = iter(self.ref_dataset.train_loader)
                ref_batch = next(self.ref_image_iter)
            ref_images, _ = ref_batch

            ref_images = ref_images.cuda()
            # print("Shape of ref images:" , ref_images.shape)

            
            ### The provided code is inconsistent on how texts should be passed in. If CC is used they encode the full validation set
            ###   but if Imagenet is passed in they use just the batch labels, which seems much more reasonable memory-wise, so we go with this
            try:
                ref_text_batch = next(self.ref_text_iter)
            except:
                self.ref_text_iter = iter(self.ref_sentences.train_loader)
                ref_text_batch = next(self.ref_text_iter)

            ref_text_batch = clip_loader.tokenize(ref_text_batch, force_eot=True).cuda()
            # print("Shape of ref text batch: ", ref_text_batch.shape)


            with torch.no_grad():
                self.model.clip_model_ref.cuda()

                # -- get ref text embedding --
                ref_embeddings = self.model.clip_model_ref(None, ref_text_batch)
                ref_embeddings = ref_embeddings / ref_embeddings.norm(dim=-1, keepdim=True)

                # -- get ref image embedding --
                ref_out = self.model.clip_model_ref(ref_images, None)
                ref_out = ref_out / ref_out.norm(dim=-1, keepdim=True)

                self.model.clip_model_ref.cpu()


            # -- get image embedding --
            ref_out_current = self.model(ref_images, None)
            ref_out_current = ref_out_current / ref_out_current.norm(dim=-1, keepdim=True)

            logits_ref = self.logit_scale.exp() * ref_out @ ref_embeddings.t()


        # -- loss --
        logits_current = self.logit_scale.exp() * ref_out_current @ ref_embeddings.t()
        loss_ZSCL = distillation(logits_ref, logits_current, T=self.args.T)


        del ref_out
        del ref_embeddings
        del ref_out_current
        del ref_images, ref_batch, ref_text_batch

        # -- final loss --
        if self.args.image_loss:
            if self.args.weight_adjust:
                loss = loss + 0.5 * loss_ZSCL 
            else:
                loss = loss + 1.0 * loss_ZSCL 

        # transpose loss (called text_loss since its using the logit_per_text result tensor)
        if self.args.text_loss:
            logits_current_2 = logits_current.t()
            logits_ref_2 = logits_ref.t()
            loss_ZSCL_2 = distillation(logits_ref_2, logits_current_2, T=self.args.T)
            if self.args.weight_adjust:
                loss += 0.5 * loss_ZSCL_2
            else:
                loss += loss_ZSCL_2
        

        # update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


        # we
        if self.args.we and self.batch_counter % self.args.avg_freq == 0:
            self.we_n += 1
            print("Merging WE model", flush=True)
            merge_we(self.model.clip_model, self.model.clip_model_we, self.we_n)


        # self.print_memory()
        self.scheduler(self.batch_counter)

        total_loss += loss.item()

        return total_loss, total_correct / total_num_data







    def online_after_task(self, task_id):

        # save activated experts & models
        if self.args.we:
            save_model(self.args, self.model.clip_model_we)
        else:
            save_model(self.args, self.model.clip_model)


        ### Reload the model under evaluation settings prior to eval phase
        self.task_id = task_id
        
        ### Update arguments for current task and propagate changes to model
        self.update_method_args(is_train=False)
        # self.model.design_details["args"] = self.args
        # self.model.args = self.args








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


                for j in range(len(y)):
                    label = y[j].item()
                    label_name = self.test_dataset.classes_names[label]
                    y[j] = test_class_name_list.index(label_name)

                x,y = x.to(self.device), y.to(self.device)

                text_tokens = self.model.text_tokens
                
                # ### Makes a list of class embeddings for each task (should be redundant though since the tasks are selected by the recognition layer)
                # zeroshot_weights = zeroshot_classifier(text_tokens, self.model.clip_model, self.args)        
                class_embeddings = self.model(None, text_tokens, use_we=True)  # embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)


                # predict
                image_features = self.model(x, None, use_we=True)
                image_features /= image_features.norm(dim=-1, keepdim=True)

                logits = 100.0 * image_features @ class_embeddings.t()
                
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



    def reset_opt(self, model):
        exclude_params_name = ["logit_scale"]
        params = [v for k, v in model.named_parameters() if k not in exclude_params_name]

        # optimizer
        self.optimizer = torch.optim.AdamW(params, lr=self.args.lr, weight_decay=self.args.wd, betas=(0.9, self.args.beta2))
        self.scheduler = cosine_lr(self.optimizer, self.args.lr, self.args.warmup_length, self.total_iterations)







# @torch.no_grad()
# def zeroshot_classifier(text_tokens, model, args):

#     zeroshot_weights = []
#     # for task_id in range(args.task_num):
#     zeroshot_weights_i = []
#     for texts in text_tokens:
#         # texts = [self.model.prompt_template.format(c) for c in labels]
#         # texts = clip.tokenize(texts).cuda()  # tokenize
#         # print("Texts shape: ", texts.shape) #[77]
#         class_embeddings = model.encode_text(texts.unsqueeze(0))  # embed with text encoder
#         class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)

#         class_embedding = class_embeddings.mean(dim=0)
#         class_embedding /= class_embedding.norm()
#         zeroshot_weights_i.append(class_embedding)


#         zeroshot_weights_i = torch.stack(zeroshot_weights_i, dim=1).cuda()
#         zeroshot_weights.append(zeroshot_weights_i)
#         print("Zeroshot weights length: ", len(zeroshot_weights))
#         print("Zeroshot weights: ", zeroshot_weights)
#     return zeroshot_weights





def save_model(args, model):
    if args.save is not None:
        # to_save_model = model.module
        save_path = os.path.join(args.save, f"state_dict.pth")
        # torch_save(model, path)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({"state_dict": model.state_dict()}, save_path)
        print("Checkpoint saved to", save_path)

def merge_we(model_0, model_1, sma_count):
    for param_q, param_k in zip(model_0.parameters(), model_1.parameters()):
        param_k.data = (param_k.data * sma_count + param_q.data) / (1.0 + sma_count)
    return model_1



def distillation(t, s, T=2):
    p = F.softmax(t / T, dim=1)
    loss = F.cross_entropy(s / T, p, reduction="mean") * (T ** 2)
    return loss



def l2_loss(model, model_ref):
    loss = 0.0
    ### Note: model_ref is the model_l2
    ### Modified for keeping model_l2 on cpu to reduce memory cost
    for param_q, param_k in zip(model.parameters(), model_ref.parameters()):
        loss += F.mse_loss(param_q, param_k.detach().to(param_q.device), reduction="sum")
    return loss

# def l2_loss(model, model_ref):
#     loss = 0.0
#     ### Note: model_ref is the model_l2
#     for param_q, param_k in zip(model.parameters(), model_ref.parameters()):
#         loss += F.mse_loss(param_q, param_k.detach(), reduction="sum")
#     return loss

















def assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lrs, warmup_length, steps):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)

    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(param_group, lr)

    return _lr_adjuster




