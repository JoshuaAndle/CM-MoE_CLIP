from calendar import c
import os
import sys
import time
import math
import random
import logging
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from collections import defaultdict
from randaugment import RandAugment

from models import get_model
from OCL_datasets import get_dataset, get_multipart_dataset
from utils.augment import Cutout
from utils.memory import Memory
from utils.online_sampler import OnlineSampler, OnlineMultiDatasetSampler, OnlineTestSampler
from utils.indexed_dataset import IndexedDataset
from utils.train_utils import select_optimizer, select_scheduler
from OCL_datasets.multiDatasets import multiDatasets

##################################################################
# This is trainer with a DistributedDataParallel                 #
# Based on the following tutorial:                               #
# https://github.com/pytorch/examples/blob/main/imagenet/main.py #
# And Deit by FaceBook                                           #
# https://github.com/facebookresearch/deit                       #
##################################################################


class _Trainer():

    def __init__(self, args, **kwargs) -> None:
        print(np.__version__)
        self.args = args

        print(args.use_dyn_moe_layer_list_visual)
        print(args.use_LEAS_list_visual)
        print(args.force_expansion_list)

        # print(args)
        self.method = args.method

        self.label_type = args.label_type

        self.per_task_datasets = args.per_task_datasets
        # self.distribution_shift = args.distribution_shift_type

        self.experiment_type = args.experiment_type
        self.train_task = args.train_task
        self.task_id = 0
        self.true_task_id = 0

        self.max_subnets = args.max_subnets
        self.skip_merging = False

        self.label_offset = {}
        self.exposed_classes = []
        self.exposed_classes_names = []
        self.batch_exposed_classes = []
        self.batch_exposed_classes_names = []
        self.seen = 0

        self.skip_evals = args.skip_evals
        self.unknown_train_task_id = args.unknown_train_task_id
        self.unknown_test_task_id = args.unknown_test_task_id

        self.n = args.n
        self.m = args.m
        self.rnd_NM = args.rnd_NM

        self.n_tasks = args.n_tasks
        self.dataset_name = args.dataset
        self.rnd_seed = args.rnd_seed

        self.memory_size = args.memory_size
        self.use_memory = self.memory_size > 0
        self.log_path = args.log_path
        self.model_name = args.model_name
        self.opt_name = args.opt_name
        self.sched_name = args.sched_name
        self.batchsize = args.batchsize
        self.n_worker = args.n_worker
        self.lr = args.lr
        self.init_model = args.init_model
        self.init_opt = args.init_opt
        self.topk = args.topk
        self.use_amp = args.use_amp
        self.transforms = args.transforms
        self.reg_coef = args.reg_coef
        self.data_dir = args.data_dir
        self.debug = args.debug
        self.note = args.note
        # self.selection_size = args.selection_size

        self.eval_period = args.eval_period
        self.temp_batchsize = args.temp_batchsize
        self.online_iter = args.online_iter
        # self.num_gpus = args.num_gpus
        # self.workers_per_gpu = args.workers_per_gpu
        # self.imp_update_period = args.imp_update_period

        self.zero_shot_evaluation = args.zero_shot_evaluation
        self.zero_shot_dataset = args.zero_shot_dataset

        # for distributed training
        self.dist_backend = 'nccl'
        self.dist_url = 'env://'

        # self.lr_step = args.lr_step  # for adaptive LR
        # self.lr_length = args.lr_length  # for adaptive LR
        # self.lr_period = args.lr_period  # for adaptive LR

        # self.memory_epoch = args.memory_epoch  # for RM
        # self.distilling = args.distilling  # for BiC
        # self.agem_batch = args.agem_batch  # for A-GEM
        # self.mir_cands = args.mir_cands  # for MIR


        self.result_dicts = {} # For storing misc/meta evaluation metrics



        self.start_time = time.time()
        self.temp_time = time.time()
        self.num_updates = 0
        self.train_count = 0

        self.ngpus_per_nodes = torch.cuda.device_count()
        self.world_size = 1
        if "WORLD_SIZE" in os.environ and os.environ["WORLD_SIZE"] != '':
            self.world_size = int(os.environ["WORLD_SIZE"]) * self.ngpus_per_nodes
        else:
            self.world_size = self.world_size * self.ngpus_per_nodes
        self.distributed = self.world_size > 1

        if self.temp_batchsize is None:
            self.temp_batchsize = self.batchsize // 2
        if self.temp_batchsize > self.batchsize:
            self.temp_batchsize = self.batchsize
        # self.memory_batchsize = self.batchsize - self.temp_batchsize
        self.memory_batchsize = self.batchsize
        print("Memory Batchsize is : ", self.memory_batchsize )


        ### Removed date to make file paths more predictable for loading data from
        if 'debug' not in self.note:
            self.log_dir = os.path.join(
                self.log_path, self.dataset_name,
                f"{self.n_tasks}TASKS",
                f"{self.method}METHOD",
                f"{self.rnd_seed}SEED",
                f"{self.note}")
        else:
            self.log_dir = os.path.join(self.log_path, "debug")
        os.makedirs(self.log_dir, exist_ok=True)




    ######################################################################################################
    ### Setup functions
    ######################################################################################################

    def setup_zero_shot_dataset(self, dataset_name):
        dataset, mean, std, _, _ = get_dataset(dataset_name)
        test_transform = transforms.Compose([
            transforms.Resize((self.inp_size, self.inp_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_dataset = dataset(root=self.data_dir,
                               train=False,
                               download=False,
                               transform=test_transform)
        classes_names = test_dataset.classes_names

        test_dataloader = DataLoader(test_dataset,
                                     batch_size=self.batchsize,
                                     shuffle=False,
                                     download=False,
                                     num_workers=self.n_worker,
                                     pin_memory=True)
        return test_dataloader, classes_names


    #!# Changed transform handling to pre-apply scaling and normalization directly to dataset with augmentations being called at runtime
    ### This is to handle multi-dataset setups where normalizations may differ between datasets, otherwise memory buffers would have to track 
    ###    transforms non-uniformly for mixed batches where samples come from various past tasks' datasets
    def setup_transforms(self):
        train_transform = []
        self.cutmix = "cutmix" in self.transforms

        if self.rot_degree > 0:
            ### Size of center crop needed to avoid inclusion of any zero-padded pixels during rotation
            #!# It is still to-be-determined if it is better to allow inclusion of the zero-padded background during rotation or if it will give erroneous information about rotations
            crop_size = math.floor(self.inp_size / math.sqrt(2))
            ### Added conversion to RGB since some datasets used are mixed RGB/Greyscale
            self.train_transform = transforms.Compose([
                transforms.Resize((self.inp_size, self.inp_size)),
                transforms.Lambda(lambda img: img.convert("RGB")),
                transforms.RandomRotation(degrees = (self.rot_degree-10,self.rot_degree+10), expand=False),
                transforms.CenterCrop(crop_size),
                transforms.Resize((self.inp_size, self.inp_size)),
                transforms.RandomCrop(self.inp_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
              ])
            # self.train_transform = transforms.Compose([
            #     transforms.Resize((self.inp_size, self.inp_size)),
            #     transforms.Lambda(lambda img: img.convert("RGB")),
            #     transforms.ToTensor(),
            #     transforms.Normalize(self.mean, self.std),
            #     transforms.RandomRotation(degrees = (self.rot_degree-10,self.rot_degree+10), expand=False),
            #     transforms.CenterCrop(crop_size),
            #     transforms.Resize((self.inp_size, self.inp_size)),
            #     transforms.RandomCrop(self.inp_size, padding=4),
            #     transforms.RandomHorizontalFlip(),
            #   ])

            self.test_transform = transforms.Compose([
                transforms.Resize((self.inp_size, self.inp_size)),
                transforms.Lambda(lambda img: img.convert("RGB")),
                transforms.RandomRotation(degrees = (self.rot_degree-10,self.rot_degree+10), expand=False),
                transforms.CenterCrop(crop_size),
                transforms.Resize((self.inp_size, self.inp_size)),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ])

        elif self.blur_degree > 0:
            ### Size of center crop needed to avoid inclusion of any zero-padded pixels during rotation
            #!# It is still to-be-determined if it is better to allow inclusion of the zero-padded background during rotation or if it will give erroneous information about rotations
            blur_magnitudes = [0,5,9,13,17]
            kernel_size = blur_magnitudes[self.blur_degree]
            ### Added conversion to RGB since some datasets used are mixed RGB/Greyscale
            self.train_transform = transforms.Compose([
                transforms.Resize((self.inp_size, self.inp_size)),
                transforms.Lambda(lambda img: img.convert("RGB")),
                transforms.GaussianBlur(kernel_size=kernel_size),
                transforms.RandomCrop(self.inp_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
              ])

            self.test_transform = transforms.Compose([
                transforms.Resize((self.inp_size, self.inp_size)),
                transforms.Lambda(lambda img: img.convert("RGB")),
                transforms.GaussianBlur(kernel_size=kernel_size),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ])




        else:
            self.train_transform = transforms.Compose([
                transforms.Resize((self.inp_size, self.inp_size)),
                transforms.Lambda(lambda img: img.convert("RGB")),
                transforms.RandomCrop(self.inp_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
              ])

            self.test_transform = transforms.Compose([
                transforms.Resize((self.inp_size, self.inp_size)),
                transforms.Lambda(lambda img: img.convert("RGB")),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ])
 


        # self.test_transform = transforms.Compose([
        #     transforms.Resize((self.inp_size, self.inp_size)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(self.mean, self.std),
        # ])

    def setup_dataset(self, splits = None, split_num = None):
        ### Gets the standard torchvision dataset for applicable datasets
        ### Added check since only splitDataset classes have arguments for split information. Could probably use kwargs instead
        if splits and split_num:
            self.train_dataset = self.dataset(root=self.data_dir,
                                              train=True,
                                              download=True,
                                              transform=self.train_transform,
                                              # transform=self.train_initial_transform,
                                              splits=splits,
                                              split_num=split_num)
            self.test_dataset = self.dataset(root=self.data_dir,
                                             train=False,
                                             download=True,
                                             transform=self.test_transform,
                                             splits=splits,
                                             split_num=split_num)

        else:
            self.train_dataset = self.dataset(root=self.data_dir,
                                              train=True,
                                              download=True,
                                              transform=self.train_transform)
            self.test_dataset = self.dataset(root=self.data_dir,
                                             train=False,
                                             download=True,
                                             transform=self.test_transform)


    def setup_dataloaders(self):
        ### get dataloaders once dataset is set up

        #!# Might need to modify this for multi-dataset streams, not sure how it works with the torchvision sequence currently
        ### Wrapper dataset class that returns index along with sample
        train_dataset = IndexedDataset(self.train_dataset)
        test_dataset = IndexedDataset(self.test_dataset)
        if self.per_task_datasets == False:
            self.train_sampler = OnlineSampler(train_dataset, self.n_tasks, self.m,self.n, self.rnd_seed, self.rnd_NM)
            self.test_sampler = OnlineSampler(test_dataset, self.n_tasks, self.m,self.n, self.rnd_seed, self.rnd_NM)
        else:
            self.train_sampler = OnlineMultiDatasetSampler(train_dataset, self.n_tasks)
            self.test_sampler = OnlineMultiDatasetSampler(test_dataset, self.n_tasks)
        

        self.train_dataloader = DataLoader(train_dataset,
                                           batch_size=self.batchsize,
                                           sampler=self.train_sampler,
                                           num_workers=self.n_worker,
                                           pin_memory=True)
        
        self.test_dataloader = DataLoader(self.test_dataset,
                                          batch_size=self.batchsize,
                                          shuffle=False,
                                          sampler=self.test_sampler,
                                          num_workers=self.n_worker,
                                          pin_memory=True)













    def setup_distributed_model(self):
        logging.info("Building model...")
        self.model = self.model.to(self.device)
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        self.model.to(self.device)
        self.model_without_ddp = self.model
        self.criterion = self.model_without_ddp.loss_fn if hasattr(self.model_without_ddp, "loss_fn") else nn.CrossEntropyLoss(reduction="mean")
        
        if self.args.defer_setup == True:
            self.optimizer = None
            self.scheduler = None    
        else:
            self.optimizer = select_optimizer(self.opt_name, self.lr, self.model)
            self.scheduler = select_scheduler(self.sched_name, self.optimizer)

        n_params = sum(p.numel() for p in self.model_without_ddp.parameters())
        logging.info(f"Total Parameters :\t{n_params}")
        n_params = sum(p.numel() for p in self.model_without_ddp.parameters() if p.requires_grad)
        logging.info(f"Learnable Parameters :\t{n_params}")

    def setup_for_distributed(self, is_master):
        """
        This function disables printing when not in master process
        """
        self.setup_root_logger(is_master=is_master)
        import builtins as __builtin__
        builtin_print = __builtin__.print

        def print(*args, **kwargs):
            force = kwargs.pop('force', False)
            if is_master or force:
                builtin_print(*args, **kwargs)

        __builtin__.print = print

    def setup_root_logger(self, is_master=True, filename="log.txt"):
        if is_master:
            root_logger = logging.getLogger()
            root_logger.setLevel(logging.INFO)
            ch = logging.StreamHandler(stream=sys.stdout)
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter("%(asctime)s | %(message)s")
            ch.setFormatter(formatter)
            root_logger.addHandler(ch)

            fh = logging.FileHandler(os.path.join(self.log_dir, filename), mode='w')
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            root_logger.addHandler(fh)
            return root_logger
        else:
            pass








    ######################################################################################################
    ### Core training loop
    ######################################################################################################

    def setup_trainer(self):
        self.gpu = 0
        self.device = torch.device(self.gpu)
        self.setup_for_distributed(True)

        # logging.info(str(self.args))

        if self.rnd_seed is not None:
            random.seed(self.rnd_seed)
            np.random.seed(self.rnd_seed)
            torch.manual_seed(self.rnd_seed)
            torch.cuda.manual_seed(self.rnd_seed)
            torch.cuda.manual_seed_all(self.rnd_seed)  # if use multi-GPU
            cudnn.deterministic = False
            # cudnn.deterministic = True
            # logging.info('You have chosen to seed training. '
            #              'This will turn on the CUDNN deterministic setting, '
            #              'which can slow down your training considerably! '
            #              'You may see unexpected behavior when restarting '
            #              'from checkpoints.')
        cudnn.benchmark = False

        logging.info(f"Select a CIL method ({self.method})")

        if self.per_task_datasets == False:
            self.dataset, self.mean, self.std, self.blur_degree, self.rot_degree, self.n_classes = get_dataset(self.dataset_name)

            #*# To-Do: Address num_classes for models that are affected by it when doing per_task_datasets
            self.model, self.inp_size = get_model(
                method=self.method, args=self.args, model_name=self.model_name,
                num_classes=self.n_classes, device=self.device,
                peft_encoder=self.args.peft_encoder,
                log_dir=self.log_dir
            )
            self.setup_transforms()
            self.setup_dataset()
            self.setup_distributed_model()

        else:
            self.multidatasets_dict = get_multipart_dataset(self.dataset_name)
            train_datasets, test_datasets = [], []

            ### We need the total number of classes prior to constructing the model
            total_classes = 0
            for task in self.multidatasets_dict.keys():
                _, _, _, _, _, n_classes, _, _ = self.multidatasets_dict[task]
                total_classes += n_classes
            
            self.n_classes = total_classes
            
            #*# To-Do: Address num_classes for models that are affected by it when doing per_task_datasets
            #!# Need to setup model here to get inp_size so we can get each datasets transforms in setup_transforms()
            self.model, self.inp_size = get_model(
                method=self.method, args=self.args, model_name=self.model_name,
                num_classes=self.n_classes, device=self.device,
                peft_encoder=self.args.peft_encoder,
                log_dir=self.log_dir
            )
            
            for i, task in enumerate(self.multidatasets_dict.keys()):
                self.dataset, self.mean, self.std, self.blur_degree, self.rot_degree, _, total_splits, split_num = self.multidatasets_dict[task]

                self.setup_transforms()
                self.setup_dataset(total_splits, split_num)
                train_datasets.append(self.train_dataset)
                test_datasets.append(self.test_dataset)
            self.train_dataset = multiDatasets(train_datasets, preprocessed=True) 
            self.test_dataset =  multiDatasets(test_datasets,  preprocessed=True) 
            self.setup_distributed_model()
        
        logging.info(f"Building model ({self.model_name})")





        if self.use_memory:
            self.memory = Memory(self.args)
    
        self.total_samples = len(self.train_dataset)

        self.setup_dataloaders()

        
        logging.info(f"Incrementally training {self.n_tasks} tasks")




    def train(self, single_task=False):
        """
        Standard sequential training of all online tasks
        Test accuracy is evaluated offline on all known test sets after each task finishes
        """
        task_records = defaultdict(list)
        eval_results = defaultdict(list)
        samples_cnt = 0

        num_eval = self.eval_period

        for task_id in range(self.n_tasks):
            if single_task == True and task_id > 0:
                print("Only training and evaluating first subnetwork")
                break

            print("\n\n\n\n")
            logging.info("#" * 50)
            logging.info(f"# Task {task_id} Session")
            logging.info("#" * 50)
            logging.info("[2-1] Prepare a datalist for the current task")

            print("Exposed class names at start of task ", task_id, ": ", self.exposed_classes_names)

            if self.use_memory:
                self.memory.add_task(task_id, self.train_dataset.classes_names_by_task[task_id])

            self.print_memory(verbose=True)

            self.task_id = task_id
            self.true_task_id = task_id

            self.result_dicts[task_id] = {
                                            "accuracy": {
                                                            "running_training": -1.0,
                                                            "online": -1.0,
                                                            "finetuning": -1.0,
                                                            "eval": [],
                                                            "forgetting": []
                                            },
                                            "times": {
                                                        "total_task_training": 0.0,
                                                        "online_step_only": 0.0,
                                                        "before_task": 0.0,
                                                        "tokenization": 0.0,
                                                        "training autoencoder": 0.0,
                                                        "training model": 0.0,
                                                        "freezing": 0.0,
                                                        "finetuning": 0.0,
                                                        "collect acts": 0.0,
                                                        "metric calculation": 0.0,
                                                        "clustering": 0.0,
                                                        "merging": 0.0,
                                                        "label management": 0.0,
                                                        "memory management": 0.0,
                                                        "eval": 0.0
                                            },
                                            "clustering": {
                                                            "predicted clusters": [],
                                                            "merged clusters": []
                                            },
                                            "metrics": {
                                                            "choosemaps": {},
                                                            "clustering choosemaps": {},
                                                            "frozen experts": {}

                                            }
                                        }


            ### Changes which task classes are included in the train_dataloader
            self.train_sampler.set_task(task_id)

            self.temp_time = time.time()
            ### Any method-specific setup
            self.online_before_task(task_id)
            self.result_dicts[self.task_id]["times"]["before_task"] = time.time()-self.temp_time

            self.print_memory(verbose=True)
            self.temp_time = time.time()

            task_time = 0.0

            temp_time_task_full = time.time()
            temp_time_batch_full = time.time()
            running_train_loss, running_train_acc = 0., 0.
            for i, (images, labels, idx) in enumerate(self.train_dataloader):
                temp_time_batch_partial = time.time()

                ### Break after the first 500 samples of training data if debugging
                if self.debug and (i + 1) * self.temp_batchsize >= 600:
                    print("Temp batch size ", self.temp_batchsize, " images shape: ", images.shape)
                    break
                samples_cnt += images.size(0) * self.world_size

                #!# Added passing batch number into step for debugging purposes, wont work with methods other than CMMoE
                # loss, acc = self.online_step(images, labels, idx, i)
                loss, acc = self.online_step(images, labels, idx)

                running_train_loss += loss
                running_train_acc += acc

                batch_prep_delay = ((time.time()-temp_time_batch_full) - (time.time()-temp_time_batch_partial))
                
                # print("Batch acc: ", acc, " - Batch loss: ", loss)
                self.report_training(samples_cnt, running_train_loss/(i+1), running_train_acc/(i+1), step="online")
                # self.report_training(samples_cnt, acc, running_train_acc/(i+1), step="online")
                # print("Time needed for reporting training: ", (time.time()-temp_time_reporting))
                if batch_prep_delay > 0.5:
                    print(f"Batch loading bottleneck occured for : {batch_prep_delay} sec")
                
                temp_time_batch_full = time.time()

                ### Added this to try and avoid erratic runtime delays caused by the computing cluster
                ###   (sometimes batches would hang for several minutes despite taking <1 second to train, unclear why)
                task_time += temp_time_batch_full - temp_time_batch_partial


            self.result_dicts[self.task_id]["times"]["online_step_only"] = task_time
            self.print_memory(verbose=True)
            
            self.online_after_task(task_id)

            self.print_memory(verbose=True)

            self.result_dicts[self.task_id]["times"]["total_task_training"] = time.time() - temp_time_task_full
            self.result_dicts[self.task_id]["accuracy"]["running_training"] = running_train_acc/(i+1)

            temp_time_eval = time.time()
            if self.skip_evals == False:
                ### Do offline testing on all previously seen tasks to assess forgetting
                test_accs = np.zeros((self.n_tasks))
                task_forgetting = np.zeros((self.n_tasks))

                eval_tasks_count = (task_id + 1) if single_task == False else self.n_tasks
                for test_task in range(eval_tasks_count):
                    ### Changes which task classes are included in the train_dataloader
                    self.test_sampler.set_task(test_task)

                    ### Get the test accuracy on the given class. When testing transfer acc we manually set the subnetwork being used
                    if self.args.method == "CMMoE":
                        task_acc = self.offline_evaluate(test_task, use_current_subnet=single_task)
                    elif self.args.method == "moe_adapters_pp":
                        task_acc = self.offline_evaluate(test_task)
                    else:
                        task_acc = self.offline_evaluate(test_task)
                    test_accs[test_task] = task_acc

                self.result_dicts[task_id]["accuracy"]["eval"] = test_accs

                for past_task in range(task_id):

                    ### Get the difference in eval acc for each past task up to the current test task
                    initial_accs = self.result_dicts[past_task]["accuracy"]["eval"]
                    print(f"Initial accuracies for task {past_task}: {initial_accs}")
                    task_forgetting[past_task] = (initial_accs[past_task] - test_accs[past_task])

                self.result_dicts[task_id]["accuracy"]["forgetting"] = task_forgetting

            self.result_dicts[self.task_id]["times"]["eval"] = time.time() - temp_time_eval

            ### Run evaluation without training for all zero-shot transfer dataset targets
            if self.zero_shot_evaluation == True and self.skip_evals == False:
                assert hasattr(self, 'offline_evaluate')

                with open(os.path.join(self.log_dir, 'result.txt'), 'w') as f:
                    f.write(f"Dataset:{self.dataset_name} | A_auc {A_auc:.5f} | A_avg {A_avg:.5f} | A_last {A_last:.5f} | F_last {F_last:.5f}\n")

                print("zero shot evaluation")
                for zs_dataset_name in self.zero_shot_dataset:
                    zs_dataset, zs_classes_names = self.setup_zero_shot_dataset(zs_dataset_name)
                    zs_acc = self.zeroshot_evaluate(zs_dataset, zs_classes_names)
                    line = f"Dataset:{zs_dataset_name} | test_acc:{zs_acc:.4f}"
                    print(line)
                    with open(os.path.join(self.log_dir, 'result.txt'), 'a') as f:
                        f.write(line + '\n')

            print("Results for task: \n", self.result_dicts[task_id])

        torch.save(self.result_dicts, (self.log_dir + "/result_dicts.pt"))

        print("\n\n\n\n\n\n\n\n\n")
        for key in self.result_dicts.keys():
            print("\n")
            print(self.result_dicts[key]["times"])
            print(self.result_dicts[key]["accuracy"])
            # self.result_dicts[self.task_id]["times"]["total_task_training"] = time.time() - temp_time_task_full
            # self.result_dicts[self.task_id]["accuracy"]["running_training"] = running_train_acc/(i+1)




    def train_ae_only(self):
        """
        Standard sequential training of all online tasks
        Test accuracy is evaluated offline on all known test sets after each task finishes
        """
        task_records = defaultdict(list)
        eval_results = defaultdict(list)
        samples_cnt = 0

        num_eval = self.eval_period

        for task_id in range(self.n_tasks):
            print("\n\n\n\n")
            logging.info("#" * 50)
            logging.info(f"# Task {task_id} Session")
            logging.info("#" * 50)
            logging.info("[2-1] Prepare a datalist for the current task")
            print("", flush=True)

            self.task_id = task_id
            self.true_task_id = task_id

            self.result_dicts[task_id] = {
                                            "accuracy": {
                                                            "online": -1.0,
                                                            "finetuning": -1.0,
                                                            "eval": [],
                                                            "forgetting": []
                                            },
                                            "times": {
                                                        "before_task": 0.0,
                                                        "tokenization": 0.0,
                                                        "training autoencoder": 0.0,
                                                        "training model": 0.0,
                                                        "freezing": 0.0,
                                                        "finetuning": 0.0,
                                                        "collect acts": 0.0,
                                                        "metric calculation": 0.0,
                                                        "clustering": 0.0,
                                                        "merging": 0.0,
                                                        "label management": 0.0,
                                                        "memory management": 0.0,
                                                        "eval": 0.0
                                            },
                                            "clustering": {
                                                            "predicted clusters": [],
                                                            "merged clusters": []
                                            },
                                            "metrics": {
                                                            "choosemaps": {},
                                                            "clustering choosemaps": {},
                                                            "frozen experts": {},
                                                            "ae_losses": [0]*self.n_tasks,


                                            }
                                        }


            ### Changes which task classes are included in the train_dataloader
            self.train_sampler.set_task(task_id)

            self.temp_time = time.time()
            ### Any method-specific setup
            if task_id > 0:
                self.online_before_task(task_id, manual_subnet=0)
            else:
                self.online_before_task(task_id)

            self.result_dicts[self.task_id]["times"]["before_task"] = time.time()-self.temp_time

            self.print_memory(verbose=True)
            self.temp_time = time.time()


            running_train_loss, running_train_acc = 0., 0.
            for i, (images, labels, idx) in enumerate(self.train_dataloader):
                ### Break after the first 500 samples of training data if debugging
                if self.debug and (i + 1) * self.temp_batchsize >= 1000:
                    print("Temp batch size ", self.temp_batchsize, " images shape: ", images.shape)
                    break
                samples_cnt += images.size(0) * self.world_size

                #!# Added passing batch number into step for debugging purposes, wont work with methods other than CMMoE
                self.online_step_ae_only(images, labels, idx)




        ### Go through all tasks to check and store the loss of each autoencoder on the first batch of data
        for test_task in range(self.n_tasks):
            ### Changes which task classes are included in the dataloaders
            self.train_sampler.set_task(test_task)
            self.test_sampler.set_task(test_task)
            self.get_ae_loss(test_task, split="train")



        torch.save(self.result_dicts, (self.log_dir + "/result_dicts.pt"))

        print("\n\n\n\n\n\n\n\n\n")
        for key in self.result_dicts.keys():
            print("\n", self.result_dicts[key])






    def measure_ood_transfer(self):
        """
        Train a single subnetwork and evaluate accuracy and metrics on progressively shifted tasks
        """



        """
        Steps
        1. Train task 0 as normal
        2. For every subsequent task, get test accuracy using subnet 0
        3. Get AutoEncoder loss (?) on each task to see how closely they would be recognized as ID vs OoD
        4. After getting task accuracy, get activations and specified metrics
            - This should be a virtual function implemented in our own method since other benchmarks won't necessarily have subnets or metrics



        """
        """
        Standard sequential training of all online tasks
        Test accuracy is evaluated offline on all known test sets after each task finishes
        """
        samples_cnt = 0

        num_eval = self.eval_period

        task_id = self.train_task


        print("\n\n\n\n")
        logging.info("#" * 50)
        logging.info(f"# Task {task_id} Session")
        logging.info("#" * 50)
        logging.info("[2-1] Prepare a datalist for the current task")

        print("Exposed class names at start of task ", task_id, ": ", self.exposed_classes_names)

        if self.use_memory:
            self.memory.add_task(task_id, self.train_dataset.classes_names_by_task[task_id])

        self.print_memory(verbose=True)

        self.task_id = task_id
        self.true_task_id = task_id

        self.result_dicts = {}
        self.result_dicts[task_id] = {
                                        "accuracy": {
                                                        "online": -1.0,
                                                        "finetuning": -1.0,
                                                        "eval": [],
                                                        "forgetting": []
                                        },
                                        "times": {
                                                    "before_task": 0.0,
                                                    "tokenization": 0.0,
                                                    "training autoencoder": 0.0,
                                                    "training model": 0.0,
                                                    "freezing": 0.0,
                                                    "finetuning": 0.0,
                                                    "collect acts": 0.0,
                                                    "metric calculation": 0.0,
                                                    "clustering": 0.0,
                                                    "merging": 0.0,
                                                    "memory management": 0.0,
                                                    "eval": 0.0
                                        },
                                        "clustering": {
                                                        "predicted clusters": [],
                                                        "merged clusters": []
                                        },
                                        "metrics": {
                                                        "choosemaps": {},
                                                        "clustering choosemaps": {},
                                                        "frozen experts": {}

                                        }
                                    }


        #######################################################################################################
        ### Train a subnetwork for a given ID, unshifted task
        #######################################################################################################

        ### Changes which task classes are included in the train_dataloader
        self.train_sampler.set_task(task_id)

        ### Any method-specific setup
        self.online_before_task(task_id)

        running_train_loss, running_train_acc = 0., 0.
        for i, (images, labels, idx) in enumerate(self.train_dataloader):
            ### Break after the first 500 samples of training data if debugging
            if self.debug and (i + 1) * self.temp_batchsize >= 500:
                print("Temp batch size ", self.temp_batchsize, " images shape: ", images.shape)
                break
            samples_cnt += images.size(0) * self.world_size

            #!# Added passing batch number into step for debugging purposes, wont work with methods other than CMMoE
            loss, acc = self.online_step(images, labels, idx, i)
            running_train_loss += loss
            running_train_acc += acc

            self.report_training(samples_cnt, running_train_loss/(i+1), running_train_acc/(i+1), step="online")

        self.online_after_task(task_id)


        #######################################################################################################
        ### Get Off-task test accuracies for trained ID subnet
        #######################################################################################################

        ### Do offline testing on all previously seen tasks to assess forgetting
        test_accs = np.zeros((self.n_tasks))
        task_forgetting = np.zeros((self.n_tasks))

        for test_task in range(self.n_tasks):

            ### Changes which task classes are included in the train_dataloader
            self.test_sampler.set_task(test_task)

            task_acc = self.offline_evaluate(test_task, use_current_subnet=True, add_task=True)
            test_accs[test_task] = task_acc

        self.result_dicts[task_id]["accuracy"]["eval"] = test_accs

        torch.save(self.result_dicts, (self.log_dir + "/result_dicts.pt"))

        #######################################################################################################
        ### Get method-specific metrics for trained ID subnet on all tasks
        #######################################################################################################

        for test_task in range(self.n_tasks):
            self.ood_task_analysis()



        print("\n\n\n\n\n\n\n\n\n")
        for key in self.result_dicts.keys():
            print("\n", self.result_dicts[key])








    def run(self):
        ### Set up the datasets to be used for training
        self.setup_trainer()

        if self.experiment_type == "train":
            ### Train each task using specified method
            self.train()
            # pass
        elif self.experiment_type == "metric_calculation":
            ### Train each task as a unique subnetwork then calculate metrics and store them
            self.skip_merging = True
            self.skip_evals = True
            self.max_subnets = self.n_tasks
            self.train()
        elif self.experiment_type == "ood_transfer":
            ### Train a single subnetwork and evaluate transfer accuracy on each OoD task
            # self.measure_ood_transfer()
            pass
        elif self.experiment_type == "transfer_acc":
            ### Train each task as a unique subnetwork then calculate metrics and store them
            self.skip_merging = True            
            self.train(single_task=True)
        elif self.experiment_type == "ae_only":
            self.train_ae_only()











    ### Add new classes with contiguous labels for multipart datasets, or raw labels for split dataset
    def add_new_class(self, labels):
        # if self.label_type == "class":
        # print("Adding new class labels: ", torch.unique(labels))
        for label in labels:
            if label.item() not in self.exposed_classes:
                self.exposed_classes.append(label.item())


        self.exposed_classes.sort()
        
        ### Changed to allow multiple labels to map to the same class name for datasets with reoccuring classes under distribution shift
        self.exposed_classes_names = []
        for i in self.exposed_classes:
            if self.train_dataset.classes_names[i] not in self.exposed_classes_names:
                self.exposed_classes_names.append(self.train_dataset.classes_names[i])
        
        if self.use_memory == True:
        	if self.args.use_memory_class_names:
	            self.memory.add_new_class(cls_list=self.exposed_classes_names)
	        else:
	            #!# Will need to be updated for duplicate class labels to work
	            self.memory.add_new_class(cls_list=self.exposed_classes)
        # else:
        #     if self.use_memory == True:
           #      ### Need to implement class handling for caption datasets
           #      self.memory.add_new_class(cls_list=[self.task_id])

        #     pass
        # if 'reset' in self.sched_name:
        #     self.update_schedule(reset=True)


    def _interpret_pred(self, y, pred, verbose=False):
        # xlable is batch
        ret_num_data = torch.zeros(self.n_classes)
        ret_corrects = torch.zeros(self.n_classes)

        xlabel_cls, xlabel_cnt = y.unique(return_counts=True)
        if verbose:
            print(y[:10], " ", xlabel_cls, " ", xlabel_cnt)
        for cls_idx, cnt in zip(xlabel_cls, xlabel_cnt):
            ret_num_data[cls_idx] = cnt

        correct_xlabel = y.masked_select(y == pred)
        correct_cls, correct_cnt = correct_xlabel.unique(return_counts=True)
        for cls_idx, cnt in zip(correct_cls, correct_cnt):
            ret_corrects[cls_idx] = cnt

        return ret_num_data, ret_corrects







    ######################################################################################################
    ### Helper/util functions
    ######################################################################################################




    def convert_class_label(self, data_info):
        #* self.class_list => original class label
        self.class_list = self.train_dataset.classes
        for key in list(data_info.keys()):
            old_key = int(key[6:])
            data_info[self.class_list[old_key]] = data_info.pop(key)

        return data_info


    def reset_opt(self):
        self.optimizer = select_optimizer(self.opt_name, self.lr, self.model)
        self.scheduler = select_scheduler(self.sched_name, self.optimizer)





    ######################################################################################################
    ### Print functions
    ######################################################################################################



    def report_test(self, sample_num, avg_loss, avg_acc):
        logging.info(f"Test | Sample # {sample_num} | test_loss {avg_loss:.4f} | test_acc {avg_acc:.4f} | ")

    def report_training(self, sample_num, train_loss, train_acc, step=""):
        print(
            f"Train | Sample # {sample_num} | loss {train_loss:.4f} | running_acc {train_acc:.4f} | "
            f"lr {self.optimizer.param_groups[0]['lr']:.6f} | "
            f"Num_Classes {len(self.exposed_classes)} | "
            f"Num_Names {len(self.exposed_classes_names)} | "
            f"running_time {datetime.timedelta(seconds=int(time.time() - self.start_time))} | ", flush=True
            # f"ETA {datetime.timedelta(seconds=int((time.time() - self.start_time) * (self.total_samples-sample_num) / sample_num))}"
        )

        if step in self.result_dicts[self.task_id]["accuracy"].keys():
            self.result_dicts[self.task_id]["accuracy"][step] = train_acc

    def current_task_data(self, train_loader):
        data_info = {}
        for i, data in enumerate(train_loader):
            _, label = data

            for b in range(label.shape[0]):
                if 'Class_' + str(label[b].item()) in data_info.keys():
                    data_info['Class_' + str(label[b].item())] += 1
                else:
                    data_info['Class_' + str(label[b].item())] = 1

        logging.info("Current Task Data Info")
        logging.info(data_info)
        logging.info("<<Convert to str>>")
        convert_data_info = self.convert_class_label(data_info)
        logging.info(convert_data_info)


    def train_data_config(self, n_task, train_dataset, train_sampler):
        for t_i in range(n_task):
            train_sampler.set_task(t_i)
            train_dataloader = DataLoader(train_dataset,
                                          batch_size=self.batchsize,
                                          sampler=train_sampler,
                                          num_workers=self.n_worker)
            data_info = {}
            for i, data in enumerate(train_dataloader):
                _, label = data
                label = label.to(self.device)
                for b in range(len(label)):
                    if 'Class_' + str(label[b].item()) in data_info.keys():
                        data_info['Class_' + str(label[b].item())] += 1
                    else:
                        data_info['Class_' + str(label[b].item())] = 1
            logging.info(f"[Train] Task{t_i} Data Info")
            logging.info(data_info)

            convert_data_info = self.convert_class_label(data_info)
            np.save(os.path.join(self.log_dir,
                             f"seed_{self.rnd_seed}_task{t_i}_train_data.npy"),
                convert_data_info)
            logging.info(convert_data_info)

    def test_data_config(self, test_dataloader, task_id):
        data_info = {}
        for i, data in enumerate(test_dataloader):
            _, label = data
            label = label.to(self.device)

            for b in range(len(label)):
                if 'Class_' + str(label[b].item()) in data_info.keys():
                    data_info['Class_' + str(label[b].item())] += 1
                else:
                    data_info['Class_' + str(label[b].item())] = 1

        logging.info("<<Exposed Class>>")
        logging.info([(x, y)
            for x, y in zip(self.exposed_classes, self.exposed_classes_names)])

        logging.info(f"[Test] Task {task_id} Data Info")
        logging.info(data_info)
        logging.info("<<Convert>>")
        convert_data_info = self.convert_class_label(data_info)
        logging.info(convert_data_info)


    def print_memory(self, verbose=False):
        free, total = torch.cuda.mem_get_info(self.device)

        used = total - free
        # print("\nGPU memory report:")
        if verbose == True:
            print(f"Total: {total/1024**2:.2f} MB")
            print(f"Used: {used/1024**2:.2f} MB")
            print(f"reserved  = {torch.cuda.memory_reserved()/1024**2:.1f} MB")
            print(f"max alloc = {torch.cuda.max_memory_allocated()/1024**2:.1f} MB")            
        print(f"torch allocated = {torch.cuda.memory_allocated()/1024**2:.1f} MB")
        print(f"Free: {free/1024**2:.2f} MB")











    ### Implemented by child classes

    def online_step(self, sample, samples_cnt):
        raise NotImplementedError()

    def online_before_task(self, task_id):
        raise NotImplementedError()

    def online_after_task(self, task_id):
        raise NotImplementedError()

    def offline_evaluate(self, test_task):
        raise NotImplementedError()


    def offline_eval_all_tasks(self):
        raise NotImplementedError()


    def ood_task_analysis(self):
        raise NotImplementedError()


    def online_step_ae_only(self):
        pass

    def get_ae_loss(self, task_id, split="train"):
        pass
