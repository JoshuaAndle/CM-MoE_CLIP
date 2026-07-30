import argparse
import torch

def base_parser():
    parser = argparse.ArgumentParser(description="Class Incremental Learning Research")

    # Method and Exp. Settings.
    parser.add_argument("--method", type=str, default="er", help="Select CIL method", )
    parser.add_argument("--dataset", type=str, default="cifar10", help="[mnist, cifar10, cifar100, imagenet]", )
    parser.add_argument("--n_tasks", type=int, help="The number of tasks")
    parser.add_argument("--n", type=int, default=100, help="The percentage of disjoint split. Disjoint=100, Blurry=0")
    parser.add_argument("--m", type=int, default=0, help="The percentage of blurry samples in blurry split. Uniform split=100, Disjoint=0")
    
    parser.add_argument("--rnd_NM", action='store_true', default=False, help="if True, N and M are randomly mixed over tasks.")
    parser.add_argument("--rnd_seed", type=int, help="Random seed number.")
    parser.add_argument("--memory_size", type=int, default=0, help="Episodic memory size")


    # Dataset
    parser.add_argument("--log_path", type=str, default="results", help="The path logs are saved.", )
    parser.add_argument("--per_task_datasets", action="store_true", help="True if each task is a separate dataset")


    # Model
    parser.add_argument("--model_name", type=str, default="resnet18", help="Model name")

    # Train
    parser.add_argument("--opt_name", type=str, default="sgd", help="Optimizer name")
    parser.add_argument("--sched_name", type=str, default="default", help="Scheduler name")
    parser.add_argument("--batchsize", type=int, default=16, help="batch size")

    parser.add_argument("--n_worker", type=int, default=1, help="The number of workers")

    parser.add_argument("--init_model", action="store_true", help="Initilize model parameters for every iterations", )
    parser.add_argument("--init_opt", action="store_true", help="Initilize optimizer states for every iterations", )

    parser.add_argument("--use_amp", action="store_true", help="Use automatic mixed precision.")

    parser.add_argument("--visible_classes", type=str, default="batch", help="Visible classes during training")

    # Transforms
    parser.add_argument("--transforms", nargs="*", default=['cutmix', 'autoaug'], help="Additional train transforms [cutmix, cutout, randaug]", )
    parser.add_argument("--gpu_transform", action="store_true", help="perform data transform on gpu (for faster AutoAug).")
    # parser.add_argument('--distribution_shift_type', type=str, default='none', choices=['none', 'blur', 'rotate'], help='Potential image shift to apply to subsequent tasks')

    # Regularization
    parser.add_argument("--reg_coef", type=int, default=100, help="weighting for the regularization loss term", )

    parser.add_argument("--data_dir", type=str, help="location of the dataset")

    # Note
    parser.add_argument("--note", type=str, help="Short description of the exp")

    # Eval period
    parser.add_argument("--eval_period", type=int, default=100, help="evaluation period for true online setup")

    parser.add_argument("--temp_batchsize", type=int, help="temporary batch size, for true online")
    parser.add_argument("--online_iter", type=float, default=1, help="number of model updates per samples seen.")




    # CLIP
    parser.add_argument('--peft_encoder', type=str, default='both', choices=['both', 'text', 'image', 'none'], help='The encoder to inject LoRa/Adapter/Prompt')
    parser.add_argument("--zero_shot_evaluation", action='store_true', default=False, help="if True, will do zero-shot evaluation.")
    parser.add_argument('--zero_shot_dataset', nargs='+', type=str, default=["food101", "caltech101", "eurosat", "flowers102", "oxford_pet"], 
                                help='Which dataset to use for zero-shot evaluation.')




    #################################################################################################################################
    ### Benchmark Args
    #################################################################################################################################


    # GDumb
    # parser.add_argument('--num_gpus', type=int, default=1, help='number of GPUs, for GDumb eval')
    # parser.add_argument('--workers_per_gpu', type=int, default=1, help='number of workers per GPU, for GDumb eval')

    # # CLIB
    # parser.add_argument("--imp_update_period", type=int, default=1, help=
    #     "period between importance update, in units of model updates (increase for heavy datasets like ImageNet)"
    # )
    # parser.add_argument('--lr_step', type=float, default=0.95, help='step of iterating lr for adaptive LR')
    # parser.add_argument('--lr_length', type=int, default=10, help='period of iterating lr for adaptive LR')
    # parser.add_argument('--lr_period', type=int, default=10, help='period of iterating lr for adaptive LR')

    # # RM & GDumb
    # parser.add_argument("--memory_epoch", type=int, default=256, help="number of training epochs after task for Rainbow Memory")

    # # BiC
    # parser.add_argument("--distilling", type=bool, default=True, help="use distillation for BiC.")

    # # AGEM
    # parser.add_argument('--agem_batch', type=int, default=240, help='A-GEM batch size for calculating gradient')

    # # MIR
    # parser.add_argument('--mir_cands', type=int, default=50, help='# candidates to use for MIR')

    # # Prompt-based (ViT)
    # # MVP
    # parser.add_argument('--use_mask', action='store_true', help='use mask for our method')
    # parser.add_argument('--use_contrastiv', action='store_true', help='use contrastive loss for our method')
    # parser.add_argument('--use_last_layer', action='store_true', help='use last layer for our method')
    # parser.add_argument('--use_afs', action='store_true', help='enable Adaptive Feature Scaling (AFS) in ours')
    # parser.add_argument('--use_gsf', action='store_true', help='enable Minor-Class Reinforcement (MCR) in ours')

    # parser.add_argument('--selection_size', type=int, default=1, help='# candidates to use for ViT_Prompt')
    # parser.add_argument('--alpha', type=float, default=0.5, help='# candidates to use for STR hyperparameter')
    # parser.add_argument('--gamma', type=float, default=2., help='# candidates to use for STR hyperparameter')
    # parser.add_argument('--margin', type=float, default=0.5, help='# candidates to use for STR hyperparameter')

    # parser.add_argument('--profile', action='store_true', help='enable profiling for ViT_Prompt')

























    # CMMoE


    # Cluster and Merge
    parser.add_argument("--max_subnets", type=int, default=6, help="Max subnets allowed before merging")
    ### Should eventually be moved into being a dataset object characteristic instead of separate arg
    parser.add_argument('--label_type', type=str, default='class', choices=['class', 'caption'], help='Whether dataset uses classes or image captions as labels')
    parser.add_argument("--finetune_epochs", type=int, default=1, help="Epochs of finetuning done using memory buffer")



    parser.add_argument('--metric_modality', type=str, default='hybrid', choices=['image', 'text', 'hybrid'], help='Which network branches to consider when clustering by metric.')
    parser.add_argument('--metric_order', type=str, default='second', choices=['first', 'second'], help='Whether to take a first or second-order metric.')
    parser.add_argument('--metric', type=str, default='dist', choices=['cos', 'corr', 'dist', 'energy'], help='Which metric to use.')
    parser.add_argument('--first_layer', type=str, default="dispatcher_combined", choices=['block_input', 'expert_inputs', 
                                                                                            'expert_down_acts', 'expert_up_acts',
                                                                                            'dispatcher_combined', 'mlp_output', 'logits_only'], 
                                                                                            help='Which layer to use for the first activation tensor.')
    parser.add_argument('--second_layer', type=str, default="dispatcher_combined", choices=['block_input', 'expert_inputs', 
                                                                                            'expert_down_acts', 'expert_up_acts',
                                                                                            'dispatcher_combined', 'mlp_output'], 
                                                                                            help='Which layer to use for the second activation tensor.')

    parser.add_argument('--num_clusters', type=int, default=2, help='Number of clusters to use for subnet merging')
    parser.add_argument('--num_blocks', type=int, default=4, help='Number of blocks to calculate metric over')
    parser.add_argument('--experts_per_subnet', type=int, default=4, help='Number of experts allocated to each subnetwork')

    parser.add_argument('--condition_by_class', action='store_true', default=False, help='Class-conditioning of activations prior to metric calculation')
    parser.add_argument('--remove_padding', action='store_true', default=False, help='Remove all zero-padded tokens from text inputs')
    parser.add_argument('--subsample_tokens', action='store_true', default=False, help='Take only the cls or eot tokens as aggregate information of all token relations')

    parser.add_argument('--block_set', type=str, default='last', choices=['first', 'last'], help='Which blocks to use.')

    parser.add_argument('--use_memory_class_names', action='store_true', default=False, help='Tracks memory per class names instead of by class id, for when multiple IDs map to the same name')
    parser.add_argument("--memory_per_class", type=int, default=0, help="Allocate memory per class instead of per task if >0")
    
    # parser.add_argument('--score_threshold', type=float, default=2.0, help='The ratio of KMeans score needed for a past subnetwork to be shared through clustering')




    parser.add_argument("--adapter_blocks_text", type=int, nargs='+',
                        default=[True] * 12,
                        # default=[False] * 9 + [True] * 3,
                        help="Which text encoder blocks to add adapters to")
    parser.add_argument("--adapter_blocks_image", type=int, nargs='+',
                        default=[True] * 12, 
                        help="Which image encoder blocks to add adapters to")


    parser.add_argument("--router_topk", type=int, default=2, help="How many experts the router selects for a given sample")



    # Debugging Flags
    parser.add_argument("--debug", action="store_true", help="Turn on Debug mode")
    parser.add_argument("--skip_evals", action="store_true", help="Whether to skip post-task evaluation steps to save runtime costs")
    parser.add_argument("--unknown_train_task_id", action="store_true", help="Whether task IDs need to be predicted at training time")
    parser.add_argument("--unknown_test_task_id", action="store_true", help="Whether task IDs need to be predicted at test time")
    parser.add_argument("--store_acts", action="store_true", help="Store activations during clustering")

    parser.add_argument('--experiment_type', type=str, default='train', choices=['train', 'ood_transfer', 
                                                                                'metric_calculation', 'transfer_acc', 'ae_only'], help='Which experiment to run.')
    parser.add_argument('--train_task', type=int, default=0, help='Which task to train on for ood_transfer experiments')




















    #######################################################################################################################
    ### Benchmarks - DIKI
    #######################################################################################################################


    parser.add_argument("--dataset_root", default = "", type=str)
    parser.add_argument("--model_backbone_name", default = "", type=str)
    parser.add_argument("--input_size", type=int, nargs=2, default=(-1, -1))
    parser.add_argument("--prompt_template", default = "", type=str)
    parser.add_argument("--scenario", default = "", type=str)
    # parser.add_argument("--dataset", default = "", type=str)
    parser.add_argument("--num_shots", default = -1, type=int)
    # parser.add_argument("--seed", default = -1, type=int)
    parser.add_argument("--use_validation", action = "store_true")
    parser.add_argument("--load_file", default = "", type=str)
    # parser.add_argument("--eval_only", action = "store_true")

    parser.add_argument("--train_one_dataset", default = -1, type=int)  # if >= 0", then only train corresponding dataset in MTIL
    parser.add_argument("--zero_shot", action = "store_true")
    parser.add_argument("--MTIL_order_2", action = "store_true")

    parser.add_argument("--prompt_depth_vision", default = 1, type=int)
    parser.add_argument("--prompt_depth_text", default = 1, type=int)
    parser.add_argument("--n_ctx_vision", default = 12, type=int)
    parser.add_argument("--n_ctx_text", default = 12, type=int)
    # parser.add_argument("--batch_size", default = 64, type=int)
    parser.add_argument("--name", default = "SGD", type=str, help="optimizer name")
    # parser.add_argument("--lr", default = 0.05, type=float)
    parser.add_argument("--max_epoch", default = 10, type=int)
    parser.add_argument("--weight_decay", default = 0, type=float)
    parser.add_argument("--lr_scheduler", default = "cosine", type=str)
    parser.add_argument("--warmup_epoch", default = 0, type=int)
    parser.add_argument("--batchwise_prompt", action = "store_true")




    #######################################################################################################################
    ### Benchmarks - ZSCL
    #######################################################################################################################






    #######################################################################################################################
    ### Benchmarks - DAC
    #######################################################################################################################


    # Dataset arguments
    parser.add_argument('--root_path', type=str, default='your_path')
    parser.add_argument('--shots', default=16, type=int)
    # Model arguments
    parser.add_argument('--backbone', default='ViT-B/16', type=str)
    # Training arguments
    #parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--n_iters', default=500, type=int)
    parser.add_argument('--linear', type=int, default=1, choices=[0, 1],
                    help='Set to 1 to enable linear layer processing, 0 to disable.')
    parser.add_argument('--adaw', type=int, default=1, choices=[0, 1],
                    help='Set to 1 to enable linear layer processing, 0 to disable.')
    parser.add_argument('--at', type=int, default=1, choices=[0, 1],
                    help='Set to 1 to enable linear layer processing, 0 to disable.')
    # LoRA arguments
    parser.add_argument('--position', type=str, default='all', choices=['bottom', 'mid', 'up', 'half-up', 'half-bottom', 'all', 'top3'], help='where to put the LoRA modules')
    parser.add_argument('--encoder', type=str, choices=['text', 'vision', 'both'], default='both')
    parser.add_argument('--params', metavar='N', type=str, nargs='+', default=['q','v'], help='list of attention matrices where putting a LoRA') 
    parser.add_argument('--r', default=16, type=int, help='the rank of the low-rank matrices')


    parser.add_argument('--k', default=8, type=int, help='the rank of the low-rank matrices')

    parser.add_argument('--dropout_rate', default=0.2, type=float, help='dropout rate applied before the LoRA module')
    
    parser.add_argument('--save_path', default='your_path', help='path to save the lora modules after training, not saved if None')
    parser.add_argument('--filename', default='lora_weights', help='file name to save the lora weights (.pt extension will be added)')
    parser.add_argument("--lora_paths", type=str, default=None)
    






    #######################################################################################################################
    ### Benchmarks - MoE-Adapters++
    #######################################################################################################################




    # hyper parameters
    parser.add_argument("--model", type=str, default="ViT-B/16")
    parser.add_argument("--val_preprocess_chooser", type=str, default=None)
    # parser.add_argument("--model_layer", type=str, default=23)
    # parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--batch_size_eval", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--lr_ae", type=float, default=1e-2, help="when train a task, the rate of data for router recording and all data")
    parser.add_argument("--wd", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--ls", type=float, default=0.0, help="Label smoothing.")
    parser.add_argument("--warmup_length", type=int, default=100)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--output_frozen_path", type=str, default=None)

    # logging setting
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--eval_interval", type=int, default=None)
    parser.add_argument("--loss_interval", type=int, default=1000)
    parser.add_argument("--eval_every_epoch", action="store_true")
    parser.add_argument('--eval_only', action='store_true', help='only evaluate the LoRA modules (save_path should not be None)')
    
    parser.add_argument("--eval_yjz", action="store_true")

    # exp setting
    parser.add_argument(
        "--train_method",
        type=str,
        default="finetune",
        choices=["finetune"], help="Method to use.",
    )
    parser.add_argument(
        "--train_mode",
        type=str,
        default="whole",
        choices=["whole", "lora", "text", "image", "image-fc", "image-fc-fixed", "fc", "adapter"], help="Train mode to use.",
    )
    parser.add_argument("--data_location", type=str, default="./data")
    parser.add_argument("--train_dataset", default=None)
    parser.add_argument("--eval_datasets", default=None, type=lambda x: x.split(","))
    parser.add_argument("--text_datasets", default=None, type=lambda x: x.split(","))
    parser.add_argument("--template", type=str, default=None)

    # save & load
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--load_federate", default=None, type=lambda x: x.split(","))
    parser.add_argument("--load_autochooser", type=str, default=None)  # the checkpoint of autochooser

    # model control for image-fc branch
    parser.add_argument("--fair", action="store_true")
    parser.add_argument("--we", action="store_true")
    parser.add_argument("--we_wise", action="store_true")
    parser.add_argument("--we_wise_alpha", type=float, default=0.98, help="wise_ft_alpha")
    parser.add_argument("--moving_avg", action="store_true")
    parser.add_argument("--avg_freq", type=int, default=100)
    parser.add_argument("--mv_avg_decay", type=float, default=0.999)
    parser.add_argument(
        "--mv_avg_model",
        type=str,
        default="n",
        choices=["n", "t", "zeroshot"], help="moving_avg_model to use.",
    )
    parser.add_argument("--l2", type=float, default=0)
    parser.add_argument(
        "--fc_init", action="store_true", help="Whether to reinitialize the model."
    )
    parser.add_argument(
        "--fc_setnone", action="store_true", help="Whether to shift the dataset."
    )
    parser.add_argument(
        "--dataset_shift", action="store_true", help="Whether to shift the dataset."
    )
    parser.add_argument("--n_class", type=int, default=10, help="Number of classes.")

    # ZSCL
    parser.add_argument(
        "--ref_wise_alpha", type=float, default=0.8, help="WiSE zeroshot reference"
    )
    parser.add_argument(
        "--ref_wise",
        default=False,
        action="store_true", help="WiSE zeroshot reference",
    )
    parser.add_argument(
        "--ref_dataset",
        default=None, help="For fine tuning or linear probe, which dataset to train on",
    )
    parser.add_argument("--ref_model", type=str, default=None)
    parser.add_argument(
        "--ref_sentences",
        default=None, help="For fine tuning or linear probe, which dataset's template and classname to train on",
    )
    parser.add_argument(
        "--T", type=float, default=2.0, help="Temperature for distillation loss"
    )
    parser.add_argument("--num", type=float, default=64)

    # # --------- #
    # # iCaRL
    # parser.add_argument("--dataset_order", default=None, type=lambda x: x.split(","))
    # parser.add_argument("--memory_size", type=int, default=10000)

    # --------- #
    # others
    parser.add_argument(
        "--weight_adjust",
        default=False,
        action="store_true", help="adjust",
    )
    parser.add_argument(
        "--feature_mse",
        default=False,
        action="store_true", help="feature_mse",
    )
    parser.add_argument(
        "--image_loss",
        default=False,
        action="store_true", help="image_loss",
    )
    parser.add_argument(
        "--text_loss",
        default=False,
        action="store_true", help="text_loss",
    )
    parser.add_argument(
        "--ablation_loss_2",
        default=False,
        action="store_true", help="ablation_loss_2",
    )

    parser.add_argument(
        "--wise_merge",
        default=False,
        action="store_true", help="Whether or not to use wise_merge (training)",
    )
    parser.add_argument(
        "--wise_ft",
        default=False,
        action="store_true", help="Whether or not to use wise_ft (evaluation)",
    )
    parser.add_argument(
        "--wise_ft_model",
        type=str,
        default="n",
        choices=["n", "zeroshot"], help="wise_ft_model to use.",
    )
    parser.add_argument("--wise_ft_alpha", type=float, default=0.8, help="wise_ft_alpha")

    parser.add_argument(
        "--exp_name",
        type=str,
        default=None, help="Name of the experiment, for organization purposes only.",
    )
    parser.add_argument(
        "--results_db",
        type=str,
        default="results.jsonl", help="Where to store the results, else does not store",
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None, help="Directory for caching features and encoder",
    )

    # Model freeze
    parser.add_argument(
        "--freeze_encoder",
        default=False,
        action="store_true", help="Whether or not to freeze the image encoder. Only relevant for fine-tuning.",
    )
    parser.add_argument(
        "--freeze_fc",
        type=int,
        default=0, help="Whether or not to freeze the fully connection layers. Only relevant for fine-tuning.",
    )

    # lwf
    parser.add_argument("--lwf", action="store_true", help="Whether to use LWF.")

    parser.add_argument(
        "--basic_model_load",
        type=lambda x: x.split(","),
        default=None, help="Optionally load _classifiers_, e.g. a zero shot classifier or probe or ensemble both.",
    )
    parser.add_argument(
        "--fc_load",
        type=lambda x: x.split(","),
        default=None, help="Optionally load _classifiers_, e.g. a zero shot classifier or probe or ensemble both.",
    )
    parser.add_argument(
        "--keep_old_heads",
        type=int,
        default=0, help="display distance between features after encoder",
    )

    # BASELINE
    parser.add_argument(
        "--baseline", action="store_true", help="Whether to use BASELINE."
    )

    # trio
    parser.add_argument("--trio", action="store_true", help="Whether to use TRIO.")
    parser.add_argument(
        "--control_dataset",
        default=None, help="For fine tuning or linear probe, which dataset to train on (against)",
    )
    parser.add_argument(
        "--control_dataset_add",
        default=None, help="For fine tuning or linear probe, which dataset to train on (against)",
    )
    parser.add_argument(
        "noise", action="store_true", help="Whether to use random noise to regularize."
    )
    parser.add_argument("--rff", action="store_true", help="Whether to use TRIO.")

    parser.add_argument("--alpha", default=0.5, type=float)
    parser.add_argument(
        "--fisher",
        type=lambda x: x.split(","),
        default=None, help="TODO",
    )
    parser.add_argument(
        "--fisher_floor",
        type=float,
        default=1e-8, help="TODO",
    )

    # Plusleft Fish
    # settings for original MoE-Adapters
    parser.add_argument("--ffn_num", type=int, default=64, help="the dim of lora")
    parser.add_argument("--ffn_adapt", action="store_true", help="Whether or not to use adapter")  # use adapter
    parser.add_argument("--ffn_option", type=str, default="parallel")
    parser.add_argument("--ffn_adapt_where", type=str,
        default="AdapterDoubleEncoder",
        choices=["AdapterImageEncoder", "AdapterDoubleEncoder"], help="Adapter is added into ImageEncoder or ImageEncoder_and_TextEncoder.",)  # not use
    parser.add_argument("--apply_moe", action="store_true", help="Whether or not to use moe")  # use moe
    parser.add_argument("--repeat_train", action="store_true", help="default is manual differentiation")
    parser.add_argument("--task_id", type=int, default=-1, help="task_id")
    parser.add_argument("--multi_experts", action="store_true", help="Whether or not to use multi_experts")
    parser.add_argument("--experts_num", type=int, default=2, help="the number of experts")
    parser.add_argument("--is_train", action="store_true", help="Whether or not to use router noise")
    parser.add_argument("--frozen", action="store_true", help="Whether or not to use adapter")
    parser.add_argument("--autorouter", action="store_true", help="Whether or not to use autorouter")
    parser.add_argument("--task_num", type=int, default=11, help="the max_num of task, the number can be dynamically changed")
    parser.add_argument("--threshold", type=float, help="threshold for zero-shot.")
    parser.add_argument("--non_text", action="store_true", help="with out text encoder")
    parser.add_argument("--frozen_path", type=str, default="frozen_list")
    parser.add_argument("--few_shot", type=int, default=-1, help="few_shot_num")  # n-shot
    parser.add_argument("--topk", type=int, default=1, help="set k when we want to set topk accuracy")
    parser.add_argument("--train_chooser", action="store_true", help="train autochooser.")

    # ZiChen Huang, settings for MoE-Adapter++: 
    # Expert Num, Flag, Treshold settings
    parser.add_argument("--dyn_moe", action="store_true", help="Whether or not to use DynMoE")
    parser.add_argument("--init_expert_num", type=int, default=2, help="when init the model, for the first task, the num of expert, set as top-k")
    parser.add_argument("--dyn_expert_num", type=int, default=22, help="when build the model, the num of expert")
    parser.add_argument('--text_expert_num_list', type=int, nargs='+', help='List of text expert flag in every layer')  # not use
    parser.add_argument('--image_expert_num_list', type=int, nargs='+', help='List of image expert flag in every layer')
    parser.add_argument("--max_expert_num", type=int, default=12, help="the max num of expert for all model")
    parser.add_argument("--router_recording_rate", type=float, default=0.1, help="when train a task, the rate of data for router recording and all data")
    parser.add_argument("--expansion_threshold_text", type=float, default=0.5, help="when training, the threshold to judge the self-expansion, i.e. Thres_e")  # not use
    parser.add_argument("--zero_shot_threshold_text", type=float, default=1/500.0, help="when training, the threshold to judge the self-expansion, i.e. Thres_zs")  # not use
    parser.add_argument("--expansion_threshold_image", type=float, default=0.4, help="when training, the threshold to judge the self-expansion, i.e. Thres_e")
    parser.add_argument("--zero_shot_threshold_image", type=float, default=1 / 0.4, help="when training, the threshold to judge the self-expansion, i.e. Thres_zs")

    # router & noise settings
    parser.add_argument("--use_dyn_moe_gate", action="store_true", help="Whether or not to use gate noise in DynMoE")
    parser.add_argument("--use_gate_noise", action="store_true", help="Whether or not to use gate noise in DynMoE")
    parser.add_argument("--gate_noise_epsilon", type=float, default=1e-2, help="when training, the gate noise epsilon of noisy gate")
    parser.add_argument("--gate_seed", type=int, default=1, help="the seed of adaptive moe gate")
    
    # Auto-Encoder setting for LEAS & DEeC
    parser.add_argument("--hidden_dims", type=int, default=64, help="the hidden_dims of Auto-Encoder")
    parser.add_argument("--text_AE_hidden_dims", type=int, default=128, help="the hidden_dims of text Auto-Encoder")  # not use
    parser.add_argument("--visual_AE_hidden_dims", type=int, default=16, help="the hidden_dims of visual Auto-Encoder")
    
    # loss setting
    parser.add_argument("--mse_weight", type=float, default=1.0, help="when training, the weight of mse_loss in Auto-Encoder")
    parser.add_argument("--ce_weight", type=float, default=10.0, help="when training, the weight of CE_loss, now adadonded")

    # eval settings
    parser.add_argument("--force_val_task_id", type=int, default=None, help="when eval, give a force task id")
    parser.add_argument("--use_LEAS_to_eval", action="store_true", help="Use LEAS to get eval_task_id")
    parser.add_argument('--force_expansion_list', type=int, nargs='+', default=[False] * 12, help='List of image expert numbers in every layer')
    parser.add_argument("--force_zero_shot", action="store_true", help="when eval, force use zero_shot")
    parser.add_argument("--force_layer_lock", action="store_true", help="when training, Prohibit shallower layers from expanding after the current layer has expanded")
    parser.add_argument("--mutil_LEAS_lock", action="store_true", help="when eval, Only the last layer of LEAS is allowed")
    parser.add_argument("--track_val_task_id", type=int, nargs='+', default=[True] * 12, help="when eval, tracking the eval task id")
    parser.add_argument("--track_val_discrepancy", type=int, nargs='+', default=[True] * 12, help="when eval, tracking the eval discrepancy in LEAS")
    parser.add_argument("--eval_acc_task_id", type=int, default=None, help="when eval, probability that eval_task_id is equal to this setting")
    parser.add_argument("--print_eval_batches", action="store_true", help="When eval, print or not print every batch's rd_sim or other info")
    parser.add_argument("--augmented_zero_shot", action="store_true", help="When eval, whether to use enhanced zs (currently abandoned)")

    # Multi-GPU settings  
    # TODO: Needs to be optimised, currently only runs on a single GPU
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    
    # log path setting
    parser.add_argument('--log_dir', default=None, type=str, help='the log path of tensorboard')
    
    # Dynamic MoE-Adapters setting
    parser.add_argument("--use_dyn_moe_layer_list_text", type=int, nargs='+',
                        default=[False] * 12,
                        # default=[False] * 9 + [True] * 3,
                        help="Deploy Dynmaic MoE-Adapters in current text layer")  # not use
    parser.add_argument("--use_dyn_moe_layer_list_visual", type=int, nargs='+',
                        default=[False] * 6 + [True] * 6, help="Deploy Dynmaic MoE-Adapters in current visual layer")
    parser.add_argument("--use_LEAS_list_text", type=int, nargs='+',
                        default=[False] * 12, help="when eval, use zero_shot")  # not use
    parser.add_argument("--use_LEAS_list_visual", type=int, nargs='+',
                        default=[False] * 12, help="when eval, use zero_shot")

    # LEAS setting
    parser.add_argument("--discrepancy_weighted_vector", type=str, default="Avg", 
                        help="Avg, Avg_norm, Z_score Z_score_norm")
    parser.add_argument("--single_router", action="store_true", help="When eval, print or not print every batch's rd_sim or other info")
    parser.add_argument("--without_LEAS", action="store_true", help="when eval for single router, don't use LEAS")
    parser.add_argument("--cut_off_rate_text", type=float, default=0.5, help="when training, the rate between sliding window and overall length, text")  # not use
    parser.add_argument("--cut_off_rate_visual", type=float, default=0.2, help="when training, the rate between sliding window and overall length, visual")
    
    # ablation setting: "Same" Input in paper
    parser.add_argument("--no_tree_strategy", action="store_true", help="ablation setting: Same Input in paper")



    parser.add_argument("--defer_setup", action="store_true", help="Defer certain shared trainer class setup steps to be handled by child class")













    args = parser.parse_args()

    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # set model layer:
    # if args.model == "ViT-L/14@336px":
    #     args.vision_layer = 24
    #     args.text_layer = 12
    # elif args.model == "Siglip-B-14-224":
    #     args.vision_layer = 12
    #     args.text_layer = 12
    # else:
    args.vision_layer = 12
    args.text_layer = 12

    assert (args.epochs is None or args.iterations is None), "Cannot specify both epoch and iterations."
    assert (args.eval_interval is None or not args.eval_every_epoch), "Cannot specify both eval_interval and eval_every_epoch."


    return args
