#!/bin/bash

#SBATCH --job-name=moaclip         # Job name
#SBATCH --cpus-per-task=12               # Run on a 4 cores per node
#SBATCH --nodes=1                       # Run on a 1 node
#SBATCH --ntasks=1                      # One process (shared-memory)
#SBATCH --partition=gpu                 # Select partition
#SBATCH --mem=128gb                     # Job memory request
#SBATCH --time=200:00:00                # Time limit hrs:min:sec
#SBATCH --gres=gpu:a100:1                    # Request one GPU


#SBATCH --output=./logs/Train_moe_adapters_pp_%j.log   # Standard output and error log


echo "====================================================="
pwd; hostname; date
echo "====================================================="


echo "Running Adversarial CL Training"

module load apptainer


GPUS=0
NB_COMMA=`echo ${GPUS} | tr -cd , | wc -c`
NB_GPUS=$((${NB_COMMA} + 1))

NOTE="" # Short description of the experiment. (WARNING: logs/results with the same note will be overwritten!)

METHOD="moe_adapters_pp"
PEFT_ENCODER='both' # both, text, image
# DATASET="cifar_mnist_2_class"
# DATASET="cifar10" # cifar10, cifar100, tinyimagenet, imagenet


# DATASET=FCP-Rot-Full
# DATASET=FCP-Rot-Seq-Full
# DATASET=FCP-Blur-Full
DATASET="FCP-Rot-Seq"
# DATASET="CPF-Rot-Seq"
# DATASET="PFC-Rot-Seq"
# DATASET="FCP-Blur-Seq"
# DATASET="CPF-Blur-Seq"
# DATASET="PFC-Blur-Seq"

N_TASKS=7
# N_TASKS=15


GPU_TRANSFORM="--gpu_transform"
USE_AMP="--use_amp"

# VIS_CLASS="batch"
VIS_CLASS="all"

# ZS_TEST="--zero_shot_evaluation"
ZS_TEST=""


echo "Checking If Else"


# if [ "$DATASET" == "cifar10" ]; then
ONLINE_ITER=1
FINETUNE_EPOCHS=1
MODEL_NAME="ViT-B/16" EVAL_PERIOD=1000
BATCHSIZE=32; OPT_NAME="adamw" SCHED_NAME="default"





# lr_ce=(1e-2 1e-3 1e-2 5e-3 1e-3 1e-3 1e-3 1e-4 1e-3 2e-3 5e-4)
# lr_ae=(1e-2 2e-3 5e-3 5e-5 1e-3 1e-4 2e-3 2e-4 5e-3 5e-3 5e-3)
# lr_ce=1e-2
# lr_ce=5e-3
# lr_ce=1e-3
lr_ce=1e-4



# lr_ae=1e-2
# lr_ae=5e-3
# lr_ae=1e-3
lr_ae=1e-4


list_v=(0 0 0 0 0 0 1 1 1 1 1 1)  # layers of image encoder, first layer of "1" is the recognition layer
list_t=(0 0 0 0 0 0 0 0 0 0 0 0)  # deploy in text encoder is not very well
list_noly_LEAS=(0 0 0 0 0 0 0 0 0 0 0 0)  # mutil-LEAS

### Dictating which layers force expansion (vision recognition only)
list_v_force=(0 0 0 0 0 0 1 0 0 0 0 0)

# TODO: zero-shot threshold for LEAS
threshold=26e-1
# threshold=35e-1

# Weighted vector of the discrepancy function 
# including "CON", "CON_norm", "STD", "STD_norm"
# in paper, "CON" is the best
discrepancy_weighted_vector=CON

# expansion threshold for DEeC
exp_threshold_image=1.1
exp_threshold_text=0.5  # if deploy MoE-Adapters++ in text encoder







echo "Variables Set"


for seed in 407
do
    note="${METHOD}_${VIS_CLASS}-${PEFT_ENCODER}_SEED${seed}"



    # time srun apptainer run --nv ~/pytorch-advcorr2.simg \
    # time srun apptainer run --nv ~/MoA_Container.sif --env INFO="${info}" --env CUDA_VISIBLE_DEVICES=${GPUS} \
    # python "main.py" --method $METHOD --dataset $DATASET $ZS_TEST \

    time srun apptainer run --nv ~/CMMoE.sif python "main.py" \
    --train_mode="adapter" \
    --lr=$lr_ce \
    --lr_ae=$lr_ae \
    --ffn_adapt_where "AdapterDoubleEncoder" \
    --expansion_threshold_image=${exp_threshold_image} \
    --expansion_threshold_text=${exp_threshold_text} \
    --visual_AE_hidden_dims=32 \
    --zero_shot_threshold_image=$threshold \
    --use_dyn_moe_layer_list_visual "${list_v[@]}" \
    --use_LEAS_list_visual "${list_noly_LEAS[@]}" \
    --discrepancy_weighted_vector=${discrepancy_weighted_vector} \
    --use_LEAS_to_eval \
    --ffn_adapt \
    --multi_experts \
    --apply_moe \
    --dyn_moe \
    --train_method "finetune" \
    --force_expansion_list "${list_v_force[@]}" \
    --use_gate_noise \
    --is_train \
    --ls 0.2 \
    --method $METHOD --dataset $DATASET $ZS_TEST --n_tasks $N_TASKS \
    --rnd_seed $seed --peft_encoder $PEFT_ENCODER \
    --model_name $MODEL_NAME --opt_name $OPT_NAME --sched_name $SCHED_NAME \
    --batchsize $BATCHSIZE --visible_classes $VIS_CLASS \
    $GPU_TRANSFORM --online_iter $ONLINE_ITER --data_dir ./data \
    --note $note --eval_period $EVAL_PERIOD --n_worker 18 \
    --per_task_datasets \
    --defer_setup \
    # --debug \
    # --skip_evals




done



echo "====================================================="
date





