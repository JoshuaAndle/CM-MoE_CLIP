#!/bin/bash

#SBATCH --job-name=continualclip         # Job name
#SBATCH --cpus-per-task=12               # Run on a 4 cores per node
#SBATCH --nodes=1                       # Run on a 1 node
#SBATCH --ntasks=1                      # One process (shared-memory)
#SBATCH --partition=gpu                 # Select partition
#SBATCH --mem=128gb                     # Job memory request
#SBATCH --time=200:00:00                # Time limit hrs:min:sec
#SBATCH --gres=gpu:a100:1                    # Request one GPU
#SBATCH --nodelist=node-g105


#SBATCH --output=./logs/Train_continual_clip_%j.log   # Standard output and error log


echo "====================================================="
pwd; hostname; date
echo "====================================================="


echo "Running Adversarial CL Training"

module load apptainer


GPUS=0
NB_COMMA=`echo ${GPUS} | tr -cd , | wc -c`
NB_GPUS=$((${NB_COMMA} + 1))

NOTE="" # Short description of the experiment. (WARNING: logs/results with the same note will be overwritten!)

METHOD="continual_clip"
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
BATCHSIZE=32; LR=5e-4 OPT_NAME="adamw" SCHED_NAME="default"








echo "Variables Set"


for seed in 400
do
    note="${METHOD}_${VIS_CLASS}-${PEFT_ENCODER}_SEED${seed}"



    # time srun apptainer run --nv ~/pytorch-advcorr2.simg \
    # time srun apptainer run --nv ~/MoA_Container.sif --env INFO="${info}" --env CUDA_VISIBLE_DEVICES=${GPUS} \
    # python "main.py" --method $METHOD --dataset $DATASET $ZS_TEST \

    time srun apptainer run --nv ~/CMMoE.sif \
    python "main.py" --method $METHOD --dataset $DATASET $ZS_TEST \
    --n_tasks $N_TASKS --finetune_epochs $FINETUNE_EPOCHS \
    --rnd_seed $seed --peft_encoder $PEFT_ENCODER \
    --model_name $MODEL_NAME --opt_name $OPT_NAME --sched_name $SCHED_NAME \
    --lr $LR --batchsize $BATCHSIZE --visible_classes $VIS_CLASS \
    $GPU_TRANSFORM --online_iter $ONLINE_ITER --data_dir ./data \
    --note $note --eval_period $EVAL_PERIOD --n_worker 18 \
    --per_task_datasets \
    # --debug \
    # --skip_evals





done



echo "====================================================="
date





