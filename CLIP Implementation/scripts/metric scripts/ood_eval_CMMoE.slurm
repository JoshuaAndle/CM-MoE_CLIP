#!/bin/bash

#SBATCH --job-name=OCL         # Job name
#SBATCH --cpus-per-task=1               # Run on a 4 cores per node
#SBATCH --nodes=1                       # Run on a 1 node
#SBATCH --ntasks=1                      # One process (shared-memory)
#SBATCH --partition=gpu                 # Select partition
#SBATCH --mem=128gb                     # Job memory request
#SBATCH --time=200:00:00                # Time limit hrs:min:sec
#SBATCH --gres=gpu:a100:1                    # Request one GPU


#SBATCH --output=./logs/ood_eval_%j.log   # Standard output and error log


echo "====================================================="
pwd; hostname; date
echo "====================================================="


echo "Running Adversarial CL Training"

module load apptainer


GPUS=0
NB_COMMA=`echo ${GPUS} | tr -cd , | wc -c`
NB_GPUS=$((${NB_COMMA} + 1))

NOTE="" # Short description of the experiment. (WARNING: logs/results with the same note will be overwritten!)

METHOD="moa-clip"
PEFT_ENCODER='both' # both, text, image
DATASET="CFP-Rot-Seq"

N_TASKS=15

GPU_TRANSFORM="--gpu_transform"
USE_AMP="--use_amp"

VIS_CLASS="batch"
# VIS_CLASS="all"

ZS_TEST=""

MEM_SIZE_PER_TASK=2000
# MEM_SIZE=10000

TRAIN_TASK=0

METRIC="cos"
METRIC_ORDER="second"
METRIC_MODAL="hybrid"
FIRST_LAYER="block_input"
SECOND_LAYER="dispatcher_combined"

# METRIC="dist"
# METRIC_ORDER="second"
# METRIC_MODAL="hybrid"
# FIRST_LAYER="block_input"
# SECOND_LAYER="dispatcher_combined"

# METRIC="corr"
# METRIC_ORDER="first"
# METRIC_MODAL="image"
# FIRST_LAYER="expert_down_acts"
# SECOND_LAYER="expert_up_acts"

# METRIC="energy"
# METRIC_ORDER="first"
# METRIC_MODAL="image"
# FIRST_LAYER="block_input"
# SECOND_LAYER="dispatcher_combined"


echo "Checking If Else"


ONLINE_ITER=1
MODEL_NAME="ViT-B/16" EVAL_PERIOD=1000
BATCHSIZE=64; LR=5e-4 OPT_NAME="adamw" SCHED_NAME="default"





echo "Variables Set"


for seed in 0
do
    note="${METHOD}_${VIS_CLASS}-${PEFT_ENCODER}_SEED${seed}"

    # time srun apptainer run --nv ~/MoA_Container.sif \
    # python "main.py" --method $METHOD --dataset $DATASET $ZS_TEST \
    # --n_tasks $N_TASKS \
    # --rnd_seed $seed --peft_encoder $PEFT_ENCODER \
    # --model_name $MODEL_NAME --opt_name $OPT_NAME --sched_name $SCHED_NAME \
    # --lr $LR --batchsize $BATCHSIZE --visible_classes $VIS_CLASS \
    # --memory_size $MEM_SIZE $GPU_TRANSFORM --online_iter $ONLINE_ITER --data_dir ./data \
    # --note $note --eval_period $EVAL_PERIOD --n_worker 1 --num_gpus ${NB_GPUS} --debug




    # time srun apptainer run --nv ~/pytorch-advcorr2.simg \
    # time srun apptainer run --nv ~/MoA_Container.sif --env INFO="${info}" --env CUDA_VISIBLE_DEVICES=${GPUS} \
    # python "main.py" --method $METHOD --dataset $DATASET $ZS_TEST \

    time srun apptainer run --nv ~/MoA_Container.sif \
    python "main.py" --method $METHOD --dataset $DATASET $ZS_TEST \
    --n_tasks $N_TASKS \
    --rnd_seed $seed --peft_encoder $PEFT_ENCODER \
    --model_name $MODEL_NAME --opt_name $OPT_NAME --sched_name $SCHED_NAME \
    --lr $LR --batchsize $BATCHSIZE --visible_classes $VIS_CLASS \
    --memory_size $MEM_SIZE_PER_TASK $GPU_TRANSFORM --online_iter $ONLINE_ITER --data_dir ./data \
    --note $note --eval_period $EVAL_PERIOD --n_worker 1 --num_gpus ${NB_GPUS} \
    --metric $METRIC --metric_order $METRIC_ORDER --metric_modal $METRIC_MODAL \
    --first_layer $FIRST_LAYER --second_layer $SECOND_LAYER --train_task $TRAIN_TASK --experiment_type "ood_transfer" \
    --per_task_datasets \
    --debug





done



echo "====================================================="
date





