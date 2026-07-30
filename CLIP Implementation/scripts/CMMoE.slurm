#!/bin/bash

#SBATCH --job-name=cmmoe         # Job name
#SBATCH --cpus-per-task=12               # Run on a 4 cores per node
#SBATCH --nodes=1                       # Run on a 1 node
#SBATCH --ntasks=1                      # One process (shared-memory)
#SBATCH --partition=gpu                 # Select partition
#SBATCH --mem=128gb                     # Job memory request
#SBATCH --time=200:00:00                # Time limit hrs:min:sec
#SBATCH --gres=gpu:a100:1                    # Request one GPU


#SBATCH --output=./logs/Train_CMMoE_%j.log   # Standard output and error log


echo "====================================================="
pwd; hostname; date
echo "====================================================="
# --nodelist=node-g105


echo "Running Adversarial CL Training"

module load apptainer


GPUS=0
NB_COMMA=`echo ${GPUS} | tr -cd , | wc -c`
NB_GPUS=$((${NB_COMMA} + 1))

NOTE="" # Short description of the experiment. (WARNING: logs/results with the same note will be overwritten!)

METHOD="CMMoE"
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

NUM_CLUSTERS=3
MAX_SUBNETS=5
# NUM_CLUSTERS=2
# MAX_SUBNETS=3

GPU_TRANSFORM="--gpu_transform"
USE_AMP="--use_amp"

# VIS_CLASS="batch"
VIS_CLASS="all"

# ZS_TEST="--zero_shot_evaluation"
ZS_TEST=""


# METRIC="energy"
# METRIC_ORDER="first"
# METRIC_MODAL="image"
# FIRST_LAYER="block_input"
# SECOND_LAYER="dispatcher_combined"


# METRIC="cos"
# METRIC_ORDER="second"
# FIRST_LAYER="dispatcher_combined"
# SECOND_LAYER="dispatcher_combined"

METRIC="dist"
METRIC_ORDER="first"
FIRST_LAYER="block_input"
SECOND_LAYER="dispatcher_combined"

# METRIC="dist"
# METRIC_ORDER="second"
# FIRST_LAYER="mlp_output"
# SECOND_LAYER="mlp_output"


echo "Checking If Else"


# if [ "$DATASET" == "cifar10" ]; then
ONLINE_ITER=1
FINETUNE_EPOCHS=1
MODEL_NAME="ViT-B/16" EVAL_PERIOD=1000
BATCHSIZE=32; OPT_NAME="adamw" SCHED_NAME="default"


# LR=5e-3
# LR=1e-3
LR=5e-4 
# LR=1e-4 
# LR=5e-5


# experts_per_subnet=6
experts_per_subnet=4
# experts_per_subnet=2


# MEM_SIZE=10000
# MEM_SIZE_PER_TASK=1000
MEM_SIZE_PER_TASK=4000
# MEM_SIZE_PER_TASK=3000
# MEM_SIZE_PER_TASK=2000
# MEM_SIZE_PER_TASK=1000

# MEM_PER_CLASS=0
# MEM_PER_CLASS=10
MEM_PER_CLASS=7

MEM_CLASS_NAMES=""
# MEM_CLASS_NAMES="--use_memory_class_names"


METRIC_MODAL="hybrid"
# METRIC_MODAL="text"
# METRIC_MODAL="image"

PEFT_ENCODER='both'
# PEFT_ENCODER='text'
# PEFT_ENCODER='image'


### Which layers to use adapters in for each encoder
# adapter_image=(1 1 1 1 1 1 1 1 1 1 1 1)  
adapter_image=(0 0 0 0 0 0 1 1 1 1 1 1)  
# adapter_image=(0 0 0 0 0 0 0 0 0 0 0 0)  


# adapter_text=(1 1 1 1 1 1 1 1 1 1 1 1)  
adapter_text=(0 0 0 0 0 0 1 1 1 1 1 1)  
# adapter_text=(0 0 0 0 0 0 0 0 0 0 0 0)  

    

echo "Variables Set"


for seed in 424
do
    note="${METHOD}_${VIS_CLASS}-${PEFT_ENCODER}_SEED${seed}"

    time srun apptainer run --nv ~/CMMoE.sif \
    python "main.py" --method $METHOD --dataset $DATASET $ZS_TEST \
    --n_tasks $N_TASKS --num_clusters $NUM_CLUSTERS --max_subnets $MAX_SUBNETS --finetune_epochs $FINETUNE_EPOCHS \
    --rnd_seed $seed --peft_encoder $PEFT_ENCODER \
    --model_name $MODEL_NAME --opt_name $OPT_NAME --sched_name $SCHED_NAME \
    --lr $LR --batchsize $BATCHSIZE --visible_classes $VIS_CLASS \
    --online_iter $ONLINE_ITER --data_dir ./data \
    --note $note --eval_period $EVAL_PERIOD --n_worker 18 \
    --metric $METRIC --metric_order $METRIC_ORDER --metric_modal $METRIC_MODAL \
    --first_layer $FIRST_LAYER --second_layer $SECOND_LAYER --experts_per_subnet $experts_per_subnet \
    --memory_per_class $MEM_PER_CLASS --memory_size $MEM_SIZE_PER_TASK \
    --adapter_blocks_image "${adapter_image[@]}" --adapter_blocks_text "${adapter_text[@]}" \
    $GPU_TRANSFORM $MEM_CLASS_NAMES \
    --per_task_datasets \
    --debug \
    --skip_evals





done



echo "====================================================="
date





