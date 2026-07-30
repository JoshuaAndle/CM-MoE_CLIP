#!/bin/bash

#SBATCH --job-name=DIKI                  # Job name
#SBATCH --cpus-per-task=12               # Run on a 4 cores per node
#SBATCH --nodes=1                       # Run on a 1 node
#SBATCH --ntasks=1                      # One process (shared-memory)
#SBATCH --partition=gpu                 # Select partition
#SBATCH --mem=128gb                     # Job memory request
#SBATCH --time=200:00:00                # Time limit hrs:min:sec
#SBATCH --gres=gpu:a100:1                    # Request one GPU
#SBATCH --nodelist=node-g105

#SBATCH --output=./logs/Train_DIKI_%j.log   # Standard output and error log


echo "====================================================="
pwd; hostname; date
echo "====================================================="


echo "Running Adversarial CL Training"

module load apptainer


GPUS=0
NB_COMMA=`echo ${GPUS} | tr -cd , | wc -c`
NB_GPUS=$((${NB_COMMA} + 1))

NOTE="" # Short description of the experiment. (WARNING: logs/results with the same note will be overwritten!)

METHOD="DIKI"


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





# lr=1e-2
# lr=20.0 
# lr=7.0 
# lr=5.0 # Value used in provided script
# lr=5e-1
lr=5e-2
# lr=500



echo "Variables Set"


for seed in 406
do
    note="${METHOD}_${VIS_CLASS}-${PEFT_ENCODER}_SEED${seed}"



    # time srun apptainer run --nv ~/pytorch-advcorr2.simg \
    # time srun apptainer run --nv ~/MoA_Container.sif --env INFO="${info}" --env CUDA_VISIBLE_DEVICES=${GPUS} \
    # python "main.py" --method $METHOD --dataset $DATASET $ZS_TEST \

    time srun apptainer run --nv ~/CMMoE.sif python "main.py" \
    --lr $lr \
    --prompt_depth_vision 8 \
    --prompt_depth_text 8 \
    --n_ctx_vision 8 \
    --n_ctx_text 8 \
    --dataset_root "./data/" \
    --model_backbone_name 'ViT-B/16' \
    --input_size 224 224 \
    --scenario "domain-task" \
    --dataset "MTIL" \
    --num_shots -1 \
    --seed $seed \
    --use_validation \
    --load_file "" \
    --train_one_dataset -1 \
    --batchwise_prompt \
    --name "SGD" \
    --max_epoch 1 \
    --weight_decay 0 \
    --lr_scheduler "cosine" \
    --warmup_epoch 0 \
    --method $METHOD --dataset $DATASET $ZS_TEST --n_tasks $N_TASKS --rnd_seed $seed \
    --model_name $MODEL_NAME --opt_name $OPT_NAME --sched_name $SCHED_NAME \
    --batchsize $BATCHSIZE --visible_classes $VIS_CLASS $GPU_TRANSFORM \
    --online_iter $ONLINE_ITER --data_dir ./data --note $note \
    --eval_period $EVAL_PERIOD --n_worker 18 \
    --per_task_datasets \
    # --debug \
    # --defer_setup \
    # --skip_evals

    # --save ckpt/exp_${exp_no}
    # --load ckpt/exp_${exp_no}/${dataset_pre}.pth



done



echo "====================================================="
date





