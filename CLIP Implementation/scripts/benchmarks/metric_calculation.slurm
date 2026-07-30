#!/bin/bash

#SBATCH --job-name=OCL         # Job name
#SBATCH --cpus-per-task=1               # Run on a 4 cores per node
#SBATCH --nodes=1                       # Run on a 1 node
#SBATCH --ntasks=1                      # One process (shared-memory)
#SBATCH --partition=gpu                 # Select partition
#SBATCH --mem=128gb                     # Job memory request
#SBATCH --time=200:00:00                # Time limit hrs:min:sec
#SBATCH --gres=gpu:1                    # Request one GPU


#SBATCH --output=./logs/metric_calc%j.log   # Standard output and error log


echo "====================================================="
pwd; hostname; date
echo "====================================================="

module load apptainer


GPUS=0

NUM_TASKS=6
NUM_SUBNETS=6
NUM_BLOCKS=4

exposedclasses="batch"
# exposedclasses="all"

# "block_input", 'expert_inputs', 'expert_down_acts', 'expert_up_acts', 'dispatcher_combined', 'mlp_output'


METRIC_ORDER="first"
# METRIC_ORDER="second"

# FIRST_LAYER="block_input"
# SECOND_LAYER="dispatcher_combined"


# FIRST_LAYER="expert_inputs"
# SECOND_LAYER="expert_up_acts"


FIRST_LAYER="expert_down_acts"
SECOND_LAYER="expert_up_acts"


# FIRST_LAYER="block_input"
# FIRST_LAYER="dispatcher_combined"
# FIRST_LAYER="mlp_output"

# FIRST_LAYER="logits_only"



# LOADPATH="./results/cifar_mnist_2_class/6TASKS/moa-clipMETHOD/10SEED/moa-clip_all-both_SEED10/"

for seed in 18 19 20
# for seed in 18
do
	# for BRANCH in "image" "text"
	# for BRANCH in "image"
	for BRANCH in "text"
	do
		for METRIC in "corr"
		# for METRIC in "dist"
		# for METRIC in "cos"
		# for METRIC in "dist" "cos"
		# for METRIC in "energy"
		do
			# for block_set in "first" "last"
			# for block_set in "first"
			for block_set in "last"
			do
			    LOADPATH="./results/CFP-Rot/6TASKS/moa-clipMETHOD/${seed}SEED/moa-clip_${exposedclasses}-both_SEED${seed}/"
			    # LOADPATH="./results/CFP-Rot/6TASKS/moa-clipMETHOD/${seed}SEED/moa-clip_all-both_SEED${seed}/"
			    echo $LOADPATH

			    time srun apptainer run --nv ~/MoA_Container.sif \
			    python "metric_calculation.py" --metric $METRIC --metric_order $METRIC_ORDER \
			    --num_tasks $NUM_TASKS --num_subnets $NUM_SUBNETS --num_blocks $NUM_BLOCKS \
			    --first_layer $FIRST_LAYER --second_layer $SECOND_LAYER --branch $BRANCH --block_set=$block_set \
			    --loadpath $LOADPATH \
			    --remove_padding \
			    # --subsample_tokens \


			    # --condition_by_class \

			    # --fold_tokens \
			    # --fold_sample \

			    # --fold_embed \


			done
		done
    done
done



echo "====================================================="
date





