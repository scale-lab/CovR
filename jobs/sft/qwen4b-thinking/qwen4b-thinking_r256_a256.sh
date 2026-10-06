#!/bin/bash

# Request half an hour of runtime:
#SBATCH -t 96:00:00

# Ask for the GPU partition and 1 GPU
#SBATCH -p gpu-he --gres=gpu:1 -C b200

#Default resources are 1 core with 2.8GB of memory, SBATCH -p gpu-he --gres=gpu:1 -C ampere.

# Use more memory (4GB) (CPU RAM):
#SBATCH --mem=100g

#SBATCH -n 10

# Set up the environment by loading modules

source ~/.bashrc

module load python/3.11.11-5e66
module load cuda

export FLASHINFER_WORKSPACE_BASE="/tmp/${USER}/flashinfer-${SLURM_JOB_ID}"
mkdir -p "$FLASHINFER_WORKSPACE_BASE"

source /oscar/data/sreda/mabdelat/LLM4TB/trl-vllm-clean-env/bin/activate

module load synopsys-vcs

export LANGCHAIN_TRACING_V2=false
export LANGSMITH_TRACING=false 

export MAX_SEQ_LENGTH=32000
export MAX_NEW_TOKENS=10000
export NUM_GPUS="${SLURM_GPUS_ON_NODE:-2}"
export NUM_CPUS="${SLURM_CPUS_PER_TASK:-32}"

export VEVAL_BATCH_SIZE=4
export CVDP_BATCH_SIZE=2

export MODEL_NAME="qwen4b-thinking"
export CFG="/oscar/data/sreda/mabdelat/LLM4TB/core/train/cfg/experiments/sft/${MODEL_NAME}/sft_r256_a256.yaml"

export RUN_NAME=${MODEL_NAME}_full_sft_r256_a256_completion_loss_only

python core/train/sft.py \
    --config ${CFG} \
    --with_reasoning \
    --max_workers ${NUM_CPUS} \
    --use_vllm \
    --run_name ${RUN_NAME} \
    --output_dir ./output/sft/${RUN_NAME} 