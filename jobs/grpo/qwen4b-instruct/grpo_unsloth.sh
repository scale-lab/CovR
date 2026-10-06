#!/bin/bash

# Request half an hour of runtime:
#SBATCH -t 72:00:00

# Ask for the GPU partition and 1 GPU
#SBATCH -p gpu-he --gres=gpu:1 -C b200

#Default resources are 1 core with 2.8GB of memory, SBATCH -p gpu-he --gres=gpu:1 -C ampere.

# Use more memory (4GB) (CPU RAM):
#SBATCH --mem=200g

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=32

# Set up the environment by loading modules

export PYTHONUTF8=1

source ~/.bashrc

module load python/3.11.11-5e66
module load cuda/12.9.0-cinr

source /oscar/data/sreda/mabdelat/LLM4TB/unsloth-b200/bin/activate

export FLASHINFER_WORKSPACE_BASE="/tmp/${USER}/flashinfer-${SLURM_JOB_ID}"
mkdir -p "$FLASHINFER_WORKSPACE_BASE"
export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
export CUDA_PATH="$CUDA_HOME"
export FLASHINFER_NVCC="$(which nvcc)"
export CUDA_LIB_PATH="${CUDA_HOME}/targets/x86_64-linux/lib"
echo "FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE}"
echo "CUDA_HOME=${CUDA_HOME}"
echo "FLASHINFER_NVCC=${FLASHINFER_NVCC}"

module load synopsys-vcs

# python3.11 core/train/grpo.py \
#     --config core/train/cfg/experiments/grpo/qwen4b_instruct/qwen4b_instruct_r512_grpo_unsloth.yaml \
#     --run_name qwen3_4b_grpo_unsloth_1000_dataset \
#     --sft /oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen4b_instruct_full_sft_r512_a512_full_data/qwen4b_instruct_full_sft_r512_a512_full_data_20260507_173012/Qwen/Qwen3-4B-Instruct-2507_finetuned \
#     --use_vllm \
#     --vllm_mode colocate \
#     --with_reasoning \
#     --max_workers 32 \
#     --loss_type grpo \
#     --use_sloth \
#     train.max_steps=1000 


python3.11 core/train/grpo.py \
    --config core/train/cfg/experiments/grpo/qwen4b_instruct/qwen4b_instruct_r512_grpo_unsloth.yaml \
    --run_name qwen3_4b_grpo_unsloth_1000_dataset \
    --sft /oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen4b_instruct_full_sft_r512_a512_full_data/qwen4b_instruct_full_sft_r512_a512_full_data_20260507_173012/Qwen/Qwen3-4B-Instruct-2507_finetuned \
    --use_vllm \
    --vllm_mode colocate \
    --with_reasoning \
    --max_workers 32 \
    --loss_type grpo \
    --use_sloth \
    train.max_steps=1000 

