#!/bin/bash

# Request half an hour of runtime:
#SBATCH -t 96:00:00

# Ask for the GPU partition and 1 GPU
#SBATCH -p gpu-he --gres=gpu:1 -C h100

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

export MODEL_NAME="qwen2.5-coder"
export CFG="/oscar/data/sreda/mabdelat/LLM4TB/core/train/cfg/experiments/sft/${MODEL_NAME}/sft_r128_a128.yaml"

export RUN_NAME=${MODEL_NAME}_full_sft_r128_a128_target_module


# python core/train/sft.py \
    # --config ${CFG} \
    # --with_reasoning \
    # --max_workers ${NUM_CPUS} \
    # --use_vllm \
    # --run_name ${RUN_NAME} \
    # --output_dir ./output/sft/${RUN_NAME} \
    # --checkpoint /oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen2.5-coder_full_sft_r128_a128_target_module/qwen2.5-coder_full_sft_r128_a128_target_module_20260428_093940/Qwen/Qwen2.5-Coder-7B-Instruct_finetuned



export RUN_NAME=${MODEL_NAME}_full_sft_r128_a128_target_module

python core/train/evaluator.py \
    --with_reasoning \
    --model ${MODEL_NAME} \
    --max_workers ${NUM_CPUS} \
    --use_vllm \
    --run_name ${RUN_NAME} \
    --output_dir ./output/sft/${RUN_NAME} \
    --lora /oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen2.5-coder_full_sft_r128_a128_target_module/qwen2.5-coder_full_sft_r128_a128_target_module_20260428_093940/Qwen/Qwen2.5-Coder-7B-Instruct_finetuned \
    --eval ./data/eval_dataset.jsonl \
    --num_samples 5 

python core/train/evaluator.py \
    --with_reasoning \
    --model ${MODEL_NAME} \
    --max_workers ${NUM_CPUS} \
    --use_vllm \
    --run_name ${RUN_NAME} \
    --output_dir ./output/sft/${RUN_NAME} \
    --lora /oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen2.5-coder_full_sft_r128_a128_target_module/qwen2.5-coder_full_sft_r128_a128_target_module_20260428_093940/Qwen/Qwen2.5-Coder-7B-Instruct_finetuned \
    --eval ./data/eval_cvdp.jsonl \
    --num_samples 5 \
    --max_seq_length 32000 \
    --max_new_tokens 10000

