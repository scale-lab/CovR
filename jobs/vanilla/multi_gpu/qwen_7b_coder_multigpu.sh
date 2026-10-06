#!/bin/bash

# Request half an hour of runtime:
#SBATCH -t 72:00:00

# Ask for the GPU partition and 2 GPUs
#SBATCH -p gpu-he --gres=gpu:2 -C h100

#Default resources are 1 core with 2.8GB of memory, SBATCH -p gpu-he --gres=gpu:1 -C ampere.

# Use more memory (4GB) (CPU RAM):
#SBATCH --mem=100g

#SBATCH -n 1
#SBATCH -c 32

# Set up the environment by loading modules
source ~/.bashrc
module load python/3.11.0s-ixrhc3q

source /oscar/data/sreda/mabdelat/LLM4TB/unsloth-b200/bin/activate

module load synopsys-vcs

cd /oscar/data/sreda/mabdelat/LLM4TB

export LANGCHAIN_TRACING_V2=false
export LANGSMITH_TRACING=false 

export MAX_SEQ_LENGTH=32000
export MAX_NEW_TOKENS=10000
export NUM_GPUS="${SLURM_GPUS_ON_NODE:-2}"
export NUM_CPUS="${SLURM_CPUS_PER_TASK:-32}"

VEVAL_BATCH_SIZE=4
CVDP_BATCH_SIZE=2

MODEL_NAME="qwen2.5-coder"

accelerate launch --num_processes "${NUM_GPUS}" --main_process_port 0 core/train/evaluator.py \
  --multi_gpu \
  --batch_size "${VEVAL_BATCH_SIZE}" \
  --max_seq_length "${MAX_SEQ_LENGTH}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --max_workers "${NUM_CPUS}" \
  --model "${MODEL_NAME}" \
  --run_name "${MODEL_NAME}_veval" \
  --output_dir "output-eval/base_models/multi_gpu/${MODEL_NAME}_veval" \
  --with_reasoning 

accelerate launch --num_processes "${NUM_GPUS}" --main_process_port 0 core/train/evaluator.py \
    --multi_gpu \
    --batch_size "${CVDP_BATCH_SIZE}" \
    --max_seq_length "${MAX_SEQ_LENGTH}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --max_workers "${NUM_CPUS}" \
    --model "${MODEL_NAME}" \
    --run_name "${MODEL_NAME}_cvdp" \
    --output_dir "output-eval/base_models/multi_gpu/${MODEL_NAME}_cvdp" \
    --eval /oscar/data/sreda/mnouh/ScaleLab/LLM4TB/data/eval_llm4cov.jsonl \
    --with_reasoning
