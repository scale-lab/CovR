#!/bin/bash

# Request half an hour of runtime:
#SBATCH -t 96:00:00

# Ask for the GPU partition and 1 GPU
#SBATCH -p gpu-he --gres=gpu:1 -C a6000

#Default resources are 1 core with 2.8GB of memory, SBATCH -p gpu-he --gres=gpu:1 -C ampere.

# Use more memory (4GB) (CPU RAM):
#SBATCH --mem=50g

#SBATCH -n 4

# Set up the environment by loading modules

source ~/.bashrc

module load python/3.11.11-5e66

source /oscar/data/sreda/mabdelat/LLM4TB/unsloth-b200/bin/activate

module load synopsys-vcs

export LANGCHAIN_TRACING_V2=false
export LANGSMITH_TRACING=false 

MAX_SEQ_LENGTH=32000
MAX_NEW_TOKENS=10000


python core/baselines/self_reflection_loop/main.py \
    --enhancer-model qwen3-8b \
    --tb-model qwen3-8b \
    --checker-model qwen3-8b \
    --reflector-model qwen3-8b \
    --curator-model qwen3-8b \
    --fixer-model qwen3-8b \
    --dataset ./data/eval_dataset.jsonl \
    --output output-eval/baselines/self_reflection_loop/vllm/qwen3-8b_veval \
    --iters 5 \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --max_new_tokens ${MAX_NEW_TOKENS}

python core/baselines/self_reflection_loop/main.py \
    --enhancer-model qwen3-8b \
    --tb-model qwen3-8b \
    --checker-model qwen3-8b \
    --reflector-model qwen3-8b \
    --curator-model qwen3-8b \
    --dataset ./data/eval_cvdp.jsonl \
    --fixer-model qwen3-8b \
    --output output-eval/baselines/self_reflection_loop/vllm/qwen3-8b_cvdp \
    --iters 5 \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --max_new_tokens ${MAX_NEW_TOKENS}
