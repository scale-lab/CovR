#!/bin/bash

# Request half an hour of runtime:
#SBATCH -t 96:00:00

# Ask for the GPU partition and 1 GPU
#SBATCH -p batch

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

export MAX_SEQ_LENGTH=32000
export MAX_NEW_TOKENS=10000

python core/baselines/self_reflection_loop/main.py \
    --enhancer-model gpt-4o-mini \
    --tb-model gpt-4o-mini \
    --checker-model gpt-4o-mini \
    --reflector-model gpt-4o-mini \
    --curator-model gpt-4o-mini \
    --fixer-model gpt-4o-mini \
    --dataset ./data/eval_dataset.jsonl \
    --iters 5 \
    --output ./output-eval/ablations/vllm/wo_curator_veval \
    --wo_curator \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --max_new_tokens ${MAX_NEW_TOKENS} 


python core/baselines/self_reflection_loop/main.py \
    --enhancer-model gpt-4o-mini \
    --tb-model gpt-4o-mini \
    --checker-model gpt-4o-mini \
    --reflector-model gpt-4o-mini \
    --curator-model gpt-4o-mini \
    --fixer-model gpt-4o-mini \
    --dataset ./data/eval_cvdp.jsonl \
    --iters 5 \
    --output ./output-eval/ablations/vllm/wo_curator_cvdp \
    --wo_curator \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --max_new_tokens ${MAX_NEW_TOKENS} 
