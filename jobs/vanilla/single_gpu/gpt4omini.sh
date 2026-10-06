#!/bin/bash

# Request half an hour of runtime:
#SBATCH -t 48:00:00

# Ask for the GPU partition and 1 GPU
#SBATCH -p batch

#Default resources are 1 core with 2.8GB of memory, SBATCH -p gpu-he --gres=gpu:1 -C ampere.

# Use more memory (4GB) (CPU RAM):
#SBATCH --mem=50g

#SBATCH -n 15

# Set up the environment by loading modules
source ~/.bashrc
source /oscar/data/sreda/mabdelat/LLM4TB/venv/bin/activate

module load synopsys-vcs 

export LANGCHAIN_TRACING_V2=false
export LANGSMITH_TRACING=false 

# python core/train/evaluator.py \
#     --output_dir output-eval/base_models/gpt4omini_veval \
#     --model gpt-4o-mini --max_workers 10 --run_name gpt4omini_veval \
#     --max_seq_length 10028 --batch_size 1 \
#     --with_reasoning \

python core/train/evaluator.py \
    --output_dir output-eval/base_models/gpt4omini_cvdp \
    --model gpt-4o-mini --max_workers 10 --run_name gpt4omini_cvdp \
    --max_seq_length 10028 --batch_size 1 \
    --eval ./data/eval_cvdp.jsonl \
    --with_reasoning 