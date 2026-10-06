#!/bin/bash

# Request half an hour of runtime:
#SBATCH -t 72:00:00

# Ask for the GPU partition and 1 GPU
#SBATCH -p batch 

#Default resources are 1 core with 2.8GB of memory, SBATCH -p gpu-he --gres=gpu:1 -C ampere.

# Use more memory (4GB) (CPU RAM):
#SBATCH --mem=100g

#SBATCH -n 15

# Set up the environment by loading modules
source ~/.bashrc
source /oscar/data/sreda/mabdelat/LLM4TB/venv/bin/activate

module load synopsys-vcs 

# python core/train/evaluator.py \
#     --answers /oscar/data/sreda/mabdelat/LLM4TB/output-eval/baselines/correctbench/answers_veval_qwen.jsonl \
#     --output_dir output-eval/baselines/correctbench \
#     --model gpt4omini \
#     --max_workers 10

# python core/train/evaluator.py \
#     --answers /oscar/data/sreda/mabdelat/LLM4TB/output-eval/baselines/correctbench/answers_cvdp_qwen.jsonl \
#     --output_dir output-eval/baselines/correctbench_cvdp \
#     --model gpt4omini \
#     --max_workers 10


python core/train/evaluator.py \
    --answers /oscar/data/sreda/mabdelat/LLM4TB/output-eval/baselines/correctbench/answers_cvdp_qwen.jsonl \
    --output_dir output-eval/baselines/correctbench_cvdp \
    --model qwen_coder_7b \
    --max_workers 10