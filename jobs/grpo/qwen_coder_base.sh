#!/bin/bash

# Request half an hour of runtime:
#SBATCH -t 72:00:00

# Ask for the GPU partition and 1 GPU
#SBATCH -p gpu-he --gres=gpu:1 -C h100

#Default resources are 1 core with 2.8GB of memory, SBATCH -p gpu-he --gres=gpu:1 -C ampere.

# Use more memory (4GB) (CPU RAM):
#SBATCH --mem=100g

#SBATCH -n 8

# Set up the environment by loading modules

source ~/.bashrc

source /oscar/data/sreda/mabdelat/LLM4TB/venv/bin/activate

module load synopsys-vcs

python core/train/grpo.py --base --config core/train/cfg/experiments/qwen_coder_r64_grpo.yaml --run_name qwen_coder_grpo_base_prompt 