#!/bin/bash

# Request half an hour of runtime:
#SBATCH -t 72:00:00

# Ask for the GPU partition and 1 GPU
#SBATCH -p gpu-he --gres=gpu:1 -C v100

#Default resources are 1 core with 2.8GB of memory, SBATCH -p gpu-he --gres=gpu:1 -C ampere.

# Use more memory (4GB) (CPU RAM):
#SBATCH --mem=50g

#SBATCH -n 4

# Set up the environment by loading modules
source ~/.bashrc
source /oscar/data/sreda/mabdelat/LLM4TB/venv/bin/activate

module load synopsys

python core/data_generation/main.py --dataset verithoughts --output output/data_generation_verithoughts 
# --model gemini-2.5-flash 
# --model gpt-5-mini
# python core/data_generation/main.py --dataset verithoughts --output output/data_generation_verithoughts --model gpt-5-mini
#  --model deepseek-chat
# --model gemini-2.5-flash