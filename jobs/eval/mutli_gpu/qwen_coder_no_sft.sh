#!/bin/bash

# Request half an hour of runtime:
#SBATCH -t 72:00:00

# Ask for the GPU partition and 1 GPU
#SBATCH -p gpu-he --gres=gpu:2 -C a6000

#Default resources are 1 core with 2.8GB of memory, SBATCH -p gpu-he --gres=gpu:1 -C ampere.

# Use more memory (4GB) (CPU RAM):
#SBATCH --mem=50g

#SBATCH -n 4

# Set up the environment by loading modules
source ~/.bashrc
# module load python/3.11.0s-ixrhc3q

source /oscar/data/sreda/mabdelat/LLM4TB/venv_2/bin/activate

module load synopsys-vcs

accelerate launch --num_processes 2 --multi_gpu  core/train/evaluator.py --max_seq_length 3028 --model qwen2.5-coder --run_name qwen_coder_base_wo_sloth  --batch_size 2 --base --multi_gpu