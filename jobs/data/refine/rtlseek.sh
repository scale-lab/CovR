#!/bin/bash

# Request half an hour of runtime:
#SBATCH -t 48:00:00

# Ask for the GPU partition and 1 GPU
#SBATCH -p batch

#Default resources are 1 core with 2.8GB of memory, SBATCH -p gpu-he --gres=gpu:1 -C ampere.

# Use more memory (4GB) (CPU RAM):
#SBATCH --mem=50g

#SBATCH -n 4

# Set up the environment by loading modules
source ~/.bashrc
source /oscar/data/sreda/mabdelat/LLM4TB/venv/bin/activate

module load synopsys-vcs

export LANGCHAIN_TRACING_V2=false
export LANGSMITH_TRACING=false 

python core/data_generation/main.py --run_not_met --output  output/data_generation_rtlseek_processed --model gemini-2.5-flash

# python core/data_generation/self_checking/main.py --dataset output/data_generation_rtlseek  --output output/data_generation_rtlseek_processed 
# --model gpt-5-mini 
# --start_idx 10411
#--model deepseek-chat
# --model gemini-2.5-flash