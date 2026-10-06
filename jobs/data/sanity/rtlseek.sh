#!/bin/bash

# Request half an hour of runtime:
#SBATCH -t 48:00:00

# Ask for the GPU partition and 1 GPU
#SBATCH -p batch

#Default resources are 1 core with 2.8GB of memory, SBATCH -p gpu-he --gres=gpu:1 -C ampere.

# Use more memory (4GB) (CPU RAM):
#SBATCH --mem=25g

#SBATCH -n 4

# Set up the environment by loading modules
source ~/.bashrc
source /oscar/data/sreda/mabdelat/LLM4TB/venv/bin/activate

module load synopsys-vcs 

export LANGCHAIN_TRACING_V2=false
export LANGSMITH_TRACING=false 

python core/data/sanity_check.py --input_dir /oscar/data/sreda/mabdelat/LLM4TB/output/data_generation_rtlseek