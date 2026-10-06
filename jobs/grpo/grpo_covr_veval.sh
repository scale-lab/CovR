#!/bin/bash

# Request half an hour of runtime:
#SBATCH -t 96:00:00

# Ask for the GPU partition and 1 GPU
#SBATCH -p gpu-he --gres=gpu:1 -C h100

#Default resources are 1 core with 2.8GB of memory, SBATCH -p gpu-he --gres=gpu:1 -C ampere.

# Use more memory (4GB) (CPU RAM):
#SBATCH --mem=100g

#SBATCH -n 8

# Set up the environment by loading modules

source ~/.bashrc

module load python/3.11.11-5e66

source /oscar/data/sreda/mabdelat/LLM4TB/venv_3/bin/activate

module load synopsys-vcs

export LANGCHAIN_TRACING_V2=false
export LANGSMITH_TRACING=false 

python core/train/evaluator.py --model qwen2.5-coder --lora /oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen_coder_grpo_base_format_syntax_func_cov_bs_32_w_sft_small_sample_unsloth_grpo/checkpoint-400/  --max_seq_length 6028 --batch_size 2 