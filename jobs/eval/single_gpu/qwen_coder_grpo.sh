#!/bin/bash

# Request half an hour of runtime:
#SBATCH -t 72:00:00

# Ask for the GPU partition and 1 GPU
#SBATCH -p gpu-he --gres=gpu:1 -C h100

#Default resources are 1 core with 2.8GB of memory, SBATCH -p gpu-he --gres=gpu:1 -C ampere.

# Use more memory (4GB) (CPU RAM):
#SBATCH --mem=50g

#SBATCH -n 4

# Set up the environment by loading modules
source ~/.bashrc
module load python/3.11.0s-ixrhc3q

source /oscar/data/sreda/mabdelat/LLM4TB/venv_2/bin/activate

module load synopsys-vcs

# python core/train/evaluator.py --model llama3.1:8b --lora /oscar/data/sreda/mabdelat/LLM4TB/output/sft/llama3_r64_20251024_140431/meta-llama/Meta-Llama-3.1-8B-Instruct_finetuned --batch_size 2


python core/train/evaluator.py --model qwen2.5-coder  --lora /oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen_coder_grpo_format_reward_grpo/checkpoint-800 --batch_size 2 --run_name qwen_coder_grpo_checkpoint_100_format_reward --use_sloth
