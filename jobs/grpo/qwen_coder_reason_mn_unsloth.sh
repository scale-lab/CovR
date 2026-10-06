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

export PYTHONUTF8=1

source ~/.bashrc

module load python/3.11.11-5e66

source /oscar/data/sreda/mabdelat/LLM4TB/venv_3/bin/activate

module load synopsys-vcs

# python3.11 core/train/grpo.py --base --config core/train/cfg/experiments/qwen_coder_r64_grpo.yaml --run_name qwen_coder_grpo_reason_unsloth_lower_lr  --use_sloth --sft /oscar/data/sreda/mnouh/ScaleLab/LLM4TB/output/sft/qwen_coder_reasoning_20251117_210504/Qwen/Qwen2.5-Coder-7B-Instruct_finetuned/

python3.11 core/train/grpo.py --base --config core/train/cfg/experiments/qwen_coder_r64_grpo.yaml --run_name qwen_coder_grpo_reason_unsloth_lower_lr  --use_sloth --sft /oscar/data/sreda/mnouh/ScaleLab/LLM4TB/output/sft/qwen_coder_reasoning_20251117_210504/Qwen/Qwen2.5-Coder-7B-Instruct_finetuned/ --checkpoint /oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen_coder_grpo_reason_unsloth_lower_lr_grpo/checkpoint-400 --wandb_id s0tzyaut