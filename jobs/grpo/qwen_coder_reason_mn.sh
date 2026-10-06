#!/bin/bash

# Request half an hour of runtime:
#SBATCH -t 72:00:00

# Ask for the GPU partition and 1 GPU
#SBATCH -p gpu-he --gres=gpu:1 -C h100

#Default resources are 1 core with 2.8GB of memory, SBATCH -p gpu-he --gres=gpu:1 -C ampere.

# Use more memory (4GB) (CPU RAM):
#SBATCH --mem=100g

#SBATCH -n 4

# Set up the environment by loading modules

export PYTHONUTF8=1

source ~/.bashrc

module load python/3.11.11-5e66

source /oscar/data/sreda/mabdelat/LLM4TB/venv_3/bin/activate

module load synopsys-vcs

# python3.11 core/train/grpo.py --base --config core/train/cfg/experiments/qwen_coder_r64_grpo_hf.yaml --run_name qwen_coder_grpo_reason_hf_lower_2e_lr_high_max_seq --sft /oscar/data/sreda/mnouh/ScaleLab/LLM4TB/output/sft/qwen_coder_reasoning_20251117_210504/Qwen/Qwen2.5-Coder-7B-Instruct_finetuned/

# python3.11 core/train/grpo.py --base --config core/train/cfg/experiments/qwen_coder_r64_grpo_hf.yaml --run_name qwen_coder_grpo_reason_hf_lower_2e_lr_high_max_seq --sft /oscar/data/sreda/mnouh/ScaleLab/LLM4TB/output/sft/qwen_coder_reasoning_20251117_210504/Qwen/Qwen2.5-Coder-7B-Instruct_finetuned/ --checkpoint /oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen_coder_grpo_reason_hf_lower_lr_high_max_seq_grpo/checkpoint-500 --wandb_id wfzhsim1

python3.11 core/train/grpo.py --base --config core/train/cfg/experiments/qwen_coder_r64_grpo_hf.yaml --run_name qwen_coder_grpo_reason_hf_lower_lr_grpo --sft /oscar/data/sreda/mnouh/ScaleLab/LLM4TB/output/sft/qwen_coder_reasoning_20251117_210504/Qwen/Qwen2.5-Coder-7B-Instruct_finetuned/ --checkpoint /oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen_coder_grpo_reason_hf_lower_lr_grpo_grpo/checkpoint-600 --wandb_id tcscwkh6
