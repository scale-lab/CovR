#!/bin/bash

# Request half an hour of runtime:
#SBATCH -t 72:00:00

# Ask for the GPU partition and 1 GPU
#SBATCH -p gpu-he --gres=gpu:2 -C a6000

#Default resources are 1 core with 2.8GB of memory, SBATCH -p gpu-he --gres=gpu:1 -C ampere.

# Use more memory (4GB) (CPU RAM):
#SBATCH --mem=100g

#SBATCH -n 4

# Set up the environment by loading modules
source ~/.bashrc
module load python/3.11.0s-ixrhc3q

source /oscar/data/sreda/mabdelat/LLM4TB/venv_3/bin/activate

module load synopsys-vcs

# python core/train/evaluator.py --model qwen2.5-coder  --base --lora /oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen_coder_grpo_base_format_syntax_func_cov_bs_32_w_sft_small_sample_unsloth_grpo/checkpoint-400/ --batch_size 2 --run_name qwen_coder_grpo_multi_gpu_checkpoint_400 

# python core/train/evaluator.py --model qwen2.5-coder  --base --lora /oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen_coder_grpo_base_format_syntax_func_cov_bs_32_w_sft_small_sample_unsloth_grpo/checkpoint-300/ --batch_size 2 --run_name qwen_coder_grpo_multi_gpu_checkpoint_300 

python core/train/evaluator.py --model qwen2.5-coder  --base --lora /oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen_coder_grpo_base_format_syntax_func_cov_bs_32_w_sft_small_sample_unsloth_grpo/checkpoint-100/ --batch_size 2 --run_name qwen_coder_grpo_multi_gpu_checkpoint_100 
