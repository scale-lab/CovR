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

export PYTHONUTF8=1

source ~/.bashrc
module load python/3.11.0s-ixrhc3q

source /oscar/data/sreda/mabdelat/LLM4TB/venv_2/bin/activate

module load synopsys
# python core/train/grpo.py --config core/train/cfg/experiments/llama3.3_r64_grpo.yaml --use_sloth --run_name unsloth_llamar64_grpo_multi_gpu --sft /oscar/data/sreda/mabdelat/LLM4TB/output/sft/llama3_r64_20251105_085246/checkpoint-2400


# python core/train/grpo.py --config core/train/cfg/experiments/qwen_coder_r64_grpo.yaml --use_sloth --run_name qwen_coder_grpo_a6000 --sft /oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen_coder_7b_r64_hf_20251111_103145/Qwen/Qwen2.5-Coder-7B-Instruct_finetuned

python core/train/grpo.py --config core/train/cfg/experiments/qwen_coder_r64_grpo.yaml --use_sloth --run_name qwen_coder_grpo_format_reward --sft /oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen_coder_7b_r64_hf_20251111_103145/Qwen/Qwen2.5-Coder-7B-Instruct_finetuned --checkpoint /oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen_coder_grpo_format_reward_grpo/checkpoint-600

# python core/train/grpo.py --config core/train/cfg/experiments/qwen_coder_r64_grpo.yaml --use_sloth --run_name qwen_coder_grpo_format_reward_run2_h100 --sft /oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen_coder_7b_r64_hf_20251111_103145/Qwen/Qwen2.5-Coder-7B-Instruct_finetuned 

# accelerate launch core/train/grpo.py --config core/train/cfg/experiments/llama3.3_r64_grpo.yaml --use_sloth --run_name unsloth_llamar64_grpo_multi_gpu --sft /oscar/data/sreda/mabdelat/LLM4TB/output/sft/llama3_r64_20251105_085246/checkpoint-2400