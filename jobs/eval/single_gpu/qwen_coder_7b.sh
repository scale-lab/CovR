#!/bin/bash

# Request half an hour of runtime:
#SBATCH -t 72:00:00

# Ask for the GPU partition and 1 GPU
#SBATCH -p gpu-he --gres=gpu:1 -C a6000

#Default resources are 1 core with 2.8GB of memory, SBATCH -p gpu-he --gres=gpu:1 -C ampere.

# Use more memory (4GB) (CPU RAM):
#SBATCH --mem=50g

#SBATCH -n 4

# Set up the environment by loading modules
source ~/.bashrc
module load python/3.11.0s-ixrhc3q

source /oscar/data/sreda/mabdelat/LLM4TB/venv_2/bin/activate

module load synopsys-vcs


# python core/train/evaluator.py --model qwen2.5-coder --lora /oscar/data/sreda/mabdelat/LLM4TB/output/sft/unsloth_qwen_20251109_211417/unsloth/Qwen2.5-7B-Instruct_finetuned --batch_size 2



# python core/train/evaluator.py --model qwen2.5-coder --lora /oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen_coder_7b_r64_all_data_20251111_100517/unsloth/Qwen2.5-7B-Instruct_finetuned --batch_size 2 --run_name qwen_r64_all_data_unsloth


# python core/train/evaluator.py --model qwen2.5-coder --lora /oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen_coder_7b_r64_hf_20251111_103145/Qwen/Qwen2.5-Coder-7B-Instruct_finetuned --batch_size 2 --run_name qwen_r64_hf



python core/train/evaluator.py --model qwen2.5-coder --lora /oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen_coder_7b_r64_unsloth_reason_synthetic_20251115_210203/unsloth/Qwen2.5-7B-Instruct_finetuned --batch_size 2 --run_name qwen_r64_synthetic_unsloth_reason --answers /oscar/data/sreda/mabdelat/LLM4TB/output-eval/qwen2.5-coder/qwen2.5-coder_qwen_r64_synthetic_unsloth_reason/20251116_113314/answers.jsonl

