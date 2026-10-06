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

source /oscar/data/sreda/mabdelat/LLM4TB/venv_3/bin/activate

module load synopsys-vcs


# python core/train/evaluator.py --use_sloth --model qwen2.5-coder --run_name qwen_coder_reason  --batch_size 2 --lora /oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen_coder_7b_r64_unsloth_reason_20251115_210203/unsloth/Qwen2.5-7B-Instruct_finetuned


# python core/train/evaluator.py --model qwen2.5-coder --run_name qwen_coder_reason_hf_w_base_prompt  --batch_size 2 --base --lora  /oscar/data/sreda/mnouh/ScaleLab/LLM4TB/output/sft/qwen_coder_reasoning_20251117_210504/Qwen/Qwen2.5-Coder-7B-Instruct_finetuned/

python core/train/evaluator.py --model qwen2.5-coder --run_name qwen_coder_reason_base_prompt_eval_cleaned_spec --batch_size 2 --base --lora  /oscar/data/sreda/mnouh/ScaleLab/LLM4TB/output/sft/qwen_coder_reasoning_20251117_210504/Qwen/Qwen2.5-Coder-7B-Instruct_finetuned/ --eval /oscar/data/sreda/mabdelat/LLM4TB/data/eval_dataset_spec_cleaned.jsonl



# python core/train/evaluator.py --model qwen2.5-coder --run_name qwen_coder_reason_hf_wo_unsloth  --batch_size 2  --answers /oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen_coder_7b_r64_hf_reason_synthetic_20251115_213913/answers.jsonl