#!/bin/bash

# Request half an hour of runtime:
#SBATCH -t 24:00:00

# Ask for the GPU partition and 1 GPU
#SBATCH -p gpu --gres=gpu:1

#Default resources are 1 core with 2.8GB of memory, SBATCH -p gpu-he --gres=gpu:1 -C ampere.

# Use more memory (4GB) (CPU RAM):
#SBATCH --mem=50g

#SBATCH -n 4

# Set up the environment by loading modules
source ~/.bashrc
module load python/3.11.0s-ixrhc3q

source /oscar/data/sreda/mabdelat/LLM4TB/venv_3/bin/activate

module load synopsys-vcs


# python core/train/evaluator.py  --answers  /oscar/data/sreda/mnouh/ScaleLab/LLM4TB/output/sft/qwen_coder_reasoning_20251117_210504/answers.jsonl --model qwen2.5-coder --run_name qwen_coder_reason_with_strict_eval_on_error 


python core/train/evaluator.py  --answers /oscar/data/sreda/mabdelat/LLM4TB/output-eval/qwen2.5-coder/qwen2.5-coder_qwen_coder_reason_hf_w_base_prompt/20260325_170513/answers_rank0.jsonl --model qwen2.5-coder --run_name qwen_coder_reason_with_strict_eval_on_error_base_prompt 

# python core/train/evaluator.py  --answers /oscar/data/sreda/mabdelat/LLM4TB/output-eval/qwen2.5-coder/qwen2.5-coder_qwen_coder_grpo_reason_unsloth_lower_lr_checkpoint_300/20260326_110622/answers_rank0.jsonl --model qwen2.5-coder --run_name qwen_coder_reason_grpo_unsloth_ckp_300_with_strict_eval_on_error 