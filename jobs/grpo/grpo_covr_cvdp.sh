#!/bin/bash

# Request half an hour of runtime:
#SBATCH -t 48:00:00

# Ask for the GPU partition and 1 GPU
#SBATCH -p batch

#Default resources are 1 core with 2.8GB of memory, SBATCH -p gpu-he --gres=gpu:1 -C ampere.

# Use more memory (4GB) (CPU RAM):
#SBATCH --mem=50g

#SBATCH -n 15

# Set up the environment by loading modules
source ~/.bashrc
source /oscar/data/sreda/mabdelat/LLM4TB/venv/bin/activate

module load synopsys-vcs 

export LANGCHAIN_TRACING_V2=false
export LANGSMITH_TRACING=false 

python core/train/evaluator.py \
     --model qwen2.5-coder \
     --lora /oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen_coder_grpo_base_format_syntax_func_cov_bs_32_w_sft_small_sample_unsloth_grpo/checkpoint-400/  --max_seq_length 6028 --eval /oscar/data/sreda/mnouh/ScaleLab/LLM4TB/data/eval_llm4cov.jsonl --batch_size 1 --answers /oscar/data/sreda/mabdelat/LLM4TB/output-eval/qwen2.5-coder/qwen2.5-coder/20260403_192305/answers_rank0.jsonl 