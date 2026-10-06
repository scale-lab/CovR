#!/bin/bash

# Request half an hour of runtime:
#SBATCH -t 48:00:00

# Ask for the GPU partition and 1 GPU
#SBATCH -p batch

#Default resources are 1 core with 2.8GB of memory, SBATCH -p gpu-he --gres=gpu:1 -C ampere.

# Use more memory (4GB) (CPU RAM):
#SBATCH --mem=100g

#SBATCH -n 15

# Set up the environment by loading modules
source ~/.bashrc
source /oscar/data/sreda/mabdelat/LLM4TB/venv/bin/activate

module load synopsys-vcs 

export LANGCHAIN_TRACING_V2=false
export LANGSMITH_TRACING=false 

# python core/train/evaluator.py --model llm4cov_stage2  --run_name llm4cov --output_dir output-eval/base_models --max_seq_length 5028 --batch_size 1 --eval /oscar/data/sreda/mnouh/ScaleLab/LLM4TB/data/eval_llm4cov.jsonl --use_llm4cov_prompt --answers /oscar/data/sreda/mabdelat/LLM4TB/output-eval/base_models/llm4cov_stage2/llm4cov_stage2_llm4cov/20260405_122819/answers_rank0.jsonl

python core/train/evaluator.py --model llm4cov_stage2  --run_name llm4cov --output_dir output-eval/base_models --max_seq_length 5028 --batch_size 1 --eval /oscar/data/sreda/mnouh/ScaleLab/LLM4TB/data/eval_llm4cov.jsonl --use_llm4cov_prompt --answers /oscar/data/sreda/mabdelat/LLM4TB/output-eval/base_models/llm4cov_stage2/llm4cov_stage2_llm4cov/20260405_122256/answers_rank0.jsonl