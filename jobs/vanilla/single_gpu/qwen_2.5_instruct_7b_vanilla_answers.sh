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

MAX_SEQ_LENGTH=32000
MAX_NEW_TOKENS=10000

# python core/train/evaluator.py --max_seq_length 10028 --model qwen2.5-instruct \
#   --run_name qwen2.5-instruct_7b_veval  --batch_size 2 \
#   --output_dir output-eval/base_models/qwen2.5-instruct_7b_veval_msl_10028 \
#   --with_reasoning \
#   --answers /oscar/data/sreda/mabdelat/LLM4TB/output-eval/base_models/qwen2.5-instruct_7b_veval_msl_10028/qwen2.5-instruct/qwen2.5-instruct_qwen2.5-instruct_7b_veval/20260417_134511/answers_rank0.jsonl \
#   --max_workers 10 


python core/train/evaluator.py --max_seq_length ${MAX_SEQ_LENGTH} --max_new_tokens ${MAX_NEW_TOKENS} --model qwen2.5-instruct \
    --run_name qwen2.5-instruct_7b_cvdp  --batch_size 2  \
    --output_dir output-eval/base_models/qwen2.5-instruct_7b_cvdp_msl_10028 \
    --eval /oscar/data/sreda/mnouh/ScaleLab/LLM4TB/data/eval_llm4cov.jsonl \
    --with_reasoning \
    --answers /oscar/data/sreda/mabdelat/LLM4TB/output-eval/base_models/qwen2.5-instruct_7b_cvdp_msl_10028/qwen2.5-instruct/qwen2.5-instruct_qwen2.5-instruct_7b_cvdp/20260418_113626/answers_rank0.jsonl \
    --max_workers 10 \
    --use_vllm
