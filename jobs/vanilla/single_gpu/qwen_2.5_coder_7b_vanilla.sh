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
source ~/.bashrc
module load python/3.11.0s-ixrhc3q

source /oscar/data/sreda/mabdelat/LLM4TB/unsloth-b200/bin/activate

module load synopsys-vcs

export LANGCHAIN_TRACING_V2=false
export LANGSMITH_TRACING=false 

export MAX_SEQ_LENGTH=32000
export MAX_NEW_TOKENS=10000

python core/train/evaluator.py --max_seq_length ${MAX_SEQ_LENGTH} --max_new_tokens ${MAX_NEW_TOKENS} --model qwen2.5-coder \
  --run_name qwen_coder_7b_veval  --batch_size 4 \
  --output_dir output-eval/base_models/vllm/qwen_coder_7b_veval \
  --with_reasoning \
  --use_vllm

python core/train/evaluator.py --max_seq_length ${MAX_SEQ_LENGTH} --max_new_tokens ${MAX_NEW_TOKENS} --model qwen2.5-coder \
    --run_name qwen_coder_7b_cvdp  --batch_size 2  \
    --output_dir output-eval/base_models/vllm/qwen_coder_7b_cvdp \
    --eval /oscar/data/sreda/mnouh/ScaleLab/LLM4TB/data/eval_llm4cov.jsonl \
    --with_reasoning \
    --use_vllm
