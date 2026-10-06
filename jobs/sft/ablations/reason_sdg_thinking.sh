#!/bin/bash

# Request half an hour of runtime:
#SBATCH -t 96:00:00

# Ask for the GPU partition and 1 GPU
#SBATCH -p gpu-he --gres=gpu:1 -C h100

#Default resources are 1 core with 2.8GB of memory, SBATCH -p gpu-he --gres=gpu:1 -C ampere.

# Use more memory (4GB) (CPU RAM):
#SBATCH --mem=100g

#SBATCH -n 10

# Set up the environment by loading modules

source ~/.bashrc

module load python/3.11.11-5e66

source /oscar/data/sreda/mabdelat/LLM4TB/trl-vllm-latest-env/bin/activate

module load synopsys-vcs

export LANGCHAIN_TRACING_V2=false
export LANGSMITH_TRACING=false 

python core/train/sft.py \
    --config /oscar/data/sreda/mabdelat/LLM4TB/core/train/cfg/experiments/qwen2.5-b/sft_sdg_thinking.yaml \
    --with_reasoning \
    --max_workers 10 \
    --use_vllm \
    --output_dir ./output/sft/sdg_thinking_new_env


export LANGCHAIN_TRACING_V2=false
export LANGSMITH_TRACING=false 

# MAX_SEQ_LENGTH=32000
# MAX_NEW_TOKENS=10000

# python core/train/evaluator.py --max_seq_length ${MAX_SEQ_LENGTH} \
#     --max_new_tokens ${MAX_NEW_TOKENS} \
#     --model qwen2.5-coder --run_name qwen_coder_7b_sdg_thinking  \
#     --batch_size 4  \
#     --output_dir  /oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen2.5-coder_reason_sft_sdg_thinking_20260425_001210/eval \
#     --with_reasoning \
#     --max_workers 10 \
#     --use_vllm \
#     --lora /oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen2.5-coder_reason_sft_sdg_thinking_20260425_001210/Qwen/Qwen2.5-Coder-7B-Instruct_finetuned