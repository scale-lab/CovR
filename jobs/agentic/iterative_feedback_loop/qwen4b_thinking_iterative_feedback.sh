#!/bin/bash

# Request half an hour of runtime:
#SBATCH -t 96:00:00

# Ask for the GPU partition and 1 GPU
#SBATCH -p gpu-he --gres=gpu:1 -C a6000

#Default resources are 1 core with 2.8GB of memory, SBATCH -p gpu-he --gres=gpu:1 -C ampere.

# Use more memory (4GB) (CPU RAM):
#SBATCH --mem=50g

#SBATCH -n 4

# Set up the environment by loading modules

source ~/.bashrc

module load python/3.11.11-5e66

source /oscar/data/sreda/mabdelat/LLM4TB/unsloth-b200/bin/activate

module load synopsys-vcs

export LANGCHAIN_TRACING_V2=false
export LANGSMITH_TRACING=false 

MAX_SEQ_LENGTH=32000
MAX_NEW_TOKENS=10000

python core/baselines/iterative_feedback_loop/main.py \
    --enhancer-model qwen4b-thinking \
    --tb-model qwen4b-thinking \
    --checker-model qwen4b-thinking \
    --reflector-model qwen4b-thinking \
    --curator-model qwen4b-thinking \
    --fixer-model qwen4b-thinking \
    --dataset ./data/eval_dataset.jsonl \
    --iters 5 \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --output ./output-eval/baselines/iterative_feedback_loop/vllm/qwen4b-thinking_veval



python core/baselines/iterative_feedback_loop/main.py \
    --enhancer-model qwen4b-thinking \
    --tb-model qwen4b-thinking \
    --checker-model qwen4b-thinking \
    --reflector-model qwen4b-thinking \
    --curator-model qwen4b-thinking \
    --dataset ./data/eval_cvdp.jsonl \
    --fixer-model qwen4b-thinking \
    --iters 5 \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --output ./output-eval/baselines/iterative_feedback_loop/vllm/qwen4b-thinking_cvdp



# python core/train/evaluator.py \
#     --model qwen4b-thinking \
#     --eval ./data/eval_cvdp.jsonl \
#     --output  ./output-eval/baselines/iterative_feedback_loop/qwen4b-thinking_cvdp \
#     --answers /oscar/data/sreda/mabdelat/LLM4TB/output-eval/baselines/iterative_feedback_loop/qwen4b-thinking_cvdp/20260415_175554/answers_llm4dv.jsonl

