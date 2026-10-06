#!/bin/bash

# Request half an hour of runtime:
#SBATCH -t 72:00:00

# Ask for the GPU partition and 1 GPU
#SBATCH -p gpu-he --gres=gpu:1 -C b200

#Default resources are 1 core with 2.8GB of memory, SBATCH -p gpu-he --gres=gpu:1 -C ampere.

# Use more memory (4GB) (CPU RAM):
#SBATCH --mem=50g

#SBATCH -n 4

# Set up the environment by loading modules
source ~/.bashrc
module load python/3.11.0s-ixrhc3q

source /oscar/data/sreda/mabdelat/LLM4TB/venv_3/bin/activate

module load synopsys-vcs

export LANGCHAIN_TRACING_V2=false
export LANGSMITH_TRACING=false 


# python core/train/evaluator.py --model llm4cov_stage2  \
#     --run_name llm4cov_stage2 --output_dir output-eval/base_models \
#     --max_seq_length 6028 --batch_size 2 --use_llm4cov_prompt

python core/train/evaluator.py --model llm4cov_stage2  \
    --run_name llm4cov_stage2_cvdp --output_dir output-eval/base_models \
    --max_seq_length 6028 --batch_size 1 \
    --eval ./data/eval_cvdp.jsonl --use_llm4cov_prompt 