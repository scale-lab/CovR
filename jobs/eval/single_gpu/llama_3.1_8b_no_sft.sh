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
# module load python/3.11.0s-ixrhc3q

source /oscar/data/sreda/mabdelat/LLM4TB/venv/bin/activate

module load synopsys-vcs

python core/train/evaluator.py --max_seq_length 3028 --model llama3.1:8b --run_name llama_wo_sloth_base_single_gpu  --batch_size 2 --base 


# --answers /oscar/data/sreda/mabdelat/LLM4TB/output-eval/llama3.1:8b/llama3.1:8b_llama_wo_sloth_base_single_gpu/20260218_121839/answers_rank0.jsonl