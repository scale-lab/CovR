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
module load python/3.11.0s-ixrhc3q

source /oscar/data/sreda/mabdelat/LLM4TB/venv_2/bin/activate

module load synopsys




python core/train/evaluator.py --run_name veval_answers --model qwen2.5-coder --lora /oscar/data/sreda/mnouh/ScaleLab/LLM4TB/output/sft/qwen_coder_reasoning_20251117_210504/Qwen/Qwen2.5-Coder-7B-Instruct_finetuned --with_reasoning --answers /oscar/data/sreda/mnouh/ScaleLab/LLM4TB/output/sft/qwen_coder_reasoning_20251117_210504/answers_verilogeval_2.jsonl


