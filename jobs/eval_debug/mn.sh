#!/bin/bash

# Request half an hour of runtime:
#SBATCH -t 72:00:00

# Ask for the GPU partition and 1 GPU
#SBATCH -p gpu-he --gres=gpu:1 -C h100

#Default resources are 1 core with 2.8GB of memory, SBATCH -p gpu-he --gres=gpu:1 -C ampere.

# Use more memory (4GB) (CPU RAM):
#SBATCH --mem=100g

#SBATCH -n 8


# Set up the environment by loading modules
source ~/.bashrc
source /oscar/data/sreda/mabdelat/LLM4TB/trl-vllm-clean-env/bin/activate

module load synopsys-vcs

python core/train/sft.py \
    --config /oscar/data/sreda/mabdelat/LLM4TB/core/train/cfg/experiments/qwen2.5-b/sft_sgd.yaml \
    --run_name qwen_coder_reas_sgd800_sftr_test --with_reasoning 
    
# python core/data_generation/reasoner_v0.2.py --input /oscar/data/sreda/mabdelat/LLM4TB/output/constant_sources/verithoughts_second_half/train_dataset_met_coverage.jsonl --output /oscar/data/sreda/mnouh/ScaleLab/LLM4TB/output/reasoning/data_generation_verithoughts_second_half/train_dataset_met_coverage_reasoner_v0_2_output_sgd.jsonl --clean-jsonl