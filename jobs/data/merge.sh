#!/bin/bash
#SBATCH -J merge_data

# Request half an hour of runtime:
#SBATCH -t 48:00:00

# Ask for the GPU partition and 1 GPU
#SBATCH -p batch

#Default resources are 1 core with 2.8GB of memory, SBATCH -p gpu-he --gres=gpu:1 -C ampere.

# Use more memory (4GB) (CPU RAM):
#SBATCH --mem=25g

#SBATCH -n 2
source ~/.bashrc
source /oscar/data/sreda/mabdelat/LLM4TB/venv/bin/activate

REPO_DIR=/oscar/data/sreda/mabdelat/LLM4TB
REPO_DIR_MN=/oscar/data/sreda/mnouh/ScaleLab/LLM4TB

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/output/data_merged_all}"

INPUT_DIRS=(
    "${REPO_DIR_MN}/output/data_generation_verithoughts"
    "${REPO_DIR_MN}/output/data_generation_verithoughts_second_half"
    "${REPO_DIR_MN}/output/data_generation_veritriplets"
    "${REPO_DIR_MN}/output/data_generation_pyra"
    "${REPO_DIR}/output/data_generation_llm4cov"
)

cd "${REPO_DIR}"

echo "Merging datasets into ${OUTPUT_DIR}"
printf '  %s\n' "${INPUT_DIRS[@]}"

python core/data/merge.py \
    --input_dirs "${INPUT_DIRS[@]}" \
    --output_dir "${OUTPUT_DIR}"
