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
module load python/3.11.11-5e66

source /oscar/data/sreda/mabdelat/LLM4TB/unsloth-b200/bin/activate

module load synopsys-vcs

echo "Job started at: $(date '+%Y-%m-%d %H:%M:%S %Z')"
start_time=$(date +%s)

python core/train/evaluator.py  \
    --model qwen2.5-instruct --with_reasoning \
    --eval ./data/eval_subset.jsonl \
    --max_seq_length 4028  --batch_size 4 \
    --output_dir ./output-eval/eval_debug/qwen2.5-instruct_vllm \
    --temperature 0.01 \
    --use_vllm

exit_code=$?
end_time=$(date +%s)

echo "Job ended at:   $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "Exit code: $exit_code"

elapsed=$((end_time - start_time))
echo "Elapsed time: $((elapsed / 3600))h $(((elapsed % 3600) / 60))m $((elapsed % 60))s"

exit $exit_code