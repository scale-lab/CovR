#!/bin/bash

# Request half an hour of runtime:
#SBATCH -t 96:00:00

# Ask for the GPU partition and 1 GPU
#SBATCH -p gpu-he --gres=gpu:1 -C b200

#Default resources are 1 core with 2.8GB of memory, SBATCH -p gpu-he --gres=gpu:1 -C ampere.

# Use more memory (4GB) (CPU RAM):
#SBATCH --mem=100g

#SBATCH -n 10

# Set up the environment by loading modules

source ~/.bashrc

module load python/3.11.11-5e66
module load cuda

export FLASHINFER_WORKSPACE_BASE="/tmp/${USER}/flashinfer-${SLURM_JOB_ID}"
mkdir -p "$FLASHINFER_WORKSPACE_BASE"

source /oscar/data/sreda/mabdelat/LLM4TB/trl-vllm-clean-env/bin/activate

module load synopsys-vcs

export LANGCHAIN_TRACING_V2=false
export LANGSMITH_TRACING=false 

export MAX_SEQ_LENGTH=32000
export MAX_NEW_TOKENS=10000
export NUM_GPUS="${SLURM_GPUS_ON_NODE:-2}"
export NUM_CPUS="${SLURM_CPUS_PER_TASK:-32}"

export VEVAL_BATCH_SIZE=4
export CVDP_BATCH_SIZE=2

export MODEL_NAME="qwen4b_instruct"
export CFG="/oscar/data/sreda/mabdelat/LLM4TB/core/train/cfg/experiments/sft/${MODEL_NAME}/sft_r512_a512.yaml"

export RUN_NAME=${MODEL_NAME}_full_sft_r512_a512_full_data_without_reasoning

if python core/train/sft.py \
    --config ${CFG} \
    --max_workers ${NUM_CPUS} \
    --use_vllm \
    --run_name ${RUN_NAME} \
    --output_dir ./output/sft/${RUN_NAME} \
    --generate_only \
    --completion_only_loss
then
SFT_OUTPUT_ROOT="./output/sft/${RUN_NAME}"

LATEST_RUN_DIR="$(find "${SFT_OUTPUT_ROOT}" -mindepth 1 -maxdepth 1 -type d -name "${RUN_NAME}_*" | sort | tail -n 1)"

if [[ -z "${LATEST_RUN_DIR}" ]]; then
  echo "No SFT run directory found under ${SFT_OUTPUT_ROOT}" >&2
  exit 1
fi

merge_answers() {
  local test_dir="$1"
  local answers_file="${test_dir}/all_answers.jsonl"
  mapfile -t answer_files < <(find "${test_dir}" -maxdepth 1 -type f -name 'answers_rank*.jsonl' | sort)
  if (( ${#answer_files[@]} == 0 )); then
    echo "No answers_rank*.jsonl files found under ${test_dir}" >&2
    return 1
  fi
  cat "${answer_files[@]}" > "${answers_file}"
  echo "${answers_file}"
}

if ! VEVAL_ANSWERS="$(merge_answers "${LATEST_RUN_DIR}/test_veval")"; then
  exit 1
fi

if ! CVDP_ANSWERS="$(merge_answers "${LATEST_RUN_DIR}/test_cvdp")"; then
  exit 1
fi

sbatch \
  --export=ALL,MODEL="${MODEL_NAME}",WITH_REASONING=0,ANSWERS_FILE="${VEVAL_ANSWERS}",NUM_SAMPLES=10,MAX_WORKERS=15,OUTPUT_DIR="${LATEST_RUN_DIR}/test_veval/synopsys" \
  jobs/evaluator/evaluate_answers.sh

sbatch \
  --export=ALL,MODEL="${MODEL_NAME}",WITH_REASONING=0,ANSWERS_FILE="${CVDP_ANSWERS}",NUM_SAMPLES=10,MAX_WORKERS=15,OUTPUT_DIR="${LATEST_RUN_DIR}/test_cvdp/synopsys" \
  jobs/evaluator/evaluate_answers.sh

fi
