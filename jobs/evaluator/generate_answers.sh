#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH --mem=50g
#SBATCH -n 4

set -euo pipefail

# --- Setup ---
REPO_DIR="${REPO_DIR:-/oscar/data/sreda/mabdelat/LLM4TB}"
cd "${REPO_DIR}"

set +u
source ~/.bashrc
set -u
module load python/3.11.11-5e66
module load cuda/12.9.0-cinr

source /oscar/data/sreda/mabdelat/LLM4TB/trl-vllm-clean-env/bin/activate

export LANGCHAIN_TRACING_V2=false
export LANGSMITH_TRACING=false

export FLASHINFER_WORKSPACE_BASE="/tmp/${USER}/flashinfer-${SLURM_JOB_ID}"
mkdir -p "$FLASHINFER_WORKSPACE_BASE"
export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
export CUDA_PATH="$CUDA_HOME"
export FLASHINFER_NVCC="$(which nvcc)"
export CUDA_LIB_PATH="${CUDA_HOME}/targets/x86_64-linux/lib"
echo "FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE}"
echo "CUDA_HOME=${CUDA_HOME}"
echo "FLASHINFER_NVCC=${FLASHINFER_NVCC}"

# --- Config ---
MODEL="${MODEL:?Set MODEL}"
RUN_NAME="${RUN_NAME:-${MODEL}_eval}"
OUTPUT_DIR="${OUTPUT_DIR:-output-eval/unified/${RUN_NAME}}"
MERGED_ANSWERS_NAME="${MERGED_ANSWERS_NAME:-all_answers.jsonl}"
EVAL_FILE="${EVAL_FILE:-./data/eval_dataset.jsonl}"

BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_SAMPLES="${NUM_SAMPLES:-10}"
TEMPERATURE="${TEMPERATURE:-0.1}"
MAX_WORKERS="${MAX_WORKERS:-3}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-32000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-10000}"

USE_VLLM="${USE_VLLM:-1}"
WITH_REASONING="${WITH_REASONING:-1}"
MULTI_GPU="${MULTI_GPU:-0}"
NUM_GPUS="${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-1}}"

# --- Prepare output ---
mkdir -p "${OUTPUT_DIR}"
RUN_BASE_DIR="${OUTPUT_DIR}/${MODEL}/${MODEL}_${RUN_NAME}"

# --- Build command ---
cmd=(
  python core/train/evaluator.py
  --model "${MODEL}"
  --run_name "${RUN_NAME}"
  --output_dir "${OUTPUT_DIR}"
  --eval "${EVAL_FILE}"
  --batch_size "${BATCH_SIZE}"
  --num_samples "${NUM_SAMPLES}"
  --temperature "${TEMPERATURE}"
  --max_seq_length "${MAX_SEQ_LENGTH}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --max_workers "${MAX_WORKERS}"
  --generate_only
)

[[ -n "${LORA:-}" ]] && cmd+=(--lora "${LORA}")
[[ "${WITH_REASONING}" == "1" ]] && cmd+=(--with_reasoning)
[[ "${USE_LLM4COV_PROMPT:-0}" == "1" ]] && cmd+=(--use_llm4cov_prompt)
[[ "${USE_SLOTH:-0}" == "1" ]] && cmd+=(--use_sloth)
[[ "${USE_VLLM}" == "1" ]] && cmd+=(--use_vllm)

if [[ "${MULTI_GPU}" == "1" ]]; then
  cmd=(accelerate launch --num_processes "${NUM_GPUS}" --main_process_port 0 "${cmd[@]}")
fi

echo "Running:"
printf ' %q' "${cmd[@]}"
echo

"${cmd[@]}"

latest_run_dir="$(
  find "${RUN_BASE_DIR}" -type f -name 'answers_rank*.jsonl' -printf '%h\n' |
    sort -u |
    while IFS= read -r run_dir; do
      timestamp_dir="$(basename "${run_dir}")"
      if [[ "${timestamp_dir}" =~ ^[0-9]{8}_[0-9]{6}$ ]]; then
        printf '%s\t%s\n' "${timestamp_dir}" "${run_dir}"
      fi
    done |
    sort |
    tail -n 1 |
    cut -f 2-
)"

if [[ -z "${latest_run_dir}" ]]; then
  echo "No timestamped run directory containing answers_rank*.jsonl found under ${RUN_BASE_DIR}!" >&2
  exit 1
fi

mapfile -t answer_files < <(
  find "${latest_run_dir}" -maxdepth 1 -type f -name 'answers_rank*.jsonl' | sort
)

if (( ${#answer_files[@]} > 0 )); then
  ANSWERS_FILE="${latest_run_dir}/${MERGED_ANSWERS_NAME}"
  cat "${answer_files[@]}" > "${ANSWERS_FILE}"
  echo "Latest run directory: ${latest_run_dir}"
  echo "Merged ${#answer_files[@]} rank output file(s) -> ${ANSWERS_FILE}"
else
  echo "No answers_rank*.jsonl files found in latest run directory ${latest_run_dir}!" >&2
  exit 1
fi
