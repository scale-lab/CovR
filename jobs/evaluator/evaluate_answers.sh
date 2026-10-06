#!/bin/bash
# CPU-only saved-answer Synopsys/VCS evaluation job.

#SBATCH -t 48:00:00
#SBATCH --mem=200g
#SBATCH -n 1
#SBATCH -c 15

set -euo pipefail

# --- Setup ---
REPO_DIR="${REPO_DIR:-/oscar/data/sreda/mabdelat/LLM4TB}"
cd "${REPO_DIR}"

set +u
source ~/.bashrc
set -u

source /oscar/data/sreda/mabdelat/LLM4TB/trl-vllm-clean-env/bin/activate
module load synopsys-vcs

export LANGCHAIN_TRACING_V2=false
export LANGSMITH_TRACING=false

# --- Config ---
MODEL="${MODEL:?Set MODEL to match the generation job}"
CPUS=${SLURM_CPUS_PER_TASK:-8}
MAX_WORKERS="${MAX_WORKERS:-$((CPUS))}"
NUM_SAMPLES="${NUM_SAMPLES:-10}"
RUN_NAME="${RUN_NAME:-${MODEL}_eval}"
OUTPUT_DIR="${OUTPUT_DIR:-output-eval/unified/${RUN_NAME}}"
MERGED_ANSWERS_NAME="${MERGED_ANSWERS_NAME:-all_answers.jsonl}"

if [[ -z "${ANSWERS_FILE:-}" ]]; then
  RUN_BASE_DIR="${OUTPUT_DIR}/${MODEL}/${MODEL}_${RUN_NAME}"
  latest_run_dir="$(
    find "${RUN_BASE_DIR}" -type f -name "${MERGED_ANSWERS_NAME}" -printf '%h\n' |
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
    echo "No timestamped run directory containing ${MERGED_ANSWERS_NAME} found under ${RUN_BASE_DIR}!" >&2
    exit 1
  fi

  ANSWERS_FILE="${latest_run_dir}/${MERGED_ANSWERS_NAME}"
fi

if [[ ! -s "${ANSWERS_FILE}" ]]; then
  echo "Answers file not found: ${ANSWERS_FILE}" >&2
  exit 1
fi

BASE_DIR="$(dirname "${ANSWERS_FILE}")"
OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${BASE_DIR}/synopsys}"
mkdir -p "${OUTPUT_DIR}"

# --- Run Synopsys/VCS evaluation ---
EVAL_RUN_NAME="${RUN_NAME}_synopsys"

echo "Evaluating: ${ANSWERS_FILE}"
echo "Run name: ${EVAL_RUN_NAME}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Workers: ${MAX_WORKERS}"

python core/train/evaluator.py \
  --model "${MODEL}" \
  --run_name "${EVAL_RUN_NAME}" \
  --output_dir "${OUTPUT_DIR}" \
  --answers "${ANSWERS_FILE}" \
  --num_samples "${NUM_SAMPLES}" \
  --max_workers "${MAX_WORKERS}" \
  --no-generate_only
