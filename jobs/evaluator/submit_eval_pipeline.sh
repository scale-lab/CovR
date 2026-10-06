#!/bin/bash
# Submit two-stage evaluation pipeline:
#   1. generate answers
#   2. evaluate saved answers with Synopsys/VCS

set -euo pipefail

REPO_DIR="${REPO_DIR:-/oscar/data/sreda/mabdelat/LLM4TB}"
cd "${REPO_DIR}"

MODEL="${MODEL:?Set MODEL}"
RUN_NAME="${RUN_NAME:-${MODEL}_eval}"
OUTPUT_DIR="${OUTPUT_DIR:-output-eval/unified/${RUN_NAME}}"
MERGED_ANSWERS_NAME="${MERGED_ANSWERS_NAME:-all_answers.jsonl}"

GEN_TIME="${GEN_TIME:-24:00:00}"
GEN_MEM="${GEN_MEM:-100g}"
GEN_NTASKS="${GEN_NTASKS:-4}"
GEN_CPUS_PER_TASK="${GEN_CPUS_PER_TASK:-}"

USE_GPU="${USE_GPU:-1}"
GPU_PARTITION="${GPU_PARTITION:-gpu-he}"
GPU_GRES="${GPU_GRES:-gpu:1}"
GPU_CONSTRAINT="${GPU_CONSTRAINT:-b200}"

EVAL_TIME="${EVAL_TIME:-48:00:00}"
EVAL_MEM="${EVAL_MEM:-100g}"
EVAL_PARTITION="${EVAL_PARTITION:-batch}"
EVAL_CPUS="${EVAL_CPUS:-8}"

mkdir -p "${OUTPUT_DIR}"

export REPO_DIR MODEL RUN_NAME OUTPUT_DIR MERGED_ANSWERS_NAME

SBATCH_EXPORT="ALL,REPO_DIR=${REPO_DIR},MODEL=${MODEL},RUN_NAME=${RUN_NAME},OUTPUT_DIR=${OUTPUT_DIR},MERGED_ANSWERS_NAME=${MERGED_ANSWERS_NAME}"
if [[ -n "${LORA:-}" ]]; then
  SBATCH_EXPORT+=",LORA=${LORA}"
fi
if [[ -n "${EVAL_FILE:-}" ]]; then
  SBATCH_EXPORT+=",EVAL_FILE=${EVAL_FILE}"
fi
if [[ -n "${ANSWERS_FILE:-}" ]]; then
  SBATCH_EXPORT+=",ANSWERS_FILE=${ANSWERS_FILE}"
fi

gen_sbatch=(
  sbatch
  --job-name "gen_${RUN_NAME}"
  --time "${GEN_TIME}"
  --mem "${GEN_MEM}"
  -n "${GEN_NTASKS}"
  --export="${SBATCH_EXPORT}"
)

if [[ -n "${GEN_CPUS_PER_TASK}" ]]; then
  gen_sbatch+=(-c "${GEN_CPUS_PER_TASK}")
fi

if [[ "${USE_GPU}" == "1" ]]; then
  gen_sbatch+=(-p "${GPU_PARTITION}" --gres "${GPU_GRES}")
  [[ -n "${GPU_CONSTRAINT}" ]] && gen_sbatch+=(-C "${GPU_CONSTRAINT}")
else
  gen_sbatch+=(-p "${EVAL_PARTITION}")
fi

gen_sbatch+=(jobs/evaluator/generate_answers.sh)

echo "Submitting generation job:"
printf ' %q' "${gen_sbatch[@]}"
echo

GEN_ID=$("${gen_sbatch[@]}" | awk '{print $4}')
echo "Generation job: ${GEN_ID}"

eval_sbatch=(
  sbatch
  --job-name "eval_${RUN_NAME}"
  --dependency "afterok:${GEN_ID}"
  --partition "${EVAL_PARTITION}"
  --time "${EVAL_TIME}"
  --mem "${EVAL_MEM}"
  -n 1
  -c "${EVAL_CPUS}"
  --export="${SBATCH_EXPORT}"
  jobs/evaluator/evaluate_answers.sh
)

echo "Submitting Synopsys evaluation job:"
printf ' %q' "${eval_sbatch[@]}"
echo
EVAL_ID=$("${eval_sbatch[@]}" | awk '{print $4}')
echo "Synopsys evaluation job: ${EVAL_ID}"
if [[ -n "${ANSWERS_FILE:-}" ]]; then
  echo "Answers file will be: ${ANSWERS_FILE}"
else
  echo "Answers file will be discovered from the latest timestamped run under ${OUTPUT_DIR}"
fi
