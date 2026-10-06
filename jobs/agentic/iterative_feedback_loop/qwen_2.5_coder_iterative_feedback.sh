#!/bin/bash

# Request half an hour of runtime:
#SBATCH -t 96:00:00

# Ask for the GPU partition and 1 GPU
#SBATCH -p gpu-he --gres=gpu:1 -C h100

#Default resources are 1 core with 2.8GB of memory, SBATCH -p gpu-he --gres=gpu:1 -C ampere.

# Use more memory (4GB) (CPU RAM):
#SBATCH --mem=100g

#SBATCH -n 1

#SBATCH -c 5

set -euo pipefail

# Set up the environment by loading modules

set +u
source ~/.bashrc
set -u

module load python/3.11.11-5e66

source /oscar/data/sreda/mabdelat/LLM4TB/unsloth-b200/bin/activate

module load synopsys-vcs

export LANGCHAIN_TRACING_V2=false
export LANGSMITH_TRACING=false 

export MAX_SEQ_LENGTH=32000
export MAX_NEW_TOKENS=10000
export NUM_SAMPLES=${NUM_SAMPLES:-10}
export SAMPLE_WORKERS=${SAMPLE_WORKERS:-10}

export BASE_MODEL=Qwen/Qwen2.5-Coder-7B-Instruct
export SERVED_MODEL=qwen2.5-coder
export VLLM_PORT=${VLLM_PORT:-8200}
export API_BASE=http://127.0.0.1:${VLLM_PORT}/v1
export VEVAL_OUTPUT_ROOT=./output-eval/baselines/iterative_feedback_loop/vllm/qwen_2.5_coder_7b_veval
export CVDP_OUTPUT_ROOT=./output-eval/baselines/iterative_feedback_loop/vllm/qwen_2.5_coder_7b_cvdp

mkdir -p ./output-eval/baselines/iterative_feedback_loop/vllm/server_logs

cleanup() {
    jobs -pr | xargs -r kill
}
trap cleanup EXIT

echo "Starting vLLM server on GPU 0..."
CUDA_VISIBLE_DEVICES=0 vllm serve "${BASE_MODEL}" \
    --host 127.0.0.1 \
    --port "${VLLM_PORT}" \
    --served-model-name "${SERVED_MODEL}" \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.9 \
    --max-model-len "${MAX_SEQ_LENGTH}" \
    --generation-config vllm \
    > ./output-eval/baselines/iterative_feedback_loop/vllm/server_logs/qwen25_coder_server.log 2>&1 &

SERVER_PID=$!

wait_for_server() {
    local name="$1"
    local url="$2"
    local pid="$3"
    local timeout_seconds=1800
    local waited=0

    echo "Waiting for ${name} at ${url}/models ..."
    until curl -fsS "${url}/models" >/dev/null; do
        if ! kill -0 "${pid}" 2>/dev/null; then
            echo "${name} exited before becoming ready. Check server logs." >&2
            exit 1
        fi

        if (( waited >= timeout_seconds )); then
            echo "Timed out waiting for ${name} after ${timeout_seconds}s." >&2
            exit 1
        fi

        sleep 10
        waited=$((waited + 10))
    done
    echo "${name} is ready."
}

wait_for_server "Qwen2.5-Coder vLLM server" "${API_BASE}" "${SERVER_PID}"

submit_eval_job() {
    local output_root="$1"
    local latest_run_dir
    local answers_file

    latest_run_dir="$(find "${output_root}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"

    if [[ -z "${latest_run_dir}" ]]; then
        echo "No iterative-feedback run directory found under ${output_root}" >&2
        exit 1
    fi

    answers_file="${latest_run_dir}/answers_llm4dv.jsonl"
    if [[ ! -s "${answers_file}" ]]; then
        echo "Answers file not found: ${answers_file}" >&2
        exit 1
    fi

    sbatch \
        --export=ALL,MODEL="${SERVED_MODEL}",ANSWERS_FILE="${answers_file}",NUM_SAMPLES="${NUM_SAMPLES}",MAX_WORKERS=15,OUTPUT_DIR="${latest_run_dir}/synopsys" \
        jobs/evaluator/evaluate_answers.sh
}


# if python core/baselines/iterative_feedback_loop/main.py \
#     --prompter-model "${SERVED_MODEL}" \
#     --enhancer-model "${SERVED_MODEL}" \
#     --tb-model "${SERVED_MODEL}" \
#     --checker-model "${SERVED_MODEL}" \
#     --reflector-model "${SERVED_MODEL}" \
#     --curator-model "${SERVED_MODEL}" \
#     --fixer-model "${SERVED_MODEL}" \
#     --api-base "${API_BASE}" \
#     --generator-api-base "${API_BASE}" \
#     --enhancer-api-base "${API_BASE}" \
#     --dataset ./data/eval_dataset.jsonl \
#     --iters 5 \
#     --num_samples "${NUM_SAMPLES}" \
#     --sample-workers "${SAMPLE_WORKERS}" \
#     --max_seq_length ${MAX_SEQ_LENGTH} \
#     --max_new_tokens ${MAX_NEW_TOKENS} \
#     --problem-workers 10 \
#     --output "${VEVAL_OUTPUT_ROOT}" \
#     --generate-only
# then
#     submit_eval_job "${VEVAL_OUTPUT_ROOT}"
# fi


if python core/baselines/iterative_feedback_loop/main.py \
    --prompter-model "${SERVED_MODEL}" \
    --enhancer-model "${SERVED_MODEL}" \
    --tb-model "${SERVED_MODEL}" \
    --checker-model "${SERVED_MODEL}" \
    --reflector-model "${SERVED_MODEL}" \
    --curator-model "${SERVED_MODEL}" \
    --fixer-model "${SERVED_MODEL}" \
    --api-base "${API_BASE}" \
    --generator-api-base "${API_BASE}" \
    --enhancer-api-base "${API_BASE}" \
    --dataset ./data/eval_cvdp.jsonl \
    --iters 5 \
    --num_samples "${NUM_SAMPLES}" \
    --sample-workers "${SAMPLE_WORKERS}" \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --problem-workers 10 \
    --output "${CVDP_OUTPUT_ROOT}" \
    --generate-only
then
    submit_eval_job "${CVDP_OUTPUT_ROOT}"
fi
