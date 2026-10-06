#!/bin/bash
#SBATCH --job-name=qwen25-coder-lora-srl

# Request half an hour of runtime:
#SBATCH -t 96:00:00

# Ask for the GPU partition and 2 GPUs: one for base-agent server, one for TB-generator/enhancer LoRA server.
#SBATCH -p gpu-he --gres=gpu:2 -C b200

#Default resources are 1 core with 2.8GB of memory, SBATCH -p gpu-he --gres=gpu:1 -C ampere.

# Use more memory (4GB) (CPU RAM):
#SBATCH --mem=100g

#SBATCH -n 1

#SBATCH -c 10

set -euo pipefail

# Set up the environment by loading modules

set +u
source ~/.bashrc
set -u

module load python/3.11.11-5e66
module load cuda/12.9.0-cinr

source /oscar/data/sreda/mabdelat/LLM4TB/unsloth-b200/bin/activate

export FLASHINFER_WORKSPACE_BASE="/tmp/${USER}/flashinfer-${SLURM_JOB_ID}"
mkdir -p "$FLASHINFER_WORKSPACE_BASE"
export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
export CUDA_PATH="$CUDA_HOME"
export FLASHINFER_NVCC="$(which nvcc)"
export CUDA_LIB_PATH="${CUDA_HOME}/targets/x86_64-linux/lib"
echo "FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE}"
echo "CUDA_HOME=${CUDA_HOME}"
echo "FLASHINFER_NVCC=${FLASHINFER_NVCC}"

module load synopsys-vcs

export LANGCHAIN_TRACING_V2=false
export LANGSMITH_TRACING=false 

export MAX_SEQ_LENGTH=32000
export MAX_NEW_TOKENS=10000
export NUM_SAMPLES=${NUM_SAMPLES:-10}
export SAMPLE_WORKERS=${SAMPLE_WORKERS:-10}

# IMPORTANT: this LoRA adapter must be trained from BASE_MODEL.
export BASE_MODEL=Qwen/Qwen2.5-Coder-7B-Instruct
export BASE_SERVED_MODEL=qwen-base
export TB_SERVED_MODEL=tb_lora

export LORA_QWEN_CODER_7B=/oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen_coder_grpo_unsloth_grpo/checkpoint-600

export LORA_PATH="${LORA_QWEN_CODER_7B}"
export AGENT_PORT=${AGENT_PORT:-8108}
export GENERATOR_PORT=${GENERATOR_PORT:-8109}
export AGENT_API_BASE=http://127.0.0.1:${AGENT_PORT}/v1
export GENERATOR_API_BASE=http://127.0.0.1:${GENERATOR_PORT}/v1
export VEVAL_OUTPUT_ROOT=./output-eval/baselines/self_reflection_loop/vllm/qwen_2.5_coder_7b_lora_veval
export CVDP_OUTPUT_ROOT=./output-eval/baselines/self_reflection_loop/vllm/qwen_2.5_coder_7b_lora_cvdp

if [[ ! -d "${LORA_PATH}" ]]; then
    echo "Missing LoRA adapter directory: ${LORA_PATH}" >&2
    exit 1
fi

if [[ ! -f "${LORA_PATH}/adapter_config.json" ]]; then
    echo "LoRA adapter_config.json not found in: ${LORA_PATH}" >&2
    exit 1
fi

mkdir -p ./output-eval/baselines/self_reflection_loop/vllm/server_logs

cleanup() {
    jobs -pr | xargs -r kill
}
trap cleanup EXIT

echo "Starting base-agent vLLM server on GPU 0..."
CUDA_VISIBLE_DEVICES=0 vllm serve "${BASE_MODEL}" \
    --host 127.0.0.1 \
    --port "${AGENT_PORT}" \
    --served-model-name "${BASE_SERVED_MODEL}" \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.9 \
    --max-model-len "${MAX_SEQ_LENGTH}" \
    --generation-config vllm \
    > ./output-eval/baselines/self_reflection_loop/vllm/server_logs/base_server.log 2>&1 &

BASE_SERVER_PID=$!

echo "Starting TB-generator/enhancer LoRA vLLM server on GPU 1..."
CUDA_VISIBLE_DEVICES=1 vllm serve "${BASE_MODEL}" \
    --host 127.0.0.1 \
    --port "${GENERATOR_PORT}" \
    --served-model-name qwen-generator-base \
    --enable-lora \
    --lora-modules "${TB_SERVED_MODEL}=${LORA_PATH}" \
    --max-lora-rank 512 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.9 \
    --max-model-len "${MAX_SEQ_LENGTH}" \
    --generation-config vllm \
    > ./output-eval/baselines/self_reflection_loop/vllm/server_logs/generator_server.log 2>&1 &

GENERATOR_SERVER_PID=$!

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

wait_for_server "base-agent server" "${AGENT_API_BASE}" "${BASE_SERVER_PID}"
wait_for_server "TB-generator/enhancer server" "${GENERATOR_API_BASE}" "${GENERATOR_SERVER_PID}"

submit_eval_job() {
    local output_root="$1"
    local latest_run_dir
    local answers_file

    latest_run_dir="$(find "${output_root}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"

    if [[ -z "${latest_run_dir}" ]]; then
        echo "No self-reflection run directory found under ${output_root}" >&2
        exit 1
    fi

    answers_file="${latest_run_dir}/answers_llm4dv.jsonl"
    if [[ ! -s "${answers_file}" ]]; then
        echo "Answers file not found: ${answers_file}" >&2
        exit 1
    fi

    sbatch -p batch \
        --export=ALL,MODEL="${TB_SERVED_MODEL}",ANSWERS_FILE="${answers_file}",NUM_SAMPLES="${NUM_SAMPLES}",MAX_WORKERS=15,OUTPUT_DIR="${latest_run_dir}/synopsys" \
        jobs/evaluator/evaluate_answers.sh
}

if python core/baselines/self_reflection_loop/main.py \
    --prompter-model "${BASE_SERVED_MODEL}" \
    --enhancer-model "${TB_SERVED_MODEL}" \
    --tb-model "${TB_SERVED_MODEL}" \
    --checker-model "${BASE_SERVED_MODEL}" \
    --reflector-model "${BASE_SERVED_MODEL}" \
    --curator-model "${BASE_SERVED_MODEL}" \
    --fixer-model "${BASE_SERVED_MODEL}" \
    --api-base "${AGENT_API_BASE}" \
    --generator-api-base "${GENERATOR_API_BASE}" \
    --enhancer-api-base "${GENERATOR_API_BASE}" \
    --dataset ./data/eval_dataset.jsonl \
    --iters 5 \
    --num_samples "${NUM_SAMPLES}" \
    --sample-workers "${SAMPLE_WORKERS}" \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --problem-workers 5 \
    --output "${VEVAL_OUTPUT_ROOT}" \
    --generate-only
then
    submit_eval_job "${VEVAL_OUTPUT_ROOT}"
fi


if python core/baselines/self_reflection_loop/main.py \
    --prompter-model "${BASE_SERVED_MODEL}" \
    --enhancer-model "${TB_SERVED_MODEL}" \
    --tb-model "${TB_SERVED_MODEL}" \
    --checker-model "${BASE_SERVED_MODEL}" \
    --reflector-model "${BASE_SERVED_MODEL}" \
    --curator-model "${BASE_SERVED_MODEL}" \
    --fixer-model "${BASE_SERVED_MODEL}" \
    --api-base "${AGENT_API_BASE}" \
    --generator-api-base "${GENERATOR_API_BASE}" \
    --enhancer-api-base "${GENERATOR_API_BASE}" \
    --dataset ./data/eval_cvdp.jsonl \
    --iters 5 \
    --num_samples "${NUM_SAMPLES}" \
    --sample-workers "${SAMPLE_WORKERS}" \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --problem-workers 5 \
    --output "${CVDP_OUTPUT_ROOT}" \
    --generate-only
then
    submit_eval_job "${CVDP_OUTPUT_ROOT}"
fi
