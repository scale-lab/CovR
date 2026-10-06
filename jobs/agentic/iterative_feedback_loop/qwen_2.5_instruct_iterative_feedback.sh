#!/bin/bash

# Request half an hour of runtime:
#SBATCH -t 96:00:00

# Ask for the GPU partition and 1 GPU
#SBATCH -p gpu-he --gres=gpu:1 -C a6000

#Default resources are 1 core with 2.8GB of memory, SBATCH -p gpu-he --gres=gpu:1 -C ampere.

# Use more memory (4GB) (CPU RAM):
#SBATCH --mem=100g

#SBATCH -n 4

# Set up the environment by loading modules

source ~/.bashrc

module load python/3.11.11-5e66

source /oscar/data/sreda/mabdelat/LLM4TB/unsloth-b200/bin/activate

module load synopsys-vcs

export LANGCHAIN_TRACING_V2=false
export LANGSMITH_TRACING=false 

MAX_SEQ_LENGTH=32000
MAX_NEW_TOKENS=10000
NUM_SAMPLES=${NUM_SAMPLES:-5}
VEVAL_OUTPUT_ROOT=./output-eval/baselines/iterative_feedback_loop/vllm/qwen_2.5_instruct_7b_veval
CVDP_OUTPUT_ROOT=./output-eval/baselines/iterative_feedback_loop/vllm/qwen_2.5_instruct_7b_cvdp

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
        --export=ALL,MODEL="qwen2.5-instruct",ANSWERS_FILE="${answers_file}",NUM_SAMPLES="${NUM_SAMPLES}",MAX_WORKERS=15,OUTPUT_DIR="${latest_run_dir}/synopsys" \
        jobs/evaluator/evaluate_answers.sh
}


if python core/baselines/iterative_feedback_loop/main.py \
    --enhancer-model  qwen2.5-instruct \
    --tb-model qwen2.5-instruct \
    --checker-model qwen2.5-instruct \
    --reflector-model qwen2.5-instruct \
    --curator-model qwen2.5-instruct \
    --fixer-model qwen2.5-instruct \
    --dataset ./data/eval_dataset.jsonl \
    --iters 5 \
    --num_samples "${NUM_SAMPLES}" \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --output "${VEVAL_OUTPUT_ROOT}" \
    --generate-only
then
    submit_eval_job "${VEVAL_OUTPUT_ROOT}"
fi


if python core/baselines/iterative_feedback_loop/main.py \
    --enhancer-model qwen2.5-instruct \
    --tb-model qwen2.5-instruct \
    --checker-model qwen2.5-instruct \
    --reflector-model qwen2.5-instruct \
    --curator-model qwen2.5-instruct \
    --dataset ./data/eval_cvdp.jsonl \
    --fixer-model qwen2.5-instruct \
    --iters 5 \
    --num_samples "${NUM_SAMPLES}" \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --output "${CVDP_OUTPUT_ROOT}" \
    --generate-only
then
    submit_eval_job "${CVDP_OUTPUT_ROOT}"
fi
