#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_NAME=qwen3-8b
GPU_CONSTRAINT=h100
USE_GPU=1
USE_VLLM="1"
MAX_SEQ_LENGTH=32000
MAX_NEW_TOKENS=10000
WITH_REASONING=1

MODEL="${MODEL_NAME}" \
GPU_CONSTRAINT="${GPU_CONSTRAINT}" \
USE_GPU="${USE_GPU}" \
USE_VLLM="${USE_VLLM}" \
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH}" \
WITH_REASONING="${WITH_REASONING}" \
RUN_NAME="qwen_coder_8b_veval" \
OUTPUT_DIR="${OUTPUT_DIR_VEVAL:-output-eval/base_models/vllm/qwen_coder_8b_veval}" \
BATCH_SIZE="8" \
"${SCRIPT_DIR}/_submit_vanilla.sh"

MODEL="${MODEL_NAME}" \
GPU_CONSTRAINT="${GPU_CONSTRAINT}" \
USE_GPU="${USE_GPU}" \
USE_VLLM="${USE_VLLM}" \
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH}" \
WITH_REASONING="${WITH_REASONING}" \
RUN_NAME="qwen3_8b_cvdp" \
OUTPUT_DIR="${OUTPUT_DIR_CVDP:-output-eval/base_models/vllm/qwen3_8b_cvdp}" \
EVAL_FILE="/oscar/data/sreda/mnouh/ScaleLab/LLM4TB/data/eval_llm4cov.jsonl" \
BATCH_SIZE="2" \
"${SCRIPT_DIR}/_submit_vanilla.sh"
