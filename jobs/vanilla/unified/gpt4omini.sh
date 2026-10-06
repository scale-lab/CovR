#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_NAME=gpt-4o-mini
USE_GPU=0
USE_VLLM=0
MAX_SEQ_LENGTH=32000
MAX_NEW_TOKENS=10000
WITH_REASONING=1
VENV=/oscar/data/sreda/mabdelat/LLM4TB/venv/bin/activate
MAX_WORKERS=10

MODEL="${MODEL_NAME}" \
RUN_NAME="gpt4omini_veval" \
OUTPUT_DIR="${OUTPUT_DIR_VEVAL:-output-eval/base_models/gpt4omini_veval}" \
USE_GPU="${USE_GPU}" \
USE_VLLM="${USE_VLLM}" \
VENV="${VENV}" \
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH}" \
WITH_REASONING="${WITH_REASONING}" \
BATCH_SIZE="1" \
MAX_WORKERS="${MAX_WORKERS}" \
"${SCRIPT_DIR}/_submit_vanilla.sh"

MODEL="${MODEL_NAME}" \
RUN_NAME="gpt4omini_cvdp" \
OUTPUT_DIR="${OUTPUT_DIR_CVDP:-output-eval/base_models/gpt4omini_cvdp}" \
EVAL_FILE="/oscar/data/sreda/mnouh/ScaleLab/LLM4TB/data/eval_llm4cov.jsonl" \
USE_GPU="${USE_GPU}" \
USE_VLLM="${USE_VLLM}" \
VENV="${VENV}" \
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH}" \
WITH_REASONING="${WITH_REASONING}" \
BATCH_SIZE="1" \
MAX_WORKERS="${MAX_WORKERS}" \
"${SCRIPT_DIR}/_submit_vanilla.sh"
