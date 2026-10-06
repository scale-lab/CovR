#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_NAME=llm4cov_stage2
GPU_CONSTRAINT=a6000
USE_GPU=1
USE_LLM4COV_PROMPT=1
WITH_REASONING=0
MAX_SEQ_LENGTH=32000
MAX_NEW_TOKENS=10000

# MODEL="${MODEL_NAME}" \
# RUN_NAME="llm4cov_stage2" \
# OUTPUT_DIR="output-eval/base_models/vllm" \
# USE_GPU="${USE_GPU}" \
# GPU_CONSTRAINT="${GPU_CONSTRAINT}" \
# USE_LLM4COV_PROMPT="${USE_LLM4COV_PROMPT}" \
# WITH_REASONING="${WITH_REASONING}" \
# MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH}" \
# BATCH_SIZE="2" \
# "${SCRIPT_DIR}/_submit_vanilla.sh"

MODEL="${MODEL_NAME}" \
RUN_NAME="llm4cov_stage2_cvdp" \
OUTPUT_DIR="output-eval/base_models/vllm" \
EVAL_FILE="./data/eval_cvdp.jsonl" \
USE_GPU="${USE_GPU}" \
GPU_CONSTRAINT="${GPU_CONSTRAINT}" \
USE_LLM4COV_PROMPT="${USE_LLM4COV_PROMPT}" \
WITH_REASONING="${WITH_REASONING}" \
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH}" \
BATCH_SIZE="8" \
"${SCRIPT_DIR}/_submit_vanilla.sh"
