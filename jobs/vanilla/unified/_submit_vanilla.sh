#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
cd "${REPO_DIR}"

export REPO_DIR
export GEN_TIME="${GEN_TIME:-24:00:00}"
export EVAL_TIME="${EVAL_TIME:-48:00:00}"
export EVAL_MEM="${EVAL_MEM:-100g}"
export EVAL_PARTITION="${EVAL_PARTITION:-batch}"

bash jobs/evaluator/submit_eval_pipeline.sh
