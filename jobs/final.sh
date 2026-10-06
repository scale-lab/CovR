#!/usr/bin/env bash
set -euo pipefail

## Final Commands to run evaluations

# Qwen2.5-Coder

## SFT 
LORA=/oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen2.5-coder_full_sft_r512_a512_full_data_without_reasoning/qwen2.5-coder_full_sft_r512_a512_full_data_without_reasoning_20260507_172807/Qwen/Qwen2.5-Coder-7B-Instruct_finetuned \
MODEL=qwen2.5-coder \
OUTPUT_DIR=/oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen2.5-coder_full_sft_r512_a512_full_data_without_reasoning/qwen2.5-coder_full_sft_r512_a512_full_data_without_reasoning_20260507_172807/cvdp_target \
RUN_NAME=sft_qwen2.5_coder_cvdp_target_cov \
EVAL_FILE=./data/eval_cvdp.jsonl \
bash jobs/evaluator/submit_eval_pipeline.sh

## SFT-W/Reason
LORA=/oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen2.5-coder_full_sft_r512_a512_completion_only_loss_full_data/qwen2.5-coder_full_sft_r512_a512_completion_only_loss_full_data_20260504_154202/Qwen/Qwen2.5-Coder-7B-Instruct_finetuned \
MODEL=qwen2.5-coder \
OUTPUT_DIR=/oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen2.5-coder_full_sft_r512_a512_completion_only_loss_full_data/qwen2.5-coder_full_sft_r512_a512_completion_only_loss_full_data_20260504_154202/cvdp_target \
RUN_NAME=sft_w_reason_qwen2.5_coder_cvdp_target_cov \
EVAL_FILE=./data/eval_cvdp.jsonl \
bash jobs/evaluator/submit_eval_pipeline.sh

## SFT-W/Reason + RL
LORA=/oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen_coder_grpo_unsloth_grpo/checkpoint-600 \
MODEL=qwen2.5-coder \
OUTPUT_DIR=/oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen_coder_grpo_unsloth_grpo/checkpoint-600 \
RUN_NAME=grpo_qwen2.5_coder_cvdp_target_cov \
EVAL_FILE=./data/eval_cvdp.jsonl \
bash jobs/evaluator/submit_eval_pipeline.sh



# Qwen4b-Instruct

## SFT 
LORA=/oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen4b_instruct_full_sft_r512_a512_full_data_without_reasoning/qwen4b_instruct_full_sft_r512_a512_full_data_without_reasoning_20260507_173003/Qwen/Qwen3-4B-Instruct-2507_finetuned \
MODEL=qwen4b-instruct \
OUTPUT_DIR=/oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen4b_instruct_full_sft_r512_a512_full_data_without_reasoning/qwen4b_instruct_full_sft_r512_a512_full_data_without_reasoning_20260507_173003/cvdp_target \
RUN_NAME=sft_qwen4b_instruct_cvdp_target_cov \
EVAL_FILE=./data/eval_cvdp.jsonl \
bash jobs/evaluator/submit_eval_pipeline.sh

## SFT-W/Reason
LORA=/oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen4b_instruct_full_sft_r512_a512_full_data/qwen4b_instruct_full_sft_r512_a512_full_data_20260507_173012/Qwen/Qwen3-4B-Instruct-2507_finetuned \
MODEL=qwen4b-instruct \
OUTPUT_DIR=/oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen4b_instruct_full_sft_r512_a512_full_data/qwen4b_instruct_full_sft_r512_a512_full_data_20260507_173012/cvdp_target \
RUN_NAME=sft_w_reason_qwen4b_instruct_cvdp_target_cov \
EVAL_FILE=./data/eval_cvdp.jsonl \
bash jobs/evaluator/submit_eval_pipeline.sh

## SFT-W/Reason + RL
LORA=/oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen3_4b_grpo_unsloth_1000_dataset_grpo/checkpoint-500 \
MODEL=qwen4b-instruct \
OUTPUT_DIR=/oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen3_4b_grpo_unsloth_1000_dataset_grpo/checkpoint-500 \
RUN_NAME=grpo_qwen4b_instruct_cvdp_target_cov \
EVAL_FILE=./data/eval_cvdp.jsonl \
bash jobs/evaluator/submit_eval_pipeline.sh
