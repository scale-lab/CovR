#!/bin/bash

# Request half an hour of runtime:
#SBATCH -t 72:00:00

# Ask for the GPU partition and 2 GPUs: GPU 0 trains, GPU 1 serves vLLM
#SBATCH -p gpu-he --gres=gpu:2 -C h100

#Default resources are 1 core with 2.8GB of memory, SBATCH -p gpu-he --gres=gpu:1 -C ampere.

# Use more memory (4GB) (CPU RAM):
#SBATCH --mem=200g

#SBATCH -n 32

# Set up the environment by loading modules

export PYTHONUTF8=1
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo

export FLASHINFER_WORKSPACE_BASE="/tmp/${USER}/flashinfer-${SLURM_JOB_ID}"
mkdir -p "$FLASHINFER_WORKSPACE_BASE"

SLURM_ALLOCATED_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
IFS=',' read -r TRAIN_GPU VLLM_GPU _ <<< "${SLURM_ALLOCATED_CUDA_VISIBLE_DEVICES}"
TRAIN_GPU="${TRAIN_GPU:-0}"
VLLM_GPU="${VLLM_GPU:-1}"

source ~/.bashrc

module load python/3.11.11-5e66
module load cuda/12.9.0-cinr

source /oscar/data/sreda/mabdelat/LLM4TB/trl-vllm-clean-env/bin/activate

module load synopsys-vcs
export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
export CUDA_PATH="$CUDA_HOME"
export FLASHINFER_NVCC="$(which nvcc)"
export CUDA_LIB_PATH="${CUDA_HOME}/targets/x86_64-linux/lib"
echo "FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE}"
echo "CUDA_HOME=${CUDA_HOME}"
echo "FLASHINFER_NVCC=${FLASHINFER_NVCC}"

cd /oscar/data/sreda/mabdelat/LLM4TB

export SERVER_HOST=127.0.0.1
export SERVER_PORT=$((18000 + SLURM_JOB_ID % 20000))
export VLLM_GROUP_PORT=$((38000 + SLURM_JOB_ID % 20000))
export BASE_MODEL=Qwen/Qwen2.5-Coder-7B-Instruct
export SFT_LORA= /oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen2.5-coder_full_sft_r512_a512_completion_only_loss_full_data/qwen2.5-coder_full_sft_r512_a512_completion_only_loss_full_data_20260504_154202/Qwen/Qwen2.5-Coder-7B-Instruct_finetuned
export GRPO_CONFIG=core/train/cfg/experiments/grpo/qwen_coder_r64_grpo_vllm_server.yaml

echo "Slurm allocated CUDA_VISIBLE_DEVICES=${SLURM_ALLOCATED_CUDA_VISIBLE_DEVICES}"
echo "Using trainer GPU=${TRAIN_GPU}, vLLM GPU=${VLLM_GPU}"

CUDA_VISIBLE_DEVICES="${VLLM_GPU}" trl vllm-serve \
    --model "${BASE_MODEL}" \
    --host "${SERVER_HOST}" \
    --port "${SERVER_PORT}" \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.85 \
    --max_model_len 5028 &

VLLM_PID=$!
trap 'kill ${VLLM_PID} 2>/dev/null' EXIT

echo "Waiting for TRL vLLM server on http://${SERVER_HOST}:${SERVER_PORT}"
for i in {1..120}; do
    if curl -fsS "http://${SERVER_HOST}:${SERVER_PORT}/get_world_size/" >/dev/null; then
        echo "TRL vLLM server is ready"
        break
    fi
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "vLLM server exited before becoming ready"
        wait "${VLLM_PID}"
        exit 1
    fi
    sleep 5
done

if ! curl -fsS "http://${SERVER_HOST}:${SERVER_PORT}/get_world_size/" >/dev/null; then
    echo "Timed out waiting for TRL vLLM server"
    exit 1
fi

CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" accelerate launch --num_processes 1 --main_process_port 0 core/train/grpo.py \
    --config "${GRPO_CONFIG}" \
    --run_name qwen_coder_multi_gpu_vllm_llm4cov_data_lr_constant \
    --sft "${SFT_LORA}" \
    --use_vllm \
    --vllm_mode server \
    --vllm_server_host "${SERVER_HOST}" \
    --vllm_server_port "${SERVER_PORT}" \
    --vllm_server_timeout 600 \
    --vllm_group_port "${VLLM_GROUP_PORT}" \
    --vllm_tensor_parallel_size 1 \
    --vllm_max_model_length 5028 \
    --with_reasoning \
    --max_workers 32
