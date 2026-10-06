#!/bin/bash

# Request half an hour of runtime:
#SBATCH -t 72:00:00

# Ask for the GPU partition and 1 GPU
#SBATCH -p gpu-he --gres=gpu:1 -C h100

#Default resources are 1 core with 2.8GB of memory, SBATCH -p gpu-he --gres=gpu:1 -C ampere.

# Use more memory (4GB) (CPU RAM):
#SBATCH --mem=50g

#SBATCH -n 4

# Set up the environment by loading modules
source ~/.bashrc
module load python/3.11.0s-ixrhc3q

source /oscar/data/sreda/mabdelat/LLM4TB/venv_3/bin/activate

module load synopsys-vcs

# python core/train/evaluator.py --model llama3.1:8b --lora /oscar/data/sreda/mabdelat/LLM4TB/output/sft/llama3_r64_20251024_140431/meta-llama/Meta-Llama-3.1-8B-Instruct_finetuned --batch_size 2


# python core/train/evaluator.py --model qwen2.5-coder  --lora /oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen_coder_grpo_reason_grpo/checkpoint-400/ --batch_size 2 --run_name qwen_coder_grpo_reason --use_sloth


# python core/train/evaluator.py --model qwen2.5-coder   --base --lora  /oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen_coder_grpo_base_format_syntax_func_cov_bs_32_w_sft_small_sample_grpo/checkpoint-300 --batch_size 2 --run_name qwen_coder_grpo_reason_hf_high_lr_checkpoint_300

# ## Unsloth Checkpoints
# python core/train/evaluator.py --model qwen2.5-coder  --base --lora  /oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen_coder_grpo_reason_unsloth_lower_lr_grpo/checkpoint-400 --batch_size 2 --run_name qwen_coder_grpo_reason_unsloth_lower_lr_checkpoint_400

# python core/train/evaluator.py --model qwen2.5-coder  --base --lora  /oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen_coder_grpo_reason_unsloth_lower_lr_grpo/checkpoint-300 --batch_size 2 --run_name qwen_coder_grpo_reason_unsloth_lower_lr_checkpoint_300

# python core/train/evaluator.py --model qwen2.5-coder  --base --lora  /oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen_coder_grpo_reason_unsloth_lower_lr_grpo/checkpoint-200 --batch_size 2 --run_name qwen_coder_grpo_reason_unsloth_lower_lr_checkpoint_200

# python core/train/evaluator.py --model qwen2.5-coder  --base --lora  /oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen_coder_grpo_reason_unsloth_lower_lr_grpo/checkpoint-100 --batch_size 2 --run_name qwen_coder_grpo_reason_unsloth_lower_lr_checkpoint_100


# ## qwen_coder_grpo_reason_hf_lower_lr_grpo checkpoints 

# python core/train/evaluator.py --model qwen2.5-coder   --base --lora   /oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen_coder_grpo_reason_hf_lower_lr_grpo/checkpoint-600 --batch_size 2 --run_name qwen_coder_grpo_reason_hf_lower_lr_checkpoint_600

# python core/train/evaluator.py --model qwen2.5-coder   --base --lora   /oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen_coder_grpo_reason_hf_lower_lr_grpo/checkpoint-500 --batch_size 2 --run_name qwen_coder_grpo_reason_hf_lower_lr_checkpoint_500

# python core/train/evaluator.py --model qwen2.5-coder   --base --lora   /oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen_coder_grpo_reason_hf_lower_lr_grpo/checkpoint-400 --batch_size 2 --run_name qwen_coder_grpo_reason_hf_lower_lr_checkpoint_400

# python core/train/evaluator.py --model qwen2.5-coder   --base --lora   /oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen_coder_grpo_reason_hf_lower_lr_grpo/checkpoint-300 --batch_size 2 --run_name qwen_coder_grpo_reason_hf_lower_lr_checkpoint_300

# python core/train/evaluator.py --model qwen2.5-coder   --base --lora   /oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen_coder_grpo_reason_hf_lower_lr_grpo/checkpoint-200 --batch_size 2 --run_name qwen_coder_grpo_reason_hf_lower_lr_checkpoint_200

# python core/train/evaluator.py --model qwen2.5-coder   --base --lora   /oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen_coder_grpo_reason_hf_lower_lr_grpo/checkpoint-100 --batch_size 2 --run_name qwen_coder_grpo_reason_hf_lower_lr_checkpoint_100


## qwen_coder_grpo_reason_hf_lower_max_seq_len_lr_grpo checkpoints 


# python core/train/evaluator.py --model qwen2.5-coder   --base --lora   /oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen_coder_grpo_reason_hf_lower_lr_high_max_seq_grpo/checkpoint-500 --batch_size 2 --run_name qwen_coder_grpo_reason_hf_lower_lr_high_max_seq_checkpoint_500

# python core/train/evaluator.py --model qwen2.5-coder   --base --lora   /oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen_coder_grpo_reason_hf_lower_lr_high_max_seq_grpo/checkpoint-400 --batch_size 2 --run_name qwen_coder_grpo_reason_hf_lower_lr_high_max_seq_checkpoint_400

# python core/train/evaluator.py --model qwen2.5-coder   --base --lora   /oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen_coder_grpo_reason_hf_lower_lr_high_max_seq_grpo/checkpoint-300 --batch_size 2 --run_name qwen_coder_grpo_reason_hf_lower_lr_high_max_seq_checkpoint_300

# python core/train/evaluator.py --model qwen2.5-coder   --base --lora   /oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen_coder_grpo_reason_hf_lower_lr_high_max_seq_grpo/checkpoint-200 --batch_size 2 --run_name qwen_coder_grpo_reason_hf_lower_lr_high_max_seq_checkpoint_200

# python core/train/evaluator.py --model qwen2.5-coder   --base --lora   /oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen_coder_grpo_reason_hf_lower_lr_high_max_seq_grpo/checkpoint-100 --batch_size 2 --run_name qwen_coder_grpo_reason_hf_lower_lr_high_max_seq_checkpoint_100




## qwen_coder_grpo_reason_hf_lower_lr_high_max_seq_large_ds_grpo checkpoints 


python core/train/evaluator.py --model qwen2.5-coder   --base --lora   /oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen_coder_grpo_reason_hf_lower_lr_high_max_seq_large_ds_grpo/checkpoint-500 --batch_size 2 --run_name qwen_coder_grpo_reason_hf_lower_lr_high_max_seq_large_ds_grpo_checkpoint_500

python core/train/evaluator.py --model qwen2.5-coder   --base --lora   /oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen_coder_grpo_reason_hf_lower_lr_high_max_seq_large_ds_grpo/checkpoint-400 --batch_size 2 --run_name qwen_coder_grpo_reason_hf_lower_lr_high_max_seq_large_ds_grpo_checkpoint_400

python core/train/evaluator.py --model qwen2.5-coder   --base --lora   /oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen_coder_grpo_reason_hf_lower_lr_high_max_seq_large_ds_grpo/checkpoint-300 --batch_size 2 --run_name qwen_coder_grpo_reason_hf_lower_lr_high_max_seq_large_ds_grpo_checkpoint_300

python core/train/evaluator.py --model qwen2.5-coder   --base --lora   /oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen_coder_grpo_reason_hf_lower_lr_high_max_seq_large_ds_grpo/checkpoint-200 --batch_size 2 --run_name qwen_coder_grpo_reason_hf_lower_lr_high_max_seq_large_ds_grpo_checkpoint_200

python core/train/evaluator.py --model qwen2.5-coder   --base --lora   /oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen_coder_grpo_reason_hf_lower_lr_high_max_seq_large_ds_grpo/checkpoint-100 --batch_size 2 --run_name qwen_coder_grpo_reason_hf_lower_lr_high_max_seq_large_ds_grpo_checkpoint_100


# python core/train/evaluator.py --answers /oscar/data/sreda/mabdelat/LLM4TB/output-eval/qwen2.5-coder/qwen2.5-coder_qwen_coder_grpo_reason_wo_sloth/20251118_080606/answers.jsonl --model qwen2.5-coder --batch_size 2 --run_name qwen_coder_grpo_reason_wo_sloth
