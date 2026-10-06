# MIT License
#
# Copyright (c) 2026, CovR, SCALE Lab, Brown University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Finetunes LLM with GRPO
"""
import os 
import sys
import yaml
import fcntl

import re
import os

try:
    from unsloth import FastLanguageModel
except ImportError as exc:
    pass 

import torch
import argparse 
import numpy as np

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from omegaconf import OmegaConf

from trl import GRPOConfig, GRPOTrainer
from huggingface_hub import login
# from accelerate import Accelerator
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from torch.utils.data import DataLoader
from functools import partial
from peft import PeftModel, PeftConfig
from vllm import SamplingParams
from contextlib import contextmanager

from core.utils import get_logger
from core.train.cfg.config import combine_cfgs
from core.train.utils import write_json
from core.train.utils import plot_loss
from core.train.instruct_dataset import InstructDataset
from core.train.utils import set_seed, print_trainable_parameters, plot_cov_pass_each_k
from core.train.instruct_dataset import HFDatasetWrapper
from core.train.model import get_model
from core.train.evaluator import Evaluator, _extract_testbrench
from core.data_generation.utils import compute_avg_coverage
from core.train.reward import reward_format, reward_coverage

access_token = os.getenv('HF_ACCESS_TOKEN')
login(token=access_token)


def grpo_unsloth(
    model_name: str, 
    dataset, 
    config, 
    logger,
    w_format,
    w_syntax,
    w_func,
    w_cov,
    checkpoint=None,
    output_dir="./results",
    sft_lora_path=None,
    max_workers=1
):
    # try:
    #     from unsloth import FastLanguageModel
    # except ImportError as exc:
    #     raise ImportError(
    #         "Unsloth is required for --use_sloth. Install/load an environment "
    #         "with unsloth, or run without --use_sloth."
    #     ) from exc

    if sft_lora_path:
        model_name = sft_lora_path

    if config.train.batch_size % config.train.num_generations != 0:
        raise ValueError(
            "For GRPO, train.batch_size must be divisible by "
            f"train.num_generations. Got batch_size={config.train.batch_size}, "
            f"num_generations={config.train.num_generations}."
        )

    max_lora_rank = config.lora.lora_r
    if sft_lora_path:
        try:
            peft_config = PeftConfig.from_pretrained(sft_lora_path)
            adapter_rank = getattr(peft_config, "r", None)
            if adapter_rank is not None:
                max_lora_rank = max(max_lora_rank, int(adapter_rank))
        except Exception as exc:
            logger.warning(f"Could not infer LoRA rank from {sft_lora_path}: {exc}")
    
    # Get local rank from torch/accelerate launch
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device_map = {"": local_rank} if config.bnb.load_in_4bit else None

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=config.train.max_seq_length,
        load_in_4bit=config.bnb.load_in_4bit,
        fast_inference=True,
        max_lora_rank=max_lora_rank,
        device_map=device_map, # << MODIFIED: Apply device map,
        use_gradient_checkpointing="unsloth",
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # model.to(f"cuda:{torch.cuda.current_device()}")

    if not sft_lora_path: 
        model = FastLanguageModel.get_peft_model(
            model,
            r=config.lora.lora_r,
            lora_alpha=config.lora.lora_alpha,
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
            use_gradient_checkpointing=config.train.gradient_checkpointing,
            random_state=config.seed,
            lora_dropout=0.0,
            bias="none",
            use_rslora=False,
            loftq_config=None,
        )


    print_trainable_parameters(model, logger)
    print(model)

    max_completion_length = config.train.max_seq_length -  config.train.max_prompt_length  

    def filter_long_prompts(example):
        prompt_text = example["prompt"]  # already templated
        # tokenize with the Unsloth tokenizer you got from FastLanguageModel.from_pretrained
        input_ids = tokenizer(
            prompt_text,
            add_special_tokens=True,      # important: keep consistent with trainer behavior
            return_attention_mask=False,
            truncation=False,              # don't hide the fact it's too long
        )["input_ids"]
        return len(input_ids) <= config.train.max_prompt_length


    dataset["train"] = dataset["train"].filter(filter_long_prompts)
    dataset["valid"] = dataset["valid"].filter(filter_long_prompts)

    dataset["train"] = dataset["train"].remove_columns(["input_ids"])
    dataset["valid"] = dataset["valid"].remove_columns(["input_ids"])

    # Randomly select 1000 training samples
    dataset["train"] = dataset["train"].shuffle(seed=42).select(
        range(min(config.train.max_data_set_size, len(dataset["train"])))
    )
    print("Training size:", len(dataset["train"]))
    dataset["valid"] = dataset["valid"].shuffle(seed=42).select(range(min(32, len(dataset["valid"]))))


    vllm_sampling_params = SamplingParams(
        min_p = 0.1,
        top_p = 1.0,
        top_k = -1,
        seed = config.seed,
        stop = [tokenizer.eos_token],
        include_stop_str_in_output = True
    )

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    training_args = GRPOConfig(
        ddp_find_unused_parameters = False,
        temperature = 1.0,
        top_p=1.0,
        top_k=0,
        min_p=0.1,
        learning_rate = config.train.learning_rate,     #5e-6
        weight_decay = 0.001,
        warmup_steps=config.train.warmup_steps,
        lr_scheduler_type = config.train.lr_scheduler_type,  #  "linear"
        optim = "adamw_8bit",
        logging_steps=config.train.logging_steps,
        per_device_train_batch_size=config.train.batch_size,
        per_device_eval_batch_size=config.train.batch_size,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps, # make finetuing 7 times slower !
        num_generations=config.train.num_generations,
        loss_type=config.train.loss_type,
        beta=config.train.beta,
        max_prompt_length = config.train.max_prompt_length,
        max_completion_length = max_completion_length,
        num_train_epochs=config.train.epochs,
        max_steps=config.train.max_steps,
        dataloader_num_workers=config.train.dataloader_num_workers,
        bf16=use_bf16,
        fp16=use_fp16,
        tf32=use_bf16,
        report_to=config.train.report_to,
        run_name=config.train.run_name,  # name of the W&B run (optional)
        output_dir=output_dir,
        logging_strategy=config.train.logging_strategy,
        eval_strategy=config.train.evaluation_strategy,
        save_strategy=config.train.save_strategy,
        save_steps=config.train.save_steps,
        eval_steps=config.train.eval_steps, 
        eval_accumulation_steps = 1,
        fp16_full_eval = True,
        load_best_model_at_end=config.train.load_best_model_at_end,
        gradient_checkpointing=config.train.gradient_checkpointing
    )


    reward_fn = partial(
        reward_coverage,
        w_format=w_format,
        w_syntax=w_syntax,
        w_func=w_func,
        w_cov=w_cov,
        num_samples=1,
        output_dir=output_dir,
        max_workers=max_workers,
    )
    reward_fn.__name__ = "reward_fn" 

    # accelerator = Accelerator()
    # model = accelerator.unwrap_model(model)
    # print(model.config)

    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs=reward_fn,
        args = training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"]
    )
    
    # trainer.model = accelerator.unwrap_model(trainer.model)
    # print(trainer.model.config)
    # if hasattr(trainer, "model") and hasattr(trainer.model, "module") and not hasattr(trainer.model, "config"):
    #     trainer.model.config = trainer.model.module.config

    if checkpoint: 
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        trainer.train()

    trainer.model.save_pretrained(os.path.join(output_dir, f"{model_name}_grpo_finetuned"))

    return model 


def grpo(
    model_name: str, 
    dataset, 
    config, 
    logger,
    w_format,
    w_syntax,
    w_func,
    w_cov,
    checkpoint=None,
    sft_lora_path=None,
    output_dir="./results",
    max_workers=1,
    use_vllm=False,
    vllm_mode="colocate",
    vllm_server_host="127.0.0.1",
    vllm_server_port=8000,
    vllm_server_timeout=600.0,
    vllm_group_port=51216,
    vllm_gpu_memory_utilization=0.3,
    vllm_tensor_parallel_size=1,
    vllm_max_model_length=None,

):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        token=access_token, 
        use_fast=config.tokenizer.use_fast,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token # end-of-sequence token
    tokenizer.padding_side = "left"  # for batched generation

    if config.train.batch_size % config.train.num_generations != 0:
        raise ValueError(
            "For GRPO, train.batch_size must be divisible by "
            f"train.num_generations. Got batch_size={config.train.batch_size}, "
            f"num_generations={config.train.num_generations}."
        )

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    bnb_config = None
    if config.bnb.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.bnb.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=config.bnb.bnb_4bit_use_double_quant,
        )

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device_map = {"": local_rank} if config.bnb.load_in_4bit else None

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=access_token,
        quantization_config=bnb_config,
        device_map = device_map,
        # attn_implementation="flash_attention_2",
        dtype=compute_dtype,      # match the 4-bit compute dtype
    )
    base_model.config.pretraining_tp = 1


    if sft_lora_path is None:
        model = base_model
        peft_config = LoraConfig(
            lora_alpha=config.lora.lora_alpha,
            lora_dropout=config.lora.lora_dropout,
            r=config.lora.lora_r,
            use_dora=config.lora.use_dora,
            bias="none",
            task_type="CAUSAL_LM",
            # target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj"]
        )
        if config.bnb.load_in_4bit:
            model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, peft_config)
    else:
        print("[LORA] Mergin Lora path: ", sft_lora_path)
        model = PeftModel.from_pretrained(base_model, sft_lora_path, is_trainable=True)
        # model = model.merge_and_unload() 
    
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    
    if hasattr(model, "generation_config"):
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id


    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    print_trainable_parameters(model, logger)
    print(model)

    # call model before finetuning 
    # test_model(model, tokenizer, dataset['test'][0:20])
   
    model.config.use_cache = False

    if config.train.gradient_checkpointing: 
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    else:
        model.gradient_checkpointing_disable()

    max_completion_length = config.train.max_seq_length -  config.train.max_prompt_length  

    def filter_long_prompts(example):
        prompt_text = example["prompt"]  
        input_ids = tokenizer(
            prompt_text,
            add_special_tokens=True,      # important: keep consistent with trainer behavior
            return_attention_mask=False,
            truncation=False,              # don't hide the fact it's too long
        )["input_ids"]
        return len(input_ids) <= config.train.max_prompt_length

    dataset["train"] = dataset["train"].filter(filter_long_prompts)
    dataset["valid"] = dataset["valid"].filter(filter_long_prompts)
    dataset["train"] = dataset["train"].remove_columns(["input_ids"])
    dataset["valid"] = dataset["valid"].remove_columns(["input_ids"])

    # Randomly select 1000 training samples
    dataset["train"] = dataset["train"].shuffle(seed=42).select(
        range(min(config.train.max_data_set_size, len(dataset["train"])))
    )

    dataset["valid"] = dataset["valid"].shuffle(seed=42).select(range(min(32, len(dataset["valid"]))))

    print("Training size:", len(dataset["train"]))
    vllm_max_model_length = vllm_max_model_length or config.train.max_seq_length

    training_args = GRPOConfig(
        # VLLM configuration
        use_vllm=use_vllm,
        vllm_mode=vllm_mode,
        vllm_server_host=vllm_server_host,
        vllm_server_port=vllm_server_port,
        vllm_server_timeout=vllm_server_timeout,
        vllm_group_port=vllm_group_port,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=vllm_tensor_parallel_size,
        vllm_max_model_length=vllm_max_model_length,
        vllm_importance_sampling_correction=False,

        temperature=1.0,
        top_p=1.0,
        top_k=0,
        min_p=0.1,
        generation_kwargs={
            "stop": [tokenizer.eos_token],
            "include_stop_str_in_output": True,
        },
        num_generations=config.train.num_generations,
        per_device_train_batch_size=config.train.batch_size,
        per_device_eval_batch_size=config.train.batch_size,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps, # make finetuing 7 times slower !
        loss_type=config.train.loss_type,
        beta=config.train.beta,
        num_train_epochs=config.train.epochs,
        warmup_steps=config.train.warmup_steps,
        dataloader_num_workers=config.train.dataloader_num_workers,
        max_steps=config.train.max_steps,
        learning_rate=config.train.learning_rate,
        lr_scheduler_type=config.train.lr_scheduler_type,
        bf16=use_bf16,
        fp16=use_fp16,
        tf32=use_bf16,
        # max_prompt_length=config.train.max_prompt_length,
        max_completion_length=max_completion_length,    
        optim="paged_adamw_8bit",
        logging_steps=config.train.logging_steps,
        eval_delay=0,
        logging_strategy=config.train.logging_strategy,
        eval_strategy=config.train.evaluation_strategy,
        save_strategy=config.train.save_strategy,
        save_steps=config.train.save_steps,
        eval_steps=config.train.eval_steps, 
        output_dir=output_dir,
        load_best_model_at_end=config.train.load_best_model_at_end,
        run_name=config.train.run_name,  # name of the W&B run (optional)
        report_to=config.train.report_to,
        gradient_checkpointing=config.train.gradient_checkpointing  # Leads to reduction in memory at slighly decrease in speed
    )

    reward_fn = partial(
        reward_coverage, 
        w_format=w_format, 
        w_syntax=w_syntax,
        w_func=w_func,
        w_cov=w_cov,
        num_samples=1, 
        output_dir=output_dir,
        max_workers=max_workers
    )
    reward_fn.__name__ = "reward_fn" 

    tokenizer.padding_side = "left"
    
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"]
    )

    # Train model
    if checkpoint: 
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        trainer.train()
        
    trainer.model.save_pretrained(os.path.join(output_dir, f"{model_name}_grpo_finetuned"))

    return model 




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help="Path to config file for training", required=False, default="core/train/cfg/experiments/llama3.3_r64.yaml")
    parser.add_argument('--checkpoint', type=str, help="Resume finetuning from checkpoint", required=False, default=None)
    parser.add_argument('--wandb_id', type=str, help="Resume reportining to same wandb id", required=False, default=None)
    parser.add_argument('--sft', type=str, help="Path to SFT checkpoint we want to grpo", required=False, default=None)
    parser.add_argument('--use_sloth', action='store_const', const=True, help="Use sloth models for faster finetuning", default=False)
    parser.add_argument('--run_name', type=str, help="Name of the run", default=None)
    parser.add_argument("--k", type=list, help="K values", default=[1, 10, 20])
    parser.add_argument('--w_format', type=float, help="format reward weight", default=0.02)
    parser.add_argument('--w_syntax', type=float, help="syntax reward weight", default=0.05)
    parser.add_argument('--w_func', type=float, help="func reward weight", default=0.08)
    parser.add_argument('--w_cov', type=float, help="coverage reward weight", default=0.85)
    parser.add_argument('--output_dir', type=str, help="Path to output directory ", required=False, default="./output/sft")
    parser.add_argument("--max_workers", type=int, help="max workers for parallel simulation", default=3)
    parser.add_argument("--use_vllm", action="store_true", help="to use VLLM in evaluation")
    parser.add_argument("--vllm_mode", choices=["colocate", "server"], default="colocate", help="TRL vLLM mode")
    parser.add_argument("--vllm_server_host", type=str, default="127.0.0.1", help="vLLM server host for server mode")
    parser.add_argument("--vllm_server_port", type=int, default=8000, help="vLLM server port for server mode")
    parser.add_argument("--vllm_server_timeout", type=float, default=600.0, help="vLLM server timeout for server mode")
    parser.add_argument("--vllm_group_port", type=int, default=51216, help="vLLM weight-sync communicator port")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.2, help="vLLM GPU memory utilization for colocate mode")
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=1, help="vLLM tensor parallel size")
    parser.add_argument("--vllm_max_model_length", type=int, default=None, help="vLLM max model length")
    parser.add_argument("--loss_type", type=str, default=None, help="GRPO loss type. Overrides train.loss_type from YAML.")
    parser.add_argument("--with_reasoning", action="store_true", help="Use reasoning annotation always true by default GRPO needs reasoning", default=True)
    parser.add_argument("--generate_only", action=argparse.BooleanOptionalAction, default=True, help="Only generate answers JSONL and skip Synopsys/VCS evaluation.")
    
    args, cli_overrides = parser.parse_known_args()

    config_file = args.config 
    checkpoint = args.checkpoint
    wandb_id = args.wandb_id
    sft = args.sft
    k = args.k 
    use_sloth = args.use_sloth
    use_vllm = args.use_vllm
    vllm_mode = args.vllm_mode
    vllm_server_host = args.vllm_server_host
    vllm_server_port = args.vllm_server_port
    vllm_server_timeout = args.vllm_server_timeout
    vllm_group_port = args.vllm_group_port
    vllm_gpu_memory_utilization = args.vllm_gpu_memory_utilization
    vllm_tensor_parallel_size = args.vllm_tensor_parallel_size
    vllm_max_model_length = args.vllm_max_model_length
    loss_type = args.loss_type
    with_reasoning = args.with_reasoning
    max_workers = args.max_workers
    run_name = args.run_name
    w_format = args.w_format
    w_syntax = args.w_syntax
    w_func = args.w_func
    w_cov = args.w_cov
    generate_only = args.generate_only

    if wandb_id: 
        os.environ["WANDB_RUN_ID"] = wandb_id
        os.environ["WANDB_RESUME"] = "must"  

    print(f"[GRPO] weights: syntax={w_syntax}, func={w_func}, cov={w_cov}, format={w_format}")

    output_dir = args.output_dir
    
    config = combine_cfgs(config_file, cli_overrides=cli_overrides)

    shuffle = config.DATASET.SHUFFLE 
    model_name = config.LLM.NAME 
    
    if run_name:
        config.train.run_name = run_name
   
    if loss_type:
        config.train.loss_type = loss_type

    output_dir = os.path.join(output_dir, f"{config.train.run_name}_grpo")
    os.makedirs(output_dir, exist_ok=True)

    logfile = os.path.join(output_dir, f"{model_name}_grpo")
    logger = get_logger(output=logfile)
    
    logger.info(f"BLUE: Number of GPUs {torch.cuda.device_count()}")
    logger.info(f"BLUE: Finetuning {model_name}")

    set_seed(config.seed, n_gpu=torch.cuda.device_count())

    with open(os.path.join(output_dir, "config.yaml"), 'w') as f:
        yaml.dump(config, f)

    model = get_model(model_name, load_model=False)
    model_id = model.model_ids[model_name]

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, 
        token=access_token, 
    ) 
    
    # tokenizer.pad_token = tokenizer.eos_token # end-of-sequence token
    # tokenizer.padding_side = "left"
    # config.tokenizer.padding_side # pad to have max seq length
    
    train_json =  config.DATASET.TRAIN 
    test_json = config.DATASET.TEST_VEVAL 

    instruct_dataset = InstructDataset(
        train_json=train_json,     
        test_json=test_json,
        train_split=0.8,
        valid_split=0.2,
        model=model,
        tokenizer=tokenizer,
        with_reasoning=with_reasoning,
        shuffle=True,
        grpo=True,
        max_seq_length=config.train.max_seq_length
    )    

    effective_batch_size = config.train.batch_size*config.train.gradient_accumulation_steps
    logger.info(f"BLUE: Training data length {len(instruct_dataset.dataset['train'])} Validation data length {len(instruct_dataset.dataset['valid'])}" )
    logger.info(f"BLUE: Training steps {(len(instruct_dataset.dataset['train'])/effective_batch_size)*config.train.epochs} ")

    logger.info(f"CYAN: Training Prompt: {instruct_dataset.dataset['train']['prompt'][0]}")

    if checkpoint:
        logger.info(f"Resuming finetuning from checkpoint {checkpoint}")
        
    # Run finetuning
    if use_sloth:
        model_id = model.unsloth_model_ids[model_name]
        model = grpo_unsloth(
            model_name=model_id, 
            dataset=instruct_dataset.dataset, 
            sft_lora_path=sft,
            config=config, 
            logger=logger,
            w_format=w_format,
            w_syntax=w_syntax,
            w_func=w_func,
            w_cov=w_cov,
            checkpoint=checkpoint, 
            output_dir=output_dir,
            max_workers=max_workers
        )
    else:
        model = grpo(
            model_name=model_id, 
            dataset=instruct_dataset.dataset, 
            sft_lora_path=sft,
            config=config, 
            logger=logger,
            checkpoint=checkpoint, 
            w_format=w_format,
            w_syntax=w_syntax,
            w_func=w_func,
            w_cov=w_cov,
            output_dir=output_dir,
            max_workers=max_workers,
            use_vllm=use_vllm,
            vllm_mode=vllm_mode,
            vllm_server_host=vllm_server_host,
            vllm_server_port=vllm_server_port,
            vllm_server_timeout=vllm_server_timeout,
            vllm_group_port=vllm_group_port,
            vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
            vllm_tensor_parallel_size=vllm_tensor_parallel_size,
            vllm_max_model_length=vllm_max_model_length,
        )
    
    logger.info(f"BLUE: DONE Finetuning, Model is saved at {output_dir}" )

    # test finetuned model here
    model = get_model(
        model_name, 
        load_model=True,
        lora_path=os.path.join(output_dir, f"{model_id}_grpo_finetuned"), 
        merge_lora=False,
        use_sloth=use_sloth,
        max_seq_length=config.train.max_seq_length
    )
    
    eval_output_dir = os.path.join(output_dir, "eval")
    os.makedirs(eval_output_dir, exist_ok=True)

    evaluator = Evaluator(
        output_dir=output_dir, 
        batch_size=config.test.batch_size
    )
    wrapped_dataset = HFDatasetWrapper(instruct_dataset.dataset['test'])
    dataloader = DataLoader(wrapped_dataset , batch_size=config.test.batch_size, num_workers=0)
    
    result = evaluator.evaluate(
        dataloader=dataloader,
        model=model,
        num_samples=config.test.num_samples,
        k_list=k,
        temperature=0.1,
        logger=logger,
        debug=not generate_only,
    )

    if generate_only:
        logger.info(
            f"CYAN: Generation-only mode completed. Answers saved under {evaluator.output_dir}"
        )
        return

    plot_cov_pass_each_k(
        result["pass_at_k_by_problem"], 
        result["cov_at_k_by_problem"],
        k_list=k,
        save_dir=evaluator.results_dir,
        prefix=f"cov_pass",
        autoscale_ylim=False  # lets values exceed 1 if they do
    )
    
    write_json(
        n=config.test.num_samples,
        k_list=k,
        pass_at_k_by_problem=result["pass_at_k_by_problem"],
        cov_at_k_by_problem=result["cov_at_k_by_problem"],
        save_path=os.path.join(evaluator.results_dir, f"result_eval.json"),
    )
   
    for i, k_val in enumerate(k):
        logger.info(
            f"YELLOW: Overall pass@{k_val}_syntax] {result['pass_at_k_by_problem_syntax'][i]}, pass@{k_val}_func] {result['pass_at_k_by_problem_func'][i]} "
        )



if __name__ == "__main__":
    main()
