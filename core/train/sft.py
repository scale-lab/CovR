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

"""Finetune LLMs on an instruction dataset using LoRA
"""
import os 
import gc
import sys
import yaml
import torch
import wandb
# import unsloth
import argparse 
import numpy as np
# from unsloth import FastLanguageModel

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)

from omegaconf import OmegaConf
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, SFTConfig
from accelerate import Accelerator
from huggingface_hub import login
from torch.utils.data import DataLoader
from datetime import datetime


from core.utils import get_logger
from core.train.cfg.config import combine_cfgs
from core.train.utils import write_json,plot_cov_pass_each_k
from core.train.utils import plot_loss
from core.train.instruct_dataset import InstructDataset
from core.train.utils import set_seed, print_trainable_parameters
from core.train.instruct_dataset import HFDatasetWrapper
from core.train.model import get_model
from core.train.evaluator import Evaluator
from core.data_generation.utils import compute_avg_coverage

# from core.train.utils import write_json
# from core.eval.dataset import EvalDataset, HFDatasetWrapper
# from core.eval.evaluator import Evaluator


access_token = os.getenv('HF_ACCESS_TOKEN')
login(token=access_token)


def fine_tune_unsloth(
    model_name, 
    dataset, 
    config, 
    output_dir,
    logger,
    checkpoint=None, 
):
    from unsloth import FastLanguageModel
   
    # Load base model
    compute_dtype = torch.float16  # or torch.float16
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=config.train.max_seq_length,
        # dtype=compute_dtype,
        load_in_4bit=config.bnb.load_in_4bit,
        fast_inference = False, # Enable vLLM fast inference
        max_lora_rank = config.lora.lora_r,
        token=os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_ACCESS_TOKEN")
    )

    print_trainable_parameters(model, logger)
    print(model)

    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora.lora_r,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        bias="none",
        use_gradient_checkpointing=config.train.gradient_checkpointing,
        random_state=config.seed,
        use_rslora=False,
        loftq_config=None,
    )
    
    print_trainable_parameters(model, logger)
    print(model)
    
    if torch.cuda.device_count() > 1: # If more than 1 GPU
        model.is_parallelizable = True
        model.model_parallel = True
    
    # call model before finetuning 
    # test_model(model, tokenizer, dataset['test'][0:20])
   
    model.config.use_cache = False
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id


    if config.train.gradient_checkpointing: 
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    else:
        model.gradient_checkpointing_disable()

    training_arguments = SFTConfig(
            per_device_train_batch_size=config.train.batch_size,
            per_device_eval_batch_size=config.train.batch_size,
            gradient_accumulation_steps=config.train.gradient_accumulation_steps, # make finetuing 7 times slower !
            num_train_epochs=config.train.epochs,
            warmup_steps=config.train.warmup_steps,
            dataloader_num_workers=config.train.dataloader_num_workers,
            max_steps=config.train.max_steps,
            learning_rate=config.train.learning_rate,
            lr_scheduler_type=config.train.lr_scheduler_type,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
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
            gradient_checkpointing=config.train.gradient_checkpointing,  # Leads to reduction in memory at slighly decrease in speed
            # gradient_checkpointing_kwargs={"use_reentrant": True},
            dataset_text_field="prompt",
            completion_only_loss=config.train.completion_only_loss,
            save_only_model=True,
            max_length=config.train.max_seq_length,
            eos_token=tokenizer.eos_token,
            pad_token=tokenizer.pad_token,
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
        # packing=True,
        args=training_arguments,
    )

    # Train model
    if checkpoint: 
        trainer.train(checkpoint)
    else:
        trainer.train()
    
    save_path = os.path.join(output_dir, f"{model_name}_finetuned")
    trainer.model.save_pretrained(save_path)
    
    plot_loss(trainer.state.log_history, os.path.join(output_dir, "loss.png"))

    # test the finetuned model on test set
    FastLanguageModel.for_inference(model)

    return model, trainer



def fine_tune(
    model_name: str, 
    dataset, 
    config, 
    logger,
    checkpoint=None,
    output_dir="./results"
):
   
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        token=access_token, 
        use_fast=config.tokenizer.use_fast,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token # end-of-sequence token
    tokenizer.padding_side = config.tokenizer.padding_side # pad to have max seq length

    # Load model in 4-bits and half precision
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.bnb.load_in_4bit, # Activates 4-bit precision loading
        bnb_4bit_quant_type=config.bnb.bnb_4bit_quant_type, # nf4
        bnb_4bit_compute_dtype=compute_dtype, 
        bnb_4bit_use_double_quant=config.bnb.bnb_4bit_use_double_quant, # False
    )

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=access_token,
        quantization_config=bnb_config,
        device_map={"": Accelerator().process_index},
        # attn_implementation="flash_attention_2",
    )
    # model.config.pretraining_tp = 1
    # model.config.eos_token_id = tokenizer.eos_token_id
    # model.config.pad_token_id = tokenizer.pad_token_id

    print_trainable_parameters(model, logger)
    print(model)
    
    # Load LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        r=config.lora.lora_r,
        use_dora=config.lora.use_dora,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    
    if torch.cuda.device_count() > 1: # If more than 1 GPU
        model.is_parallelizable = True
        model.model_parallel = True
    
    # call model before finetuning 
    # test_model(model, tokenizer, dataset['test'][0:20])
   
    model.config.use_cache = False

    if config.train.gradient_checkpointing: 
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    else:
        model.gradient_checkpointing_disable()

    training_arguments = SFTConfig(
        per_device_train_batch_size=config.train.batch_size,
        per_device_eval_batch_size=config.train.batch_size,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps, # make finetuing 7 times slower !
        num_train_epochs=config.train.epochs,
        warmup_steps=config.train.warmup_steps,
        dataloader_num_workers=config.train.dataloader_num_workers,
        max_steps=config.train.max_steps,
        learning_rate=config.train.learning_rate,
        lr_scheduler_type=config.train.lr_scheduler_type,
        bf16=use_bf16,
        fp16=use_fp16,
        tf32=use_bf16,
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
        gradient_checkpointing=config.train.gradient_checkpointing,  # Leads to reduction in memory at slighly decrease in speed
        # gradient_checkpointing_kwargs={"use_reentrant": True},
        dataset_text_field="prompt",
        completion_only_loss=config.train.completion_only_loss,
        max_length=config.train.max_seq_length,
        eos_token=tokenizer.eos_token,
        pad_token=tokenizer.pad_token,
    )
    
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
        args=training_arguments,
        peft_config=peft_config,
    )

    logger.info(
        f"BLUE: LoRA config r={config.lora.lora_r}, alpha={config.lora.lora_alpha}, "
        f"dropout={config.lora.lora_dropout}"
    )
    print_trainable_parameters(trainer.model, logger)

    # Train model
    if checkpoint: 
        trainer.train(checkpoint)
    else:
        trainer.train()
        
    trainer.model.save_pretrained(os.path.join(output_dir, f"{model_name}_finetuned"))
    
    plot_loss(trainer.state.log_history, os.path.join(output_dir, "loss.png"))

    return model, trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help="Path to config file for training", required=False, default="core/train/cfg/experiments/llama3.3_r64.yaml")
    parser.add_argument('--checkpoint', type=str, help="Resume finetuning from checkpoint", required=False, default=None)
    parser.add_argument('--use_sloth', action='store_const', const=True, help="Use sloth models for faster finetuning", default=False)
    parser.add_argument('--run_name', type=str, help="Name of the run", default=None)
    parser.add_argument("--k", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument('--output_dir', type=str, help="Path to output directory ", required=False, default="./output/sft")
    parser.add_argument("--with_reasoning", action="store_true", help="to use reasoning in finetuning")
    parser.add_argument("--completion_only_loss", action="store_true", default=None, help="Train only on completion tokens using prompt/completion dataset columns")
    parser.add_argument("--use_vllm", action="store_true", help="to use VLLM in evaluation", default=False)
    parser.add_argument('--multi_gpu', action='store_const', const=True, help="Use Multiple GPUs: fully sharded data parallel mode (You have to launch this with accelerate!)", default=False)
    parser.add_argument("--debug", dest="debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--max_workers", type=int, help="max workers for parallel simulation", default=3)
    parser.add_argument("--generate_only", action=argparse.BooleanOptionalAction, default=True, help="Only generate answers JSONL and skip Synopsys/VCS evaluation.")

    args, cli_overrides = parser.parse_known_args()
    config_file = args.config

    config = combine_cfgs(config_file, cli_overrides=cli_overrides)

    checkpoint = args.checkpoint
    k = args.k
    use_sloth = args.use_sloth
    run_name = args.run_name
    output_dir = args.output_dir
    multi_gpu = args.multi_gpu
    use_vllm = args.use_vllm
    debug = args.debug 
    max_workers = args.max_workers
    generate_only = args.generate_only

    if generate_only:
        debug = False

    print(f"Finetuning reasoning flag value {args.with_reasoning}")
    if args.completion_only_loss is not None:
        config.train.completion_only_loss = args.completion_only_loss

    if run_name:
        config.train.run_name = run_name
    
    shuffle = config.DATASET.SHUFFLE 
    model_name = config.LLM.NAME 

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = os.path.join(output_dir, f"{config.train.run_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    logfile = os.path.join(output_dir, f"{model_name}")
    logger = get_logger(output=logfile)
    
    logger.info(f"BLUE: Number of GPUs {torch.cuda.device_count()}")
    logger.info(f"BLUE: Finetuning {model_name}")
   
    set_seed(config.seed, n_gpu=torch.cuda.device_count())

    with open(os.path.join(output_dir, "config.yaml"), 'w') as f:
        f.write(config.dump())

    model = get_model(model_name, load_model=False)
    model_id = model.model_ids[model_name]

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, 
        token=access_token, 
    ) 
    
    tokenizer.pad_token = tokenizer.eos_token # end-of-sequence token
    tokenizer.padding_side = config.tokenizer.padding_side # pad to have max seq length
    
    train_json =  config.DATASET.TRAIN 
    test_veval_json = config.DATASET.TEST_VEVAL 
    test_cvdp_json = config.DATASET.TEST_CVDP

    instruct_dataset = InstructDataset(
        train_json=train_json,     
        test_json=test_veval_json,
        train_split=0.8,
        valid_split=0.2,
        model=model,
        tokenizer=tokenizer,
        shuffle=True,
        with_reasoning=args.with_reasoning,
        max_seq_length=config.train.max_seq_length,
        completion_only_loss=config.train.completion_only_loss
    )    

    effective_batch_size = config.train.batch_size*config.train.gradient_accumulation_steps
    logger.info(f"CYAN: Training Prompt: {instruct_dataset.dataset['train']['prompt'][0]}")
    logger.info(f"BLUE: Training data length {len(instruct_dataset.dataset['train'])} Validation data length {len(instruct_dataset.dataset['valid'])}" )
    logger.info(f"BLUE: Training steps {(len(instruct_dataset.dataset['train'])/effective_batch_size)*config.train.epochs} ")
    logger.info(f"MAGENTA: Test Prompt: {instruct_dataset.dataset['test']['prompt'][0]}")

    if checkpoint:
        logger.info(f"Resuming finetuning from checkpoint {checkpoint}")
    
    # Run finetuning
    if use_sloth:
        model_id = model.unsloth_model_ids[model_name]
        model, trainer = fine_tune_unsloth(
            model_name=model_id, 
            dataset=instruct_dataset.dataset, 
            config=config, 
            logger=logger,
            checkpoint=checkpoint, 
            output_dir=output_dir
        )
    else:
        model, trainer = fine_tune(
            model_name=model_id, 
            dataset=instruct_dataset.dataset, 
            config=config, 
            logger=logger,
            checkpoint=checkpoint, 
            output_dir=output_dir
    )
    
    logger.info(f"BLUE: DONE Finetuning, Model is saved at {output_dir}" )

    if use_vllm: 
        del trainer
        del model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    model = get_model(
        model_name, 
        load_model=True,
        lora_path=os.path.join(output_dir, f"{model_id}_finetuned"), 
        merge_lora=False,
        use_sloth=use_sloth,
        max_seq_length=config.test.max_seq_length,
        max_new_tokens=config.test.max_new_tokens,
        use_vllm=use_vllm,
        gpu_memory_utilization=0.85
    )
    
    if not use_vllm: 
        model.model.eval()
        model.model.config.use_cache = True

    for test_set_name in ["test_veval", "test_cvdp"]:
        logger.info(f"Evaluating on {test_set_name} dataset")
        avg_pass_at_k_syntax = [0.0] * len(k)
        avg_pass_at_k_func = [0.0] * len(k)
        avg_cov_at_k = []
        
        if test_set_name == "test_veval":
            test_json = test_veval_json
        else:
            test_json = test_cvdp_json

        eval_output_dir = os.path.join(output_dir, "eval")
        os.makedirs(eval_output_dir, exist_ok=True)
        
        evaluator = Evaluator(
            output_dir=os.path.join(output_dir, test_set_name), 
            batch_size=config.test.batch_size,
            max_workers=max_workers
        )
        
        instruct_dataset = InstructDataset(
            train_json=None,     
            test_json=test_json,
            train_split=0.8,
            valid_split=0.2,
            model=model,
            tokenizer=tokenizer,
            shuffle=True,
            with_reasoning=args.with_reasoning,
            max_seq_length=config.train.max_seq_length
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
            debug=debug
        )

        if generate_only:
            logger.info(
                f"CYAN: Generation-only mode completed for {test_set_name}. "
                f"Answers saved under {evaluator.output_dir}"
            )
            continue

        if not debug: 
            all_pass_at_k_by_problem = []
            all_cov_at_k_by_problem = []

            world_size = model.accelerator.num_processes if multi_gpu else 1
            for r in range(world_size):
                p = os.path.join(evaluator.output_dir, f"answers_rank{r}.jsonl")
                if os.path.exists(p):
                    pass_p, cov_p = evaluator.evaluate_json(p, k_list=k, logger=logger)
                    all_pass_at_k_by_problem.extend(pass_p)
                    all_cov_at_k_by_problem.extend(cov_p)
        else:
            all_pass_at_k_by_problem = result['pass_at_k_by_problem']
            all_cov_at_k_by_problem = result['cov_at_k_by_problem']

        plot_cov_pass_each_k(
            all_pass_at_k_by_problem, 
            all_cov_at_k_by_problem,
            k_list=k,
            save_dir=evaluator.results_dir,
            prefix=f"cov_pass_{test_set_name}",
            autoscale_ylim=False  # lets values exceed 1 if they do
        )
        
        write_json(
            n=config.test.num_samples,
            k_list=k,
            pass_at_k_by_problem=all_pass_at_k_by_problem,
            cov_at_k_by_problem=all_cov_at_k_by_problem,
            save_path=os.path.join(evaluator.results_dir, f"result_eval_{test_set_name}.json"),
        )

        num_problems = len(all_pass_at_k_by_problem)
        avg_pass_at_k_syntax = [
            sum(p["pass@k_syntax"][i] for p in all_pass_at_k_by_problem) / num_problems
            for i in range(len(k))
        ]
        avg_pass_at_k_func = [
            sum(p["pass@k_func"][i] for p in all_pass_at_k_by_problem) / num_problems
            for i in range(len(k))
        ]

        avg_cov_at_k = [
            sum((compute_avg_coverage(p["cov@k"][i]) or 0.0) for p in all_cov_at_k_by_problem) / num_problems
            for i in range(len(k))
        ]

        metrics = {}
        for i, k_val in enumerate(k):
            logger.info(
                f"YELLOW: Overall {test_set_name} pass@{k_val}_syntax] {avg_pass_at_k_syntax[i]}, pass@{k_val}_func] {avg_pass_at_k_func[i]} "
            )

            logger.info(
                f"YELLOW: Overall {test_set_name} cov@{k_val}] {avg_cov_at_k[i]}"
            )

            metrics[f"eval/{test_set_name}/pass@{k_val}_syntax"] = avg_pass_at_k_syntax[i]
            metrics[f"eval/{test_set_name}/pass@{k_val}_func"] = avg_pass_at_k_func[i]
            metrics[f"eval/{test_set_name}/cov@{k_val}"] = avg_cov_at_k[i]

        if use_vllm:
            if wandb.run is not None:
                wandb.log(metrics)
        else:
            trainer.log(metrics)


if __name__ == "__main__":
    main()
