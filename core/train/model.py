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

import os
import re
import sys
import json
import argparse
from typing import List, Optional

try:
    pass
    # import unsloth  
    # from unsloth import FastLanguageModel
except:
    FastLanguageModel = None

import torch
import openai
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from openai import OpenAI
from peft import PeftModel, PeftConfig
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader
from accelerate.utils import gather_object
try: 
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
except: 
    pass 
    
from core.utils import get_logger
from core.train.prompts import SYS_PROMPT, USER_PROMPT_SECTIONED, LLM4_COV_SYSTEM_PROMPT, LLM4COV_USER_TEMPLATE

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _resolve_max_new_tokens(max_seq_length: int, max_new_tokens: Optional[int]) -> int:
    """Preserve the legacy behavior unless a separate generation cap is provided."""
    return max_seq_length if max_new_tokens is None else max_new_tokens


class GPTModel:
    names = [
        'gpt-3.5-turbo',
        'gpt-3.5-turbo-0125',
        'gpt-4-turbo-2024-04-09',
        'gpt-4o-2024-05-13',
        'gpt-4o-mini-2024-07-18',
        'gpt-4o-mini',
        'gpt-4-turbo'
    ]

    def __init__(self, name):
        self.name = name
        self.tokenizer = None
        api_key = os.environ.get('OPENAI_KEY')
        self.client = OpenAI(api_key=api_key)

    def generate_chat_template(self, system_prompt: str, user_prompt: str, assistant_answer: str = "", tokenizer=None, add_generation_prompt: bool = True):
        messages = []

        if system_prompt:
            messages.append({
                'role': 'system',
                'content': system_prompt
            })

        if assistant_answer:
            messages.append({
                'role': 'assistant',
                'content': assistant_answer
            })

        messages.append({
            'role': 'user',
            'content': user_prompt
        })

        return messages

    def message(self, system_prompt, user_prompt, temperature=0.4, num_samples=1):
        answers = []
        prompt = self.generate_chat_template(
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )
        try:
            response = self.client.chat.completions.create(
                model=self.name,
                messages=prompt,
                temperature=temperature,
                n=num_samples,
                seed=0
            )

            for choice in response.choices:
                answers.append(choice.message.content)
            return answers
        except openai.OpenAIError as e:
            print(f"OpenAIError: {e}.", flush=True)
            sys.exit(0)

    def fix_prompt_format(self, sys_prompt, user_prompt):
        fixed = []
        fixed.append({
            "role": 'system',
            "content": sys_prompt
        })
        fixed.append({
            "role": 'user',
            "content": user_prompt
        })
        return fixed

    def message_batch(self, prompts, temperature=0.4, num_samples=1):
        all_answers = []
        for sys_prompt, user_prompt in zip(prompts[0]['content'], prompts[1]['content']):
            try:
                prompt = self.fix_prompt_format(
                    sys_prompt,
                    user_prompt
                )
                response = self.client.chat.completions.create(
                    model=self.name,
                    messages=prompt,
                    temperature=temperature,
                    n=num_samples,
                    seed=0
                )
                answers = [choice.message.content for choice in response.choices]
                all_answers.extend(answers)
            except openai.OpenAIError as e:
                print(f"OpenAIError for prompt {prompt}: {e}.", flush=True)
                all_answers.append([None] * num_samples)
        return all_answers


class DeepSeekModel():

    model_ids = {
        "deepseek-r1": "deepseek-reasoner",
        "deepseek-v3": "deepseek-chat"
    }

    def __init__(self, name):
        super().__init__()
        self.tokenizer = None
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.name = DeepSeekModel.model_ids[name]

    def generate_chat_template(self, system_prompt: str, user_prompt: str, assistant_answer: str = "", tokenizer=None, add_generation_prompt: bool = True):
        messages = []
        if system_prompt:
            messages = [{
                'role': 'system',
                'content': system_prompt
            }]

        messages.append({'role': 'user', 'content': user_prompt})

        return messages

    def message(self, system_prompt, user_prompt, temperature=0.1, num_samples=1):
        answers = []
        prompt = self.generate_chat_template(
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )
        response = self.client.chat.completions.create(
            model=self.name,
            messages=prompt,
            temperature=temperature,
            n=num_samples,
        )

        for choice in response.choices:
            answers.append(choice.message.content)
        return answers

    def fix_prompt_format(self, sys_prompt, user_prompt):
        fixed = []
        fixed.append({
            "role": 'system',
            "content": sys_prompt
        })
        fixed.append({
            "role": 'user',
            "content": user_prompt
        })
        return fixed

    def message_batch(self, prompts, temperature=0.4, num_samples=1):
        all_answers = []
        for sys_prompt, user_prompt in zip(prompts[0]['content'], prompts[1]['content']):
            try:
                prompt = self.fix_prompt_format(
                    sys_prompt,
                    user_prompt
                )
                answers = []
                for _ in range(num_samples):
                    response = self.client.chat.completions.create(
                        model=self.name,
                        messages=prompt,
                        temperature=temperature,
                        seed=0
                    )
                    answers.append(response.choices[0].message.content)
                all_answers.extend(answers)
            except openai.OpenAIError as e:
                print(f"OpenAIError for prompt {prompt}: {e}.", flush=True)
                all_answers.append([None] * num_samples)
        
        return all_answers


class HFModel:

    model_instruct_ids = {
        "mistral": "mistralai/Mistral-7B-Instruct-v0.1",
        "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "llama3:8b": "meta-llama/Meta-Llama-3-8B-Instruct",
        "llama3:70b": "meta-llama/Meta-Llama-3-70B-Instruct",
        "llama3.1": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "llama3.1:70b": "meta-llama/Llama-3.1-70B-Instruct",
        "llama3.1:8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "llama3.3:70b": "meta-llama/Llama-3.3-70B-Instruct",
        "codellama:70b": "meta-llama/CodeLlama-70b-Instruct-hf",
        "qwen2.5-coder": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "qwen2.5-coder-covr-r1024": "manarabdelatty/qwen2.5-coder-covr-target-module-r1024-merged",
        "qwen4b-instruct-covr-r1024-msl-6028": "manarabdelatty/qwen4b-instruct-covr-completion-only-loss-r1024-msl-6028",
        "qwen4b-instruct-covr-r1024-msl-8028": "manarabdelatty/qwen4b-instruct-covr-completion-only-loss-r1024-msl-8028",
        "qwen2.5-coder-covr-comp-only-r1024": "manarabdelatty/qwen2.5-coder-covr-completion-only-loss-r1024-merged",
        "qwen2.5-coder-covr-comp-only-r1024-msl-6028": "manarabdelatty/qwen2.5-coder-covr-completion-only-loss-r1024-merged_msl_6028",
        "qwen2.5-coder-covr-comp-only-r1024-msl-8028": "manarabdelatty/qwen2.5-coder-covr-completion-only-loss-r1024-merged_msl_8028",
        "codegen-16b-verilog": "shailja/fine-tuned-codegen-16B-Verilog",
        "codegen-6b-verilog": "shailja/fine-tuned-codegen-6B-Verilog",
        "rtlcoder-deepseek": "ishorn5/RTLCoder-Deepseek-v1.1",
        "rtlcoder-mistral": "ishorn5/RTLCoder-v1.1",
        "verithoughts-r:14b": "nyu-dice-lab/VeriThoughts-Reasoning-14B",
        "verithoughts-r:7b": "nyu-dice-lab/VeriThoughts-Reasoning-7B",
        "verithoughts-i:7b": "nyu-dice-lab/VeriThoughts-Instruct-7B",
        "qwen2.5-instruct": "Qwen/Qwen2.5-7B-Instruct" ,
        "qwen4b-instruct": "Qwen/Qwen3-4B-Instruct-2507",
        "qwen3-8b": "Qwen/Qwen3-8B",
        "qwen3.5-4b": "Qwen/Qwen3.5-4B",
        "qwen4b-thinking": "Qwen/Qwen3-4B-Thinking-2507",
        "qwen8b-instruct": "Qwen/Qwen3-8B",
        "deepseek-coder-33b": "deepseek-ai/deepseek-coder-33b-instruct",
        "starcoder2:15b": "bigcode/starcoder2-15b-instruct-v0.1",
        "llm4cov_stage2": "hez2024/LLM4Cov-Qwen3-4B-SFT-Stage2",
        "llm4cov_stage1": "hez2024/LLM4Cov-Qwen3-4B-SFT-Stage1",
        "llm4cov_stage0": "hez2024/LLM4Cov-Qwen3-4B-SFT-Stage0"
    }

    def __init__(self, use_accelerate=False, use_sloth=False):
        self.access_token = os.environ.get('HF_ACCESS_TOKEN')
        self.cache_dir = os.environ.get('HF_CACHE_DIR')
        self.accelerator = Accelerator() if use_accelerate else None
        self.use_sloth = use_sloth

    def load(self, model_id, add_special_tokens=True, lora_path=None, merge_lora=False,  device_map='cuda', max_seq_length=1048):
        
        if self.use_sloth:
            if FastLanguageModel is None:
                raise ImportError(
                    "Unsloth is required when use_sloth=True. "
                    "Install unsloth in this environment or run without --use_sloth."
                )

            use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
            if lora_path: 
                model_id = lora_path 

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_id,
                max_seq_length=max_seq_length,
                dtype=compute_dtype,
                load_in_4bit=True,
                fast_inference = True, # Enable vLLM fast inference
                # enforce_eager=True,  # important
                # compilation_config=0,
                token=os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_ACCESS_TOKEN")
            )

            # if self.accelerator is not None:
            #     model = self.accelerator.prepare(model)
        else: 
            use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )

            effective_device_map = None if self.accelerator is not None else device_map

            base_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map=effective_device_map, 
                token=self.access_token,
                quantization_config=bnb_config,
                cache_dir=self.cache_dir
            )

            if lora_path is None:
                model = base_model
            else:
                print("[LoRA] Loading lora from : ", lora_path)
                model = PeftModel.from_pretrained(base_model, lora_path, device_map=device_map)

                # if merge_lora:
                #     model = model.merge_and_unload() 

            if self.accelerator is not None:
                # this replicates the model across GPUs / handles device placement
                model = self.accelerator.prepare(model)

            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                token=self.access_token,
                cache_dir=self.cache_dir,
            )
            if add_special_tokens:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.padding_side = "left"
        
        self.model = model
        self.tokenizer = tokenizer

        return model, tokenizer

    def apply_chat_template(self, tokenizer, messages, add_generation_prompt=True):
        # padding_side_default = tokenizer.padding_side
        # tokenizer.padding_side = "left"

        encodeds = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt
        )

        # tokenizer.padding_side = padding_side_default
        return encodeds

    def call(self, prompt: List[dict], max_tokens: int = 1124, top_k: int = 50, top_p: int = 1, temperature: float = 0.1, terminators=None, num_return_sequences: bool = 1):

        input_ids = self.tokenizer(
            prompt,
            padding='longest',
            add_special_tokens=True,
            return_tensors="pt"
        )
        
        if self.accelerator is not None and not self.use_sloth:
            model_to_use = self.model.module
            # .module
            input_ids = {k: v.to(self.accelerator.device) for k, v in input_ids.items()}
        else:
            model_to_use = self.model
            device = next(self.model.parameters()).device
            input_ids = input_ids.to(device)
            # input_ids = input_ids.to(self.model.device)

        with torch.no_grad():
            if terminators:
                outputs = model_to_use.generate(
                    **input_ids,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    eos_token_id=terminators,
                    num_return_sequences=num_return_sequences
                )
            else:
                outputs = model_to_use.generate(
                    **input_ids,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    num_return_sequences=num_return_sequences
                )

        # if self.accelerator is not None:
        #     outputs = self.accelerator.gather(outputs)
        #     if not self.accelerator.is_main_process:
        #         return []

        input_len = input_ids["input_ids"].shape[-1]
        generated_ids = outputs[:, input_len:]

        answers = self.tokenizer.batch_decode(
            sequences=generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        return [answer.strip() for answer in answers]


class VLLMModel:
    model_instruct_ids = HFModel.model_instruct_ids

    @staticmethod
    def resolve_model_id(name: str) -> str:
        if name in QwenModel.model_ids:
            return QwenModel.model_ids[name]
        if name in LLM4CovModel.model_ids:
            return LLM4CovModel.model_ids[name]
        supported = sorted(set(QwenModel.model_ids) | set(LLM4CovModel.model_ids))
        raise ValueError(
            f"vLLM backend is only configured for QwenModel and LLM4CovModel names, "
            f"got: {name}. Supported names: {supported}"
        )

    def __init__(
        self,
        name,
        lora_path=None,
        max_seq_length=1048,
        max_new_tokens=None,
        gpu_memory_utilization=0.9,
    ):
        self.name = name
        self.model_id = self.resolve_model_id(name)
        self.lora_path = lora_path
        self.max_seq_length = max_seq_length
        self.max_new_tokens = _resolve_max_new_tokens(max_seq_length, max_new_tokens)
        self.lora_request = None
        max_lora_rank = 16

        if self.lora_path:
            adapter_config = os.path.join(self.lora_path, "adapter_config.json")
            if os.path.exists(adapter_config):
                with open(adapter_config, "r", encoding="utf-8") as f:
                    max_lora_rank = int(json.load(f).get("r", 16))
            self.lora_request = LoRARequest(
                lora_name=f"{name}_sft",
                lora_int_id=1,
                lora_path=self.lora_path,
                base_model_name=self.model_id,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            token=os.environ.get("HF_ACCESS_TOKEN"),
        )

        llm_kwargs = {
            "model": self.model_id,
            "dtype": "bfloat16",
            "max_model_len": max_seq_length,
            "quantization": "bitsandbytes",
            "load_format": "bitsandbytes",
            "gpu_memory_utilization": gpu_memory_utilization,
            "enable_prefix_caching": True,
            "trust_remote_code": False,
        }
        if self.lora_request is not None:
            llm_kwargs.update({
                "enable_lora": True,
                "max_lora_rank": max_lora_rank,
            })

        print(
            f"[vLLM LoRA] enabled={self.lora_request is not None}, "
            f"path={self.lora_path}, rank={max_lora_rank}",
            flush=True,
        )

        self.llm = LLM(**llm_kwargs)

    def apply_chat_template(self, tokenizer, messages, add_generation_prompt=True):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    def generate_chat_template(
        self,
        tokenizer,
        system_prompt: str,
        user_prompt: str,
        assistant_answer: str = "",
        add_generation_prompt: bool = True,
    ):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": user_prompt})

        if assistant_answer:
            messages.append({"role": "assistant", "content": assistant_answer})

        return self.apply_chat_template(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            tokenizer=tokenizer,
        )


    def message_batch(self, prompts, top_k = 50, top_p = 1, temperature=0.1, num_samples=1):
        # Manually truncate prompts to avoid exceeding max_seq_length
        max_prompt_tokens = self.max_seq_length - self.max_new_tokens - 50
        truncated_prompts = []
        for prompt in prompts:
            tokens = self.tokenizer.encode(prompt)
            if len(tokens) > max_prompt_tokens:
                truncated_tokens = tokens[:max_prompt_tokens]
                truncated_prompt = self.tokenizer.decode(truncated_tokens)
                truncated_prompts.append(truncated_prompt)
            else:
                truncated_prompts.append(prompt)
        
        sampling_params = SamplingParams(
            n=num_samples,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=self.max_new_tokens,
            seed=0
        )

        outputs = self.llm.generate(
            truncated_prompts,
            sampling_params,
            lora_request=self.lora_request,
        )

        answers = []
        for output in outputs:
            for completion in output.outputs:
                answers.append(completion.text.strip())

        return answers

    def message(self, system_prompt, user_prompt, temperature=0.1, num_samples=1):
        prompt = self.generate_chat_template(
            tokenizer=self.tokenizer,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        answers = self.message_batch(
            prompts=[prompt],
            top_k=50,
            top_p=1,
            temperature=temperature,
            num_samples=num_samples,
        )
        return answers if num_samples > 1 else answers[0]


class LLM4CovModel(HFModel):
    model_ids = {
        "llm4cov_stage0": "hez2024/LLM4Cov-Qwen3-4B-SFT-Stage0",
        "llm4cov_stage1": "hez2024/LLM4Cov-Qwen3-4B-SFT-Stage1",
        "llm4cov_stage2": "hez2024/LLM4Cov-Qwen3-4B-SFT-Stage2",
        "qwen3_4b_base": "Qwen/Qwen3-4B-Instruct-2507",
    }

    def __init__(self, name, load_model=True, lora_path=None, merge_lora=False, max_seq_length=3028, max_new_tokens=None):
        super().__init__()

        self.model = None
        self.tokenizer = None

        if load_model:
            self.model, self.tokenizer = self.load(
                model_id=LLM4CovModel.model_ids[name],
                add_special_tokens=True,
                lora_path=lora_path,
                merge_lora=merge_lora,
                max_seq_length=max_seq_length
            )

        self.max_seq_length = max_seq_length
        self.max_new_tokens = _resolve_max_new_tokens(max_seq_length, max_new_tokens)

    def generate_chat_template(
        self,
        tokenizer,
        system_prompt: str,
        user_prompt: str,
        assistant_answer: str = "",
        add_generation_prompt: bool = True,
    ):
        messages = []
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })

        messages.append({
            "role": "user",
            "content": user_prompt
        })

        if assistant_answer:
            messages.append({
                "role": "assistant",
                "content": assistant_answer
            })

        qwen_template = self.apply_chat_template(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            tokenizer=tokenizer
        )

        return qwen_template

    def message(self, system_prompt, user_prompt, temperature=0.1, num_samples=1):
        prompt = self.generate_chat_template(
            tokenizer=self.tokenizer,
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )

        answers = self.call(
            prompt=prompt,
            max_tokens=self.max_new_tokens,
            top_k=50,
            top_p=1.0,
            temperature=temperature,
            terminators=None,
            num_return_sequences=num_samples
        )

        batch_size = len(prompt) if isinstance(prompt, list) else 1
        batch_responses = []
        for i in range(0, num_samples * batch_size, num_samples):
            response_parsed = [answer for answer in answers[i:i + num_samples]]
            batch_responses.append(response_parsed)

        return batch_responses

    def message_batch(self, prompts, temperature=0.1, num_samples=1):
        answers = self.call(
            prompt=prompts,
            max_tokens=self.max_new_tokens,
            top_k=50,
            top_p=1.0,
            temperature=temperature,
            terminators=None,
            num_return_sequences=num_samples
        )
        return [answer for answer in answers]



class LlamaModel(HFModel):

    model_ids = {
        "llama3.1:8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "llama3:8b":   "meta-llama/Meta-Llama-3-8B-Instruct",
        "llama3.1:70b":"meta-llama/Llama-3.1-70B-Instruct",
        "llama3.3:70b":"meta-llama/Llama-3.3-70B-Instruct"
    }

    unsloth_model_ids = {
        "llama3.1:8b": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        "llama3.1:70b":"unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit"
    }

    def __init__(self, name, load_model=True, lora_path=None, merge_lora=False, use_sloth=False, multi_gpu=False, max_seq_length=1048, max_new_tokens=None):
        super().__init__(use_sloth=use_sloth, use_accelerate=multi_gpu)

        self.model = None
        self.tokenizer = None
        if load_model:
            self.model, self.tokenizer = self.load(
                model_id=LlamaModel.model_ids[name],
                add_special_tokens=True,
                lora_path=lora_path,
                merge_lora=merge_lora,
                max_seq_length=max_seq_length
            )

        self.max_seq_length = max_seq_length
        self.max_new_tokens = _resolve_max_new_tokens(max_seq_length, max_new_tokens)

    def _parse_answer(self, answer):
        pattern = r'(?:.*assistant)([\s\S]*)'
        match = re.findall(pattern, answer, re.DOTALL)
        if match:
            result = match[-1]
        else:
            result = None
        return result

    def generate_chat_template(self, tokenizer, system_prompt: str, user_prompt: str, assistant_answer: str = "", add_generation_prompt: bool = True) -> List[dict]:
        messages = []
        if system_prompt:
            messages = [{"role": "assistant", "content": system_prompt}]

        messages.append({"role": "user", "content": user_prompt})

        if assistant_answer:
            messages.append({"role": "assistant", "content": assistant_answer})

        llama_template = self.apply_chat_template(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            tokenizer=tokenizer
        )

        return llama_template

    def message(self, system_prompt, user_prompt, temperature=0.1, num_samples=1):

        prompt = self.generate_chat_template(
            tokenizer=self.tokenizer,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        # TODO: FIX ME
        prompt = [prompt]
        answers = self.call(
            prompt=prompt,
            max_tokens=self.max_new_tokens,
            top_k=50,
            top_p=1,
            temperature=temperature,
            terminators=None,
            num_return_sequences=num_samples
        )

        batch_size = len(prompt)
        batch_responses = []

        for i in range(0, num_samples * batch_size, num_samples):
            response_parsed = [self._parse_answer(answer) for answer in answers[i:i + num_samples]]
            batch_responses.append(response_parsed)

        return batch_responses[0]  # TODO: FIX ME

    def message_batch(self, prompts, temperature=0.1, num_samples=1):
        answers = self.call(
            prompt=prompts,
            max_tokens=self.max_new_tokens,
            top_k=50,
            top_p=1,
            temperature=temperature,
            terminators=None,
            num_return_sequences=num_samples
        )

        batch_size = len(prompts)
        batch_responses = []

        for i in range(0, num_samples * batch_size, num_samples):
            response_parsed = [self._parse_answer(answer) for answer in answers[i:i + num_samples]]
            batch_responses.append(response_parsed if num_samples > 1 else response_parsed[0])

        flat_responses = []
        for responses in batch_responses:
            if isinstance(responses, list):
                flat_responses.extend(responses)
            else:
                flat_responses.append(responses)

        return flat_responses  # always a flat list of strings


class CodeLlamaModel(HFModel):
    model_ids = {
        "codellama:70b": "codellama/CodeLlama-70b-Instruct-hf",
        "codellama:34b": "codellama/CodeLlama-34b-Instruct-hf"
    }

    def __init__(self, name, load_model=True, lora_path=None, merge_lora=False, use_sloth=False, max_seq_length=4048, max_new_tokens=None):
        super().__init__()
        self.model = None
        self.tokenizer = None
        if load_model:
            self.model, self.tokenizer = self.load(
                model_id=CodeLlamaModel.model_ids[name],
                add_special_tokens=True,
                lora_path=lora_path,
                merge_lora=merge_lora,
                max_seq_length=max_seq_length
            )
        self.max_seq_length = max_seq_length
        self.max_new_tokens = _resolve_max_new_tokens(max_seq_length, max_new_tokens)

    def _parse_answer(self, answer):
        pattern = r'(?:.*assistant)([\s\S]*)'
        match = re.findall(pattern, answer, re.DOTALL)
        return match[-1] if match else None

    def generate_chat_template(
        self,
        tokenizer,
        system_prompt: str,
        user_prompt: str,
        assistant_answer: str = "",
        add_generation_prompt: bool = True,
    ):
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": user_prompt})

        if assistant_answer:
            messages.append({"role": "assistant", "content": assistant_answer})

        return self.apply_chat_template(
            messages=messages,
            add_generation_prompt=False,
            tokenizer=tokenizer
        )

    def message(self, system_prompt, user_prompt, temperature=0.1, num_samples=1):
        prompt = self.generate_chat_template(
            tokenizer=self.tokenizer,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        prompt = [prompt]
        answers = self.call(
            prompt=prompt,
            max_tokens=self.max_new_tokens,
            top_k=50,
            top_p=1,
            temperature=temperature,
            terminators=None,
            num_return_sequences=num_samples
        )

        response_parsed = [self._parse_answer(ans) for ans in answers[:num_samples]]
        return response_parsed if num_samples > 1 else response_parsed[0]

    def message_batch(self, prompts, temperature=0.1, num_samples=1):
        answers = self.call(
            prompt=prompts,
            max_tokens=self.max_new_tokens,
            top_k=50,
            top_p=1,
            temperature=temperature,
            terminators=None,
            num_return_sequences=num_samples
        )

        batch_size = len(prompts)
        batch_responses = []
        for i in range(0, num_samples * batch_size, num_samples):
            parsed = [self._parse_answer(ans) for ans in answers[i:i + num_samples]]
            batch_responses.append(parsed if num_samples > 1 else parsed[0])

        flat_responses = []
        for response in batch_responses:
            if isinstance(response, list):
                flat_responses.extend(response)
            else:
                flat_responses.append(response)

        return flat_responses


class DeepSeekCoderModel(HFModel):
    model_ids = {
        "deepseek-coder-33b": "deepseek-ai/deepseek-coder-33b-instruct"
    }

    def __init__(self, name, load_model=True, lora_path = None, merge_lora=False, use_sloth=False, max_seq_length=4048, max_new_tokens=None):
        super().__init__()
        self.model = None
        self.tokenizer = None
        if load_model:
            self.model, self.tokenizer = self.load(
                model_id=self.model_ids[name],
                add_special_tokens=True,
                max_seq_length = max_seq_length
            )

        self.max_seq_length = max_seq_length
        self.max_new_tokens = _resolve_max_new_tokens(max_seq_length, max_new_tokens)

    def generate_chat_template(
        self,
        tokenizer,
        system_prompt: str,
        user_prompt: str,
        assistant_answer: str = "",
        add_generation_prompt: bool = True,
    ):
        # This is not a chat model — treat everything as a single flat prompt string
        return (system_prompt + "\n" + user_prompt).strip()

    def message(self, system_prompt, user_prompt, temperature=0.1, num_samples=1):
        prompt = self.generate_chat_template(
            tokenizer=self.tokenizer,
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )
        prompt = [prompt]  # Wrap for batching
        answers = self.call(
            prompt=prompt,
            max_tokens=self.max_new_tokens,
            top_k=50,
            top_p=1,
            temperature=temperature,
            terminators=None,
            num_return_sequences=num_samples
        )
        parsed = [ans for ans in answers[:num_samples]]
        return parsed if num_samples > 1 else parsed[0]

    def message_batch(self, prompts, temperature=0.1, num_samples=1):
        answers = self.call(
            prompt=prompts,
            max_tokens=self.max_new_tokens,
            top_k=50,
            top_p=1,
            temperature=temperature,
            terminators=None,
            num_return_sequences=num_samples
        )
        batch_size = len(prompts)
        batch_responses = []
        for i in range(0, num_samples * batch_size, num_samples):
            parsed = [ans for ans in answers[i:i + num_samples]]
            batch_responses.append(parsed if num_samples > 1 else parsed[0])

        flat_responses = []
        for responses in batch_responses:
            if isinstance(responses, list):
                flat_responses.extend(responses)
            else:
                flat_responses.append(responses)

        return flat_responses


class MistralModel(HFModel):
    model_ids = {
        "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
        "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    }

    def __init__(self, name, load_model=True, lora_path=None, merge_lora=False, max_seq_length=4028, max_new_tokens=None):
        super().__init__()

        self.model = None
        self.tokenizer = None
        if load_model:
            self.model, self.tokenizer = self.load(
                model_id=MistralModel.model_ids[name],
                add_special_tokens=True,
                lora_path=lora_path,
                merge_lora=merge_lora
            )
        self.max_seq_length = max_seq_length
        self.max_new_tokens = _resolve_max_new_tokens(max_seq_length, max_new_tokens)

    def generate_chat_template(self, tokenizer, system_prompt: str, user_prompt: str, assistant_answer: str = "", add_generation_prompt: bool = True) -> List[dict]:
        combined_prompt = user_prompt
        if system_prompt:
            combined_prompt = system_prompt + '\n\n' + user_prompt

        messages = [{
            'role': 'user',
            'content': combined_prompt
        }]

        if assistant_answer:
            messages.append({"role": "assistant", "content": assistant_answer})

        mistral_template = self.apply_chat_template(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            tokenizer=tokenizer
        )

        return mistral_template

    def message(self, system_prompt, user_prompt, temperature=0.1, num_samples=1):
        prompt = self.generate_chat_template(
            tokenizer=self.tokenizer,
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )
        answers = self.call(
            prompt=prompt,
            max_tokens=self.max_new_tokens,
            top_k=50,
            top_p=1,
            temperature=temperature,
            terminators=None,
            num_return_sequences=num_samples
        )

        batch_size = len(prompt)
        batch_responses = []
        for i in range(0, num_samples * batch_size, num_samples):
            response_parsed = [answer for answer in answers[i:i + num_samples]]
            batch_responses.append(response_parsed)

        return batch_responses

    def message_batch(self, prompts, temperature=0.1, num_samples=1):
        answers = self.call(
            prompt=prompts,
            max_tokens=self.max_new_tokens,
            top_k=50,
            top_p=1,
            temperature=temperature,
            terminators=None,
            num_return_sequences=num_samples
        )
        return [answer for answer in answers]

class QwenModel(HFModel):
    model_ids = {
        "qwen2.5-coder": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "qwen2.5-instruct": "Qwen/Qwen2.5-7B-Instruct",
        "qwen3-8b": "Qwen/Qwen3-8B",
        "qwen3-30b": "Qwen/Qwen3-30B-A3B",
        "qwen3-coder-30b": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "qwen4b-instruct": "Qwen/Qwen3-4B-Instruct-2507",
        "qwen4b-thinking": "Qwen/Qwen3-4B-Thinking-2507",
        "qwen4b-instruct-covr-r1024-msl-6028": "manarabdelatty/qwen4b-instruct-covr-completion-only-loss-r1024-msl-6028",
        "qwen4b-instruct-covr-r1024-msl-8028": "manarabdelatty/qwen4b-instruct-covr-completion-only-loss-r1024-msl-8028",
        "qwen2.5-coder-covr-comp-only-r1024": "manarabdelatty/qwen2.5-coder-covr-completion-only-loss-r1024-merged",
        "qwen2.5-coder-covr-comp-only-r1024-msl-6028": "manarabdelatty/qwen2.5-coder-covr-completion-only-loss-r1024-merged_msl_6028",
        "qwen2.5-coder-covr-comp-only-r1024-msl-8028": "manarabdelatty/qwen2.5-coder-covr-completion-only-loss-r1024-merged_msl_8028",
    }
    unsloth_model_ids = {
        "qwen2.5-instruct": "unsloth/Qwen2.5-7B-Instruct",
        "qwen2.5-coder": "unsloth/Qwen2.5-Coder-7B-Instruct",
        "qwen4b-instruct": "unsloth/Qwen3-4B-Instruct-2507",
        "qwen4b-thinking": "unsloth/Qwen3-4B-Thinking-2507",
        "qwen3-8b": "unsloth/Qwen3-8B",
        "qwen3-30b": "unsloth/Qwen3-30B-A3B"
    }

    def __init__(self, name, load_model=True, lora_path=None, merge_lora=False, use_sloth=False,  multi_gpu=False, max_seq_length=1048, max_new_tokens=None):
        super().__init__(use_accelerate=multi_gpu, use_sloth=use_sloth)
        self.name = name
        self.model = None
        self.tokenizer = None

        if load_model:
            self.model, self.tokenizer = self.load(
                model_id=QwenModel.model_ids[name],
                add_special_tokens=True,
                lora_path=lora_path,
                merge_lora=merge_lora,
                max_seq_length=max_seq_length
            )

        self.max_seq_length = max_seq_length
        self.max_new_tokens = _resolve_max_new_tokens(max_seq_length, max_new_tokens)

    def generate_chat_template(
        self,
        tokenizer,
        system_prompt: str,
        user_prompt: str,
        assistant_answer: str = "",
        add_generation_prompt: bool = True,
    ):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": user_prompt})

        if assistant_answer:
            messages.append({"role": "assistant", "content": assistant_answer})

        return self.apply_chat_template(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            tokenizer=tokenizer
        )
    
    def message(self, system_prompt, user_prompt, temperature=0.1, num_samples=1):
        prompt = self.generate_chat_template(
            tokenizer=self.tokenizer,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        # prompt = [prompt] 

        answers = self.call(
            prompt=prompt,
            max_tokens=self.max_new_tokens,
            top_k=50,
            top_p=0.95,
            temperature=temperature,
            terminators=None,
            num_return_sequences=num_samples
        )
        return answers if num_samples > 1 else answers[0]
    
    def message_batch(self, prompts, temperature=0.1, num_samples=1):
        answers = self.call(
            prompt=prompts,
            max_tokens=self.max_new_tokens,
            top_k=50,
            top_p=0.95,
            temperature=temperature,
            terminators=None,
            num_return_sequences=num_samples
        )
        return answers

class StarCoderModel(HFModel):
    model_ids = {
        "starcoder2:15b": "bigcode/starcoder2-15b-instruct-v0.1"
    }

    def __init__(self, name, load_model=True, lora_path = None, merge_lora=False, use_sloth=False, multi_gpu=False, max_seq_length=4048, max_new_tokens=None):
        super().__init__(use_accelerate=multi_gpu, use_sloth=use_sloth)
        self.model = None
        self.tokenizer = None
        if load_model:
            self.model, self.tokenizer = self.load(
                model_id=StarCoderModel.model_ids[name],
                add_special_tokens=True,
                lora_path=lora_path,
                merge_lora=merge_lora,
                max_seq_length = max_seq_length
            )

        self.max_seq_length = max_seq_length
        self.max_new_tokens = _resolve_max_new_tokens(max_seq_length, max_new_tokens)

    def generate_prompt(self, system_prompt, user_prompt, assistant_answer=""):
        prompt = ""
        if system_prompt:
            prompt += system_prompt.strip() + "\n"
        prompt += user_prompt.strip()
        if assistant_answer:
            prompt += "\n" + assistant_answer.strip()
        return prompt

    def generate_chat_template(self, tokenizer, system_prompt, user_prompt, assistant_answer="", add_generation_prompt: bool = True):
        return self.generate_prompt(system_prompt, user_prompt, assistant_answer)

    def message(self, system_prompt, user_prompt, temperature=0.1, num_samples=1):
        prompt = [self.generate_prompt(system_prompt, user_prompt)]
        answers = self.call(
            prompt=prompt,
            max_tokens=self.max_new_tokens,
            top_k=50,
            top_p=1,
            temperature=temperature,
            terminators=None,
            num_return_sequences=num_samples
        )
        return answers if num_samples > 1 else answers[0]

    def message_batch(self, prompts, temperature=0.1, num_samples=1):
        answers = self.call(
            prompt=prompts,
            max_tokens=self.max_new_tokens,
            top_k=50,
            top_p=1,
            temperature=temperature,
            terminators=None,
            num_return_sequences=num_samples
        )
        # batch_size = len(prompts)
        # batch_responses = []

        # for i in range(0, num_samples * batch_size, num_samples):
        #     response_parsed = [answer for answer in answers[i:i + num_samples]]
        #     batch_responses.append(response_parsed if num_samples > 1 else response_parsed[0])

        # flat_responses = []
        # for responses in batch_responses:
        #     if isinstance(responses, list):
        #         flat_responses.extend(responses)
        #     else:
        #         flat_responses.append(responses)

        return answers


class VeriGenModel(HFModel):
    model_ids = {
        "codegen-16b-verilog": "shailja/fine-tuned-codegen-16B-Verilog",
        "codegen-6b-verilog": "shailja/fine-tuned-codegen-6B-Verilog",
        "codegen-2b-verilog": "shailja/fine-tuned-codegen-2B-Verilog"
    }

    def __init__(self, name, load_model=True, lora_path = None, merge_lora=False, use_sloth=False, max_seq_length=4048, max_new_tokens=None):
        super().__init__()

        self.model = None
        self.tokenizer = None
        if load_model:
            self.model, self.tokenizer = self.load(
                model_id=VeriGenModel.model_ids[name],
                add_special_tokens=True,
                lora_path=lora_path,
                merge_lora=merge_lora,
                max_seq_length = max_seq_length
            )

        self.max_seq_length = max_seq_length
        self.max_new_tokens = _resolve_max_new_tokens(max_seq_length, max_new_tokens)

    def generate_chat_template(self, tokenizer, system_prompt, user_prompt, assistant_answer: str = "", add_generation_prompt: bool = True):
        return system_prompt + user_prompt

    def message(self, system_prompt, user_prompt, temperature=0.1, num_samples=1):
        prompt = self.generate_chat_template(
            tokenizer=self.tokenizer,
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )
        answer = self.call(
            prompt=prompt,
            max_tokens=self.max_new_tokens,
            top_k=50,
            top_p=1,
            temperature=temperature,
            terminators=None,
            num_return_sequences=num_samples
        )
        return answer

    def message_batch(self, prompts, temperature=0.1, num_samples=1):
        answers = self.call(
            prompt=prompts,
            max_tokens=self.max_new_tokens,
            top_k=50,
            top_p=1,
            temperature=temperature,
            terminators=None,
            num_return_sequences=num_samples
        )

        return answers


class RTLCoderModel(HFModel):

    model_ids = {
        "rtlcoder-deepseek": "ishorn5/RTLCoder-Deepseek-v1.1",
        "rtlcoder-mistral": "ishorn5/RTLCoder-v1.1"
    }

    def __init__(self, name, load_model=True, lora_path = None, merge_lora=False, use_sloth=False, multi_gpu=False, max_seq_length=4048, max_new_tokens=None):
        super().__init__(use_accelerate=multi_gpu, use_sloth=use_sloth)

        self.model = None
        self.tokenizer = None
        if load_model:
            self.model, self.tokenizer = self.load(
                model_id=RTLCoderModel.model_ids[name],
                add_special_tokens=False,
                lora_path=lora_path,
                merge_lora=merge_lora,
                max_seq_length=max_seq_length
            )

        self.max_seq_length = max_seq_length
        self.max_new_tokens = _resolve_max_new_tokens(max_seq_length, max_new_tokens)

    def generate_chat_template(self, tokenizer, system_prompt, user_prompt, assistant_answer: str = "", add_generation_prompt: bool = True):
        return system_prompt + user_prompt + assistant_answer

    def message(self, system_prompt, user_prompt, temperature=0.1, num_samples=1):
        prompt = self.generate_chat_template(
            tokenizer=self.tokenizer,
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )
        answer = self.call(
            prompt=prompt,
            max_tokens=self.max_new_tokens,
            top_k=50,
            top_p=1,
            temperature=temperature,
            terminators=None,
            num_return_sequences=num_samples
        )
        return answer

    def message_batch(self, prompts, temperature=0.1, num_samples=1):
        answers = self.call(
            prompt=prompts,
            max_tokens=self.max_new_tokens,
            top_k=50,
            top_p=1,
            temperature=temperature,
            terminators=None,
            num_return_sequences=num_samples
        )

        return answers


class CodeVModel(HFModel):
    model_ids = {
        "codev-7b": "yang-z/CodeV-CL-7B", # codellama/CodeLlama-7b-Python-hf,
        "codev-code-qwen-7b": "yang-z/CodeV-QW-7B", # codellama/CodeLlama-7b-Python-hf
        "codev-qwen-coder-7b": "yang-z/CodeV-QC-7B", # codellama/CodeLlama-7b-Python-hf
        "codev-all-code-qwen-7b": "yang-z/CodeV-All-CQ", # Qwen/CodeQwen1.5-7B-Chat
        "codev-all-qwen-coder-7b": "yang-z/CodeV-All-QC" # Qwen/CodeQwen1.5-7B-Chat

    }

    def __init__(self, name, load_model=True, max_seq_length=2048, max_new_tokens=None, **kwargs) -> None:
        super().__init__()
        self.max_seq_length = max_seq_length
        self.max_new_tokens = _resolve_max_new_tokens(max_seq_length, max_new_tokens)

        self.generator = None
        self.tokenizer = None
        if load_model:
            self.generator = pipeline(
                model=CodeVModel.model_ids[name],
                task="text-generation",
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )

    def generate_chat_template(self, tokenizer, system_prompt, user_prompt, assistant_answer: str = "", add_generation_prompt: bool = True):
        return system_prompt + user_prompt + assistant_answer

    def message(self, system_prompt, user_prompt, temperature=0.1, num_samples=1):
        prompt = self.generate_chat_template(
            tokenizer=self.tokenizer,
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )
        result = self.generator(
            prompt,
            max_new_tokens=self.max_new_tokens,
            num_return_sequences=num_samples,
            temperature=temperature
        )
        response = result[0]["generated_text"]
        return [response]

    def message_batch(self, prompts, temperature=0.1, num_samples=1):
        do_sample = True if num_samples > 1 else False  # Required for multiple generations
        results = self.generator(
            prompts,
            max_new_tokens=self.max_new_tokens,
            num_return_sequences=num_samples,
            temperature=temperature,
            return_full_text=True,
            do_sample=do_sample
        )

        if isinstance(results[0], list):
            flat_results = [r["generated_text"] for batch in results for r in batch]
        else:
            flat_results = [r["generated_text"] for r in results]

        return flat_results


class VeriThoughtsModel(HFModel):
    model_ids = {
        "verithoughts-r:14b": "nyu-dice-lab/VeriThoughts-Reasoning-14B",
        "verithoughts-r:7b": "nyu-dice-lab/VeriThoughts-Reasoning-7B",
        "verithoughts-i:7b": "nyu-dice-lab/VeriThoughts-Instruct-7B"
    }

    def __init__(self, name, load_model=True, lora_path=None, merge_lora=False, use_sloth=False, multi_gpu=False, max_seq_length=1048, max_new_tokens=None):
        super().__init__(use_accelerate=multi_gpu, use_sloth=use_sloth)

        self.name = name
        self.model = None
        self.tokenizer = None
        if load_model:
            self.model, self.tokenizer = self.load(
                model_id=VeriThoughtsModel.model_ids[name],
                add_special_tokens=True,
                lora_path=lora_path,
                merge_lora=merge_lora,
                max_seq_length=max_seq_length
            )

        self.max_seq_length = max_seq_length
        self.max_new_tokens = _resolve_max_new_tokens(max_seq_length, max_new_tokens)

    def generate_chat_template(self, tokenizer, system_prompt, user_prompt, assistant_answer="", add_generation_prompt: bool = True):
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": user_prompt})
        if assistant_answer:
            messages.append({"role": "assistant", "content": assistant_answer})

        return self.apply_chat_template(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            tokenizer=tokenizer,
        )

    def message(self, system_prompt, user_prompt, temperature=0.1, num_samples=1):
        prompt = self.generate_chat_template(
            tokenizer=self.tokenizer,
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )
        prompt = [prompt]
        answers = self.call(
            prompt=prompt,
            max_tokens=self.max_new_tokens,
            top_k=50,
            top_p=0.95,
            temperature=temperature,
            terminators=None,
            num_return_sequences=num_samples
        )

        batch_responses = []
        for i in range(0, num_samples, 1):
            batch_responses.append(answers[i])

        return batch_responses if num_samples > 1 else batch_responses[0]

    def message_batch(self, prompts, temperature=0.1, num_samples=1):
        responses = self.call(
            prompt=prompts,
            max_tokens=self.max_new_tokens,
            top_k=50,
            top_p=1,
            temperature=temperature,
            terminators=None,
            num_return_sequences=num_samples
        )

        parsed = [ans for ans in responses]
        return parsed


def get_model(name: str, load_model=True, lora_path=None, merge_lora=False, use_sloth=False, use_vllm=False, multi_gpu=False, max_seq_length=1048, max_new_tokens=None, gpu_memory_utilization=0.9):
    if name in GPTModel.names:
        return GPTModel(
            name=name
        )

    if use_vllm:
        VLLMModel.resolve_model_id(name)
        return VLLMModel(
            name=name,
            lora_path=lora_path,
            max_seq_length=max_seq_length,
            max_new_tokens=max_new_tokens,
            gpu_memory_utilization=gpu_memory_utilization
        )

    if name in RTLCoderModel.model_ids.keys():
        return RTLCoderModel(
            name=name,
            load_model=load_model, 
            lora_path=lora_path,
            merge_lora=merge_lora,
            max_seq_length=max_seq_length,
            max_new_tokens=max_new_tokens,
            use_sloth=use_sloth,
            multi_gpu=multi_gpu
        )

    if name in LLM4CovModel.model_ids.keys():
        return LLM4CovModel(
            name=name,
            load_model=load_model, 
            max_seq_length=max_seq_length,
            max_new_tokens=max_new_tokens,
        )

    if name in VeriGenModel.model_ids.keys():
        return VeriGenModel(
            name=name,
            load_model=load_model, 
            lora_path=lora_path,
            merge_lora=merge_lora,
            max_seq_length=max_seq_length,
            max_new_tokens=max_new_tokens,
            use_sloth=use_sloth,
            multi_gpu=multi_gpu
        )

    if name in MistralModel.model_ids.keys():
        return MistralModel(
            name=name,
            load_model=load_model, 
            lora_path=lora_path,
            merge_lora=merge_lora,
            max_seq_length=max_seq_length,
            max_new_tokens=max_new_tokens,
            use_sloth=use_sloth,
            multi_gpu=multi_gpu
        )

    if name in LlamaModel.model_ids.keys():
        return LlamaModel(
            name=name,
            load_model=load_model,
            lora_path=lora_path,
            merge_lora=merge_lora,
            use_sloth=use_sloth,
            multi_gpu=multi_gpu,
            max_seq_length=max_seq_length,
            max_new_tokens=max_new_tokens,
        )

    if name in CodeLlamaModel.model_ids.keys():
        return CodeLlamaModel(
            name=name,
            load_model=load_model,
            lora_path=lora_path,
            merge_lora=merge_lora,
            max_seq_length=max_seq_length,
            max_new_tokens=max_new_tokens,
            use_sloth=use_sloth,
            multi_gpu=multi_gpu
        )

    if name in QwenModel.model_ids.keys():
        return QwenModel(
            name=name,
            load_model=load_model,
            lora_path=lora_path,
            merge_lora=merge_lora,
            max_seq_length=max_seq_length,
            max_new_tokens=max_new_tokens,
            use_sloth=use_sloth,
            multi_gpu=multi_gpu
        )

    if name in DeepSeekModel.model_ids.keys():
        return DeepSeekModel(
            name=name
        )

    if name in DeepSeekCoderModel.model_ids.keys():
        return DeepSeekCoderModel(
            name=name,
            load_model=load_model,
            lora_path=lora_path,
            merge_lora=merge_lora,
            max_seq_length=max_seq_length,
            max_new_tokens=max_new_tokens,
            use_sloth=use_sloth,
            multi_gpu=multi_gpu
        )

    if name in StarCoderModel.model_ids.keys():
        return StarCoderModel(
            name=name,
            load_model=load_model,
            lora_path=lora_path,
            merge_lora=merge_lora,
            max_seq_length=max_seq_length,
            use_sloth=use_sloth,
            multi_gpu=multi_gpu
        )

    if name in CodeVModel.model_ids.keys():
        return CodeVModel(
            name=name,
            load_model=load_model,
            lora_path=lora_path,
            merge_lora=merge_lora,
            use_sloth=use_sloth,
            max_seq_length=max_seq_length,
            max_new_tokens=max_new_tokens,
            multi_gpu=multi_gpu
        )

    if name in VeriThoughtsModel.model_ids.keys():
        return VeriThoughtsModel(
            name=name,
            load_model=load_model,
            lora_path=lora_path,
            merge_lora=merge_lora,
            use_sloth=use_sloth,
            max_seq_length=max_seq_length,
            multi_gpu=multi_gpu
        )

    print(f"[ERROR] Invalid Model name: {name}")
    sys.exit(0)


class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help="Model name", required=True)
    parser.add_argument('--use_sloth', action='store_const', const=True, help="Use sloth models for faster finetuning", default=False)
    parser.add_argument('--multi_gpu', action='store_const', const=True, help="Use multi gpus for faster finetuning", default=False)
    parser.add_argument("--use_vllm", action="store_const",  const=True, help="Use VLLM for faster inference", default=False)

    args = parser.parse_args()

    name = args.name
    use_vllm = args.use_vllm
    use_sloth = args.use_sloth
    multi_gpu = args.multi_gpu

    logger = get_logger('model.log')

    # gpt_model = GPTModel(
    #     name='gpt-3.5-turbo'
    # )

    # answer = gpt_model.message(
    #     system_prompt=SYS_PROMPT_P1,
    #     user_prompt=USER_INSTRUCTION_P1
    # )

    # logger.info(f"YELLOW: GPT Answer@1 is {answer[0]}")

    # llama_model = LlamaModel(
    #     name='llama3.1:8b',
    # )

    # answer = llama_model.message(
    #     system_prompt=SYS_PROMPT_P1,
    #     user_prompt=USER_INSTRUCTION_P1
    # )

    # logger.info(f"YELLOW: Llama Answer@1 is {answer[0][0]}")

    # codellama_model = CodeLlamaModel(
    #     name='codellama:70b',
    # )

    # answer = codellama_model.message(
    #     system_prompt=SYS_PROMPT_P1,
    #     user_prompt=USER_INSTRUCTION_P1
    # )

    # logger.info(f"YELLOW: Code Llama Answer@1 is {answer[0][0]}")

    # mistral_model = MistralModel(
    #     name='mixtral',
    # )
    # answer = mistral_model.message(
    #     system_prompt=SYS_PROMPT_P1,
    #     user_prompt=USER_INSTRUCTION_P1
    # )

    # logger.info(f"YELLOW: Mixtral Answer@1 is {answer[0][0]}")

    # rtlcoder_model = RTLCoderModel(
    #     name='rtlcoder-deepseek',
    # )
    # answer = rtlcoder_model.message(
    #     system_prompt=SYS_PROMPT_P1,
    #     user_prompt=USER_INSTRUCTION_P1
    # )

    # logger.info(f"YELLOW: RTLCoder Answer@1 is {answer}")

    # starcoder_model = StarCoderModel(
    #     name='starcoder2:15b',
    # )
    # answer = starcoder_model.message(
    #     system_prompt=SYS_PROMPT_P1,
    #     user_prompt=USER_INSTRUCTION_P1
    # )
    # logger.info(f"YELLOW: Star Coder Answer@1 is {answer[0][0]}")

    # verigen_model = VeriGenModel(
    #     name='codegen-6b-verilog'
    # )
    # answer = verigen_model.message(
    #     system_prompt=SYS_PROMPT_P1,
    #     user_prompt=USER_INSTRUCTION_P1
    # )

    # logger.info(f"YELLOW: VeriGen Answer@1 is {answer}")

    # deepseek_model = DeepSeekModel(
    #     name="deepseek-v3"
    # )
    # answer = deepseek_model.message(
    #     system_prompt=SYS_PROMPT_P1,
    #     user_prompt=USER_INSTRUCTION_P1
    # )
    # logger.info(f"YELLOW: DeepSeek Answer@1 is {answer}")

    # verithoughts_model = VeriThoughtsModel(
    #     name='verithoughts-i:7b',
    # )
    # answer = verithoughts_model.message(
    #     system_prompt=SYS_PROMPT_P1,
    #     user_prompt=USER_INSTRUCTION_P1
    # )
    # logger.info(f"YELLOW: VeriThoughts Answer@1 is {answer[0][0]}")

    texts = ["Hello, what is 2*2 ? ", "Hellom, can you trell me a joke? "]
    dataset = TextDataset(texts)
    dataloader = DataLoader(dataset, batch_size=1)

    model = get_model(
        name=name,
        multi_gpu=multi_gpu,
        use_sloth=use_sloth,
        use_vllm=use_vllm,
        max_seq_length=32000,
        max_new_tokens=10000
    )
    
    # if multi_gpu: 
    #     dataloader = qwen_model.accelerator.prepare(dataloader)

    # all_outputs = []
    # for batch in dataloader:
    #     answers = qwen_model.message_batch(
    #         prompts=batch,
    #         num_samples = 10
    #     )
    #     # gathered_outputs = qwen_model.accelerator.gather(answers)
    #     all_outputs.append(answers)

    # all_outputs = gather_object(all_outputs)
    # print(all_outputs)

    
    # if multi_gpu and qwen_model.accelerator.is_main_process:
    #     print(all_outputs)

    # # Cleanup to prevent warning
    # if multi_gpu: 
    #     qwen_model.accelerator.free_memory()
    #     if torch.distributed.is_initialized():
    #         torch.distributed.destroy_process_group()

    # model = LLM4CovModel(
    #     name=name,
    # )
    rtl_code = """
        module AdlerChecksum (
        input [7:0] data,
        input clk,
        output reg [31:0] checksum
        );

        // no parameters needed

        // two 16-bit registers, initialized to 1
        reg [15:0] s1 = 16'h0001;
        reg [15:0] s2 = 16'h0001;

        // operations for each data byte
        always @ (posedge clk) begin
        s1 <= (s1 + data) % 65521;
        s2 <= (s2 + s1) % 65521;
        end

        // compute Adler checksum after all bytes have been processed
        always @* begin
        checksum = (s2 << 16) + s1;
        end

        endmodule
    """
    specs = """
Write a Verilog module named `AdlerChecksum` that computes a 32-bit checksum for incoming 8-bit data bytes. The module should have the following interface:
`input [7:0] data`, `input clk`, `output reg [31:0] checksum`.

The checksum is based on two 16-bit intermediate sums, commonly referred to as A and B. Initialize both intermediate sums to a value of 1.

On each positive edge of the `clk` signal, process the incoming `data` byte according to the following steps:
1. Update sum A by adding the current `data` byte to its current value. The result must be taken modulo 65521.
2. Update sum B by adding the *newly computed* value of sum A to its current value. This result must also be taken modulo 65521.

The final 32-bit `checksum` is formed by combining the current values of sum A and sum B, where sum B constitutes the most significant 16 bits and sum A constitutes the least significant 16 bits. This `checksum` output should combinatorially reflect the current state of the intermediate sums.

Your implementation should use registers to store the intermediate sums and assign the final checksum based on their current values.    
"""

    instructions = """
Generate a SystemVerilog testbench for the given RTL design. The top module is 'top_module', and its instance name in the testbench should be 'dut'. Target coverage only in your testbench: Try to achieve as high coverage as possible, including line, toggle, FSM, branch, etc. Also, don't include assertion statements in the testbench; they'll be handled separately.
"""

    answer = model.message(
        system_prompt=LLM4_COV_SYSTEM_PROMPT,
        user_prompt = LLM4COV_USER_TEMPLATE(
            context=specs + rtl_code,
            instructions=instructions
        )
    )

    print(answer)

    # answer = qwen_model.message(
    #     system_prompt=SYS_PROMPT,
    #     user_prompt=USER_PROMPT(natural_spec="", rtl_code="")
    # )
    # print(answers)
    # logger.info(f"YELLOW: VeriThoughts Answer@1 is {answer}")


if __name__ == '__main__':
    main()
