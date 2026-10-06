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
import sys 
import torch 

from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFacePipeline
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from peft import PeftModel

from langchain_core.language_models.llms import LLM
from typing import Optional, List, Any
from pydantic import BaseModel

from core.train.model import get_model 
from core.train.cfg.config import combine_cfgs

from dotenv import load_dotenv
load_dotenv()

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
except:
    pass 


class HFChatModel(BaseChatModel):
    model_name: str
    model: Any
    tokenizer: Any
    max_new_tokens: int = 1048

    model_config = {
        "arbitrary_types_allowed": True,
    }

    @property
    def _llm_type(self) -> str:
        return "hf_chat_model"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt_messages = []
        for message in messages:
            if isinstance(message, SystemMessage):
                role = "system"
            elif isinstance(message, HumanMessage):
                role = "user"
            else:
                role = "assistant"
            prompt_messages.append({"role": role, "content": message.content})

        prompt = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        generation_kwargs = {
            "max_new_tokens": kwargs.get("max_new_tokens", self.max_new_tokens),
            "do_sample": kwargs.get("do_sample", False),
            "temperature": kwargs.get("temperature", 0.0),
            "top_p": kwargs.get("top_p", None),
            "pad_token_id": self.model.generation_config.pad_token_id,
            "eos_token_id": self.model.generation_config.eos_token_id,
        }

        outputs = self.model.generate(**inputs, **generation_kwargs)
        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][prompt_len:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=text))]
        )


class TrainModelChatAdapter(BaseChatModel):
    model_name: str
    backend: Any
    temperature: float = 0.0

    model_config = {
        "arbitrary_types_allowed": True,
    }

    @property
    def _llm_type(self) -> str:
        return "train_model_chat_adapter"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs: Any,
    ) -> ChatResult:
        system_parts = []
        user_parts = []
        assistant_parts = []

        for message in messages:
            if isinstance(message, SystemMessage):
                system_parts.append(message.content)
            elif isinstance(message, HumanMessage):
                user_parts.append(message.content)
            else:
                assistant_parts.append(message.content)

        system_prompt = "\n\n".join(part for part in system_parts if part)
        user_prompt = "\n\n".join(
            part for part in [*assistant_parts, *user_parts] if part
        )

        response = self.backend.message(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=self.temperature,
            num_samples=1,
        )

        text = self._normalize_response(response)
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=text))]
        )

    def _normalize_response(self, response: Any) -> str:
        if response is None:
            return ""
        if isinstance(response, str):
            return response
        if isinstance(response, list):
            if not response:
                return ""
            first = response[0]
            if isinstance(first, str):
                return first
            if isinstance(first, list):
                return self._normalize_response(first[0] if first else "")
        return str(response)


class DeepSeek:

    def __init__(self, name) -> None:
        self.deepseek_models = [
            'deepseek-reasoner',
            'deepseek-chat',
            'deepseek-v3'
        ]
        api_key = os.env.get("DEEPSEEK_API_KEY")
        self.client = OpenAI(api_key='sk-a827931061134448bef500a29a79d4a0', base_url="https://api.deepseek.com")
        self.name = name 


    def call(self, sys_prompt, user_prompt):
        response = self.client.chat.completions.create(
            model=self.name,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=False
        )
        output = response.choices[0].message.content
        return output


class Agent:
    
    def __init__(self, max_seq_length=32000, max_new_tokens=10000) -> None:
        self.gpt_models = [
            'gpt-3.5-turbo-0125', 
            'gpt-3.5-turbo', 
            'gpt-4-turbo', 
            'gpt-4-turbo-2024-04-09', 
            'gpt-4o-2024-05-13', 
            'gpt-4o-turbo',
            'gpt-4o-mini-2024-07-18',
            'gpt-5-mini'

        ] 

        self.gemini_models = [
            'gemini-2.5-flash',
            'gemini-1.5-flash',
            'gemini-2.5-flash-lite',
            'gemini-2.5-pro'
        ]
        self.deepseek_models = [
            'deepseek-reasoner',
            'deepseek-chat'
        ]

        self.llama_models = [
            'llama3.1',
            'llama3.1:70b',
            'llama3.3:70b',
            'llama3.1:8b',
            'llama3.2:8b',
            'llama3.1:8b-instruct-q8_0',
            'llama3.1:8b-instruct-q4_0',
            'llama3:70b-instruct',
            'llama3.1:70b-instruct-q4_K_S',
            'llama3.3:70b',
            'mannix/defog-llama3-sqlcoder-8b',
            'llama3-groq-tool-use:70b',
            'deepseek-r1:70b'
        ]
        
        self.train_model_aliases = {
            "gpt-3.5-turbo": "gpt-3.5-turbo",
            "gpt-3.5-turbo-0125": "gpt-3.5-turbo-0125",
            "gpt-4-turbo": "gpt-4-turbo",
            "gpt-4-turbo-2024-04-09": "gpt-4-turbo-2024-04-09",
            "gpt-4o-2024-05-13": "gpt-4o-2024-05-13",
            "gpt-4o-mini-2024-07-18": "gpt-4o-mini-2024-07-18",
            "meta-llama/Meta-Llama-3.1-8B-Instruct": "llama3.1",
            "meta-llama/Meta-Llama-3.1-70B-Instruct": "llama3.1:70b",
            "llama3.1": "llama3.1",
            "llama3.1:8b": "llama3.1:8b",
            "llama3.1:70b": "llama3.1:70b",
            "llama3.3:70b": "llama3.3:70b",
            "Qwen/Qwen2.5-Coder-7B-Instruct": "qwen2.5-coder",
            "Qwen/Qwen2.5-7B-Instruct": "qwen2.5-instruct",
            "qwen2.5-coder": "qwen2.5-coder",
            "qwen2.5-instruct": "Qwen/Qwen2.5-7B-Instruct",
        }
        
        self.max_seq_length = max_seq_length
        self.max_new_tokens = max_new_tokens
        
    def get_model(
        self,
        model,
        temperature=0,
        use_vllm=True,
        lora_path=None,
        api_base=None,
        api_key="EMPTY",
        seed=0,
    ):
        train_model_name = self.train_model_aliases.get(model, model)

        if api_base:
            if lora_path:
                raise ValueError(
                    "Do not pass a local LoRA path when using an OpenAI-compatible "
                    "endpoint. Load the LoRA in the vLLM server and set model to the "
                    "served LoRA name instead."
                )
            truncate_prompt_tokens = max(self.max_seq_length - self.max_new_tokens, 0)
            model_kwargs = {
                "model": model,
                "openai_api_base": api_base,
                "openai_api_key": api_key,
                "temperature": temperature,
                "max_tokens": self.max_new_tokens,
                "extra_body": {"truncate_prompt_tokens": truncate_prompt_tokens},
            }
            if seed is not None:
                model_kwargs["seed"] = seed
            return ChatOpenAI(**model_kwargs)
        
        if model in self.gpt_models: 
            if lora_path:
                raise ValueError(f"LoRA adapters can only be used with local/HF models, got OpenAI model: {model}")
            model_kwargs = {
                "model": model,
                # "temperature": temperature,
                # "top_p": 0.0000000000000000000001,
            }
            if seed is not None:
                model_kwargs["seed"] = seed
            return ChatOpenAI(**model_kwargs)

        if model in self.gemini_models:
            if lora_path:
                raise ValueError(f"LoRA adapters can only be used with local/HF models, got Gemini model: {model}")
            return ChatGoogleGenerativeAI(
                model=model,
                # temperature=temperature,
                max_retries=2,
            )
        if model in self.deepseek_models:
            if lora_path:
                raise ValueError(f"LoRA adapters can only be used with local/HF models, got DeepSeek API model: {model}")
            # llm_api_key = os.env.get("DEEPSEEK_API_KEY")
            model_kwargs = {
                "model": model,
                # "temperature": temperature,
                # "top_p": 0.0000000000000000000001,
                "max_retries": 5,
                "openai_api_key": 'sk-a827931061134448bef500a29a79d4a0',
                "openai_api_base": 'https://api.deepseek.com',
                "max_tokens": 5028,
            }
            if seed is not None:
                model_kwargs["seed"] = seed
            return ChatOpenAI(**model_kwargs)
        

        # try:
        backend = get_model(
            name=train_model_name,
            load_model=True,
            max_seq_length=self.max_seq_length,
            max_new_tokens=self.max_new_tokens,
            lora_path=lora_path,
            merge_lora=False,
            use_sloth=False,
            multi_gpu=False,
            use_vllm=use_vllm
        )
        return TrainModelChatAdapter(
            model_name=model,
            backend=backend,
            temperature=temperature,
        )
        # except Exception:
        #     pass

       
        if model in self.llama_models:
            return ChatOllama(
                model=model, 
                temperature=temperature,
                # format=format,
                num_ctx='15028',
                num_predict=1048,
                seed=0,
                # num_gpu=3
            ) 

        print(f"[Error] Invalid Model Type : {model}")
        sys.exit(1)
  
