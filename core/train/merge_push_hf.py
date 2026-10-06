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

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = "Qwen/Qwen2.5-Coder-7B-Instruct"

# adapter = "/oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen2.5-coder_full_sft_r1024_a1024_completion_only_loss_full_data/qwen2.5-coder_full_sft_r1024_a1024_completion_only_loss_full_data_20260504_174138/Qwen/Qwen2.5-Coder-7B-Instruct_finetuned"
# hub_id = "manarabdelatty/qwen2.5-coder-covr-completion-only-loss-r1024-merged"

# adapter = "/oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen2.5-coder_full_sft_r1024_a1024_completion_only_loss_full_data/qwen2.5-coder_full_sft_r1024_a1024_completion_only_loss_full_data_20260504_174138/Qwen/Qwen2.5-Coder-7B-Instruct_finetuned"
# hub_id = "manarabdelatty/qwen2.5-coder-covr-completion-only-loss-r1024-merged"

# adapter = "/oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen2.5-coder_full_sft_r1024_a1024_completion_only_loss_full_data_msl_6028/qwen2.5-coder_full_sft_r1024_a1024_completion_only_loss_full_data_msl_6028_20260505_231356/Qwen/Qwen2.5-Coder-7B-Instruct_finetuned"
# hub_id = "manarabdelatty/qwen2.5-coder-covr-completion-only-loss-r1024-merged_msl_6028"

# adapter="/oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen2.5-coder_full_sft_r1024_a1024_completion_only_loss_full_data_msl_8028/qwen2.5-coder_full_sft_r1024_a1024_completion_only_loss_full_data_msl_8028_20260505_230008/Qwen/Qwen2.5-Coder-7B-Instruct_finetuned"
# hub_id = "manarabdelatty/qwen2.5-coder-covr-completion-only-loss-r1024-merged_msl_8028"

base = "Qwen/Qwen3-4B-Instruct-2507"

# adapter = "/oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen4b_instruct_full_sft_r1024_a1024_completion_only_loss_full_data_msl_6028/qwen4b_instruct_full_sft_r1024_a1024_completion_only_loss_full_data_msl_6028_20260505_232518/Qwen/Qwen3-4B-Instruct-2507_finetuned"
# hub_id = "manarabdelatty/qwen4b-instruct-covr-completion-only-loss-r1024-msl-6028"

adapter = "/oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen4b_instruct_full_sft_r1024_a1024_completion_only_loss_full_data_msl_8028/qwen4b_instruct_full_sft_r1024_a1024_completion_only_loss_full_data_msl_8028_20260505_232208/Qwen/Qwen3-4B-Instruct-2507_finetuned"
hub_id = "manarabdelatty/qwen4b-instruct-covr-completion-only-loss-r1024-msl-8028"


model = AutoModelForCausalLM.from_pretrained(
    base,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

model = PeftModel.from_pretrained(model, adapter)
model = model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained(base)

model.push_to_hub(hub_id, private=True, safe_serialization=True)
tokenizer.push_to_hub(hub_id, private=True)