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

from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch.utils.data import DataLoader, Dataset

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.encodings = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    
    def __len__(self):
        return len(self.encodings.input_ids)
    
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}


# Load model and tokenizer
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map=None
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Setup
accelerator = Accelerator()
texts = ["Text 1", "Text 2", "Text 3"]
dataset = TextDataset(texts, tokenizer)
dataloader = DataLoader(dataset, batch_size=2)

# Prepare with accelerator
model, dataloader = accelerator.prepare(model, dataloader)

# Inference
all_outputs = []
for batch in dataloader:
    with torch.no_grad():
        outputs = model.module.generate(**batch, max_length=512)
    all_outputs.extend(accelerator.gather(outputs))

# IMPORTANT: Clean up the process group
accelerator.free_memory()
if torch.distributed.is_initialized():
    torch.distributed.destroy_process_group()

print(all_outputs)
# # Initialize accelerator
# accelerator = Accelerator()


# # Prepare model with accelerator
# model = accelerator.prepare(model)

# # Run inference
# inputs = tokenizer("Hello, how are you?", return_tensors="pt")
# inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}

# with torch.no_grad():
#     outputs = model.module.generate(**inputs, max_length=50)

# print(tokenizer.decode(outputs[0]))