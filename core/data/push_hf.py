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

"""Push Json Dataset to huggingface
"""
import os 

from huggingface_hub import login
from huggingface_hub import upload_folder
from huggingface_hub import HfApi
from datasets import load_dataset


def main():

    access_token = os.getenv('HF_ACCESS_TOKEN')
    login(token=access_token)

    api = HfApi(token=access_token)

    configs_dict = {
        "all_merged": "output/deduplication/all_merged.jsonl",
        "met_merged": "output/deduplication/met_merged.jsonl",
        "not_met_merged": "output/deduplication/not_met_merged.jsonl",
    }

    for cfg, json_path in configs_dict.items():
        ds = load_dataset(
            "json",
            data_files={"train": json_path},
        )
        ds.push_to_hub("scale-lab/CovR", config_name=cfg)


    dataset = load_dataset("scale-lab/CovR", config_name="met_merged", token=access_token, split="train")
    print(len(dataset))
    print(dataset)

if __name__ == '__main__':
    main()