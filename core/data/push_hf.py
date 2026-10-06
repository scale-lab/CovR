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