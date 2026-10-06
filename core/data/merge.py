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

"""Merge multiple json datasets together
"""

import os 
import sys 
import json
import argparse

from core.data_generation.utils import read_jsonl


def merge_jsons(data_paths, output_path):
    
    merged_data = []

    for data_path in data_paths:
        if not os.path.exists(data_path):
            print(f"Warning: Path not found -> {data_path}")
            continue 
        
        if data_path.endswith(".jsonl"):
            print(f"Reading: {data_path}")
            try:
                data = read_jsonl(data_path)
                merged_data.extend(data)
            except Exception as e:
                print(f"Error reading {data_path}: {e}")

        with open(output_path, "w", encoding="utf-8") as f:
            for entry in merged_data:
                json.dump(entry, f, ensure_ascii=False)
                f.write("\n")

    print(f"Merged {len(merged_data)} entries into {output_path}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_dirs',
        nargs='+',
        help="Paths to directories containing JSON data",
        required=False,
        default=[
            "./output/data_generation_verithoughts",
            "./output/data_generation_pyra",
            "./output/data_generation_verithoughts_second_half",
            "./output/data_generation_verireason",
            "./output/data_generation_veritriplets",
            "./output/data_generation_llm4cov",
        ],
    )
    # parser.add_argument('--input_dirs', type=list[str], help="Path to directory containing JSON data", required=False, default=["./output/data_generation_verithoughts", "./output/data_generation_pyra", "./output/data_generation_verithoughts_second_half", "./output/data_generation_verireason", "./output/data_generation_veritriplets", "./output/data_generation_rtlseek"]) 
    parser.add_argument('--output_dir', type=str, help="Path to output directory", required=False, default="./output/data_merged")
    args = parser.parse_args()

    input_dirs = args.input_dirs 
    output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)

    met_data_paths = []
    not_met_data_paths = []

    for dir in input_dirs:
        met_data_paths.append(os.path.join(dir, "train_dataset_met_coverage_reasoner_v0_2_output_sdg.jsonl"))
  
    for dir in input_dirs:
        not_met_data_paths.append(os.path.join(dir, "train_dataset_not_met_coverage_reasoner_v0_2_output_sdg.jsonl"))
  
    merge_jsons(
        data_paths=met_data_paths,
        output_path=os.path.join(output_dir, "met_merged.jsonl")
    )

    merge_jsons(
        data_paths=not_met_data_paths,
        output_path=os.path.join(output_dir, "not_met_merged.jsonl")
    )

    merge_jsons(
        data_paths=met_data_paths + not_met_data_paths,
        output_path=os.path.join(output_dir, "all_merged.jsonl")
    )


if __name__ == '__main__':
    main()
