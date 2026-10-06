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
import tqdm
import json
import argparse 

from core.data_generation.self_checking.runner import Runner 
from core.data_generation.utils import read_jsonl, compute_avg_coverage
from core.data_generation.config.config import FlowConfigs, LMConfigs
from core.data_generation.main import write_record
from core.data_generation.utils import _max_sample_id
from core.data_generation.cleaner.cleaner import StripSelfCheckingRefiner
from core.data_generation.complexity.analyzer import Analyzer


def get_sample_ids(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return set()

    sample_ids = set()

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                sample_id = int(data["sample_id"])
                sample_ids.add(sample_id)
            except (json.JSONDecodeError, KeyError, ValueError):
                continue

    return sample_ids


def append_jsonl(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Model name', default='gpt-4o-mini-2024-07-18')
    parser.add_argument("--resume", action="store_true", help="Resume from the last checkpoint", default=True) 
    parser.add_argument('--dataset', type=str, help='Path to directory containing dataset', default=None)
    parser.add_argument('--output', type=str, help='Path to output direcotyr', default='./output/data_generation')
    parser.add_argument("--run_failed", action="store_true", help="Run failed designs only", default=False) 

    args = parser.parse_args()
 
    model = args.model 
    resume = args.resume
    dataset = args.dataset
    output = args.output 
    run_failed = args.run_failed

    os.makedirs(output, exist_ok=True)
    met_path_output = os.path.join(output, "train_dataset_met_coverage_complexity.jsonl")
    not_met_path_output = os.path.join(output, "train_dataset_not_met_coverage_complexity.jsonl")

    input_met_path = os.path.join(dataset, "train_dataset_met_coverage.jsonl")
    input_not_met_path = os.path.join(dataset, "train_dataset_not_met_coverage.jsonl")

    config = FlowConfigs(
        graph_recursion_limit=20,
        lm_config=LMConfigs(
            prompter_lm = model,
            fixer_lm = model,
            tb_generator_lm = model,
            enhancer_lm = model,
            refiner_lm = model,
            checker_lm = model 
        ),
        max_refine_iters=5,
        max_enhance_iters=5,
    )

    analyzer = Analyzer(
        config=config 
    ) 

    met_ids = get_sample_ids(met_path_output)
    not_met_ids = get_sample_ids(not_met_path_output)

    input_files = [input_met_path, input_not_met_path]
    output_files = [met_path_output, not_met_path_output]
    file_ids = [met_ids, not_met_ids]

    for input_file, output_file, processed_ids in zip(input_files, output_files, file_ids):
        data = read_jsonl(input_file)

        for entry in tqdm.tqdm(data):
            
            sid = entry['sample_id']
            if int(sid) in processed_ids:
                print("Skipping already proceesed sample: ", sid)
                continue 

            output = analyzer.forward({
                "testbench": entry['testbench'],
                "rtl_code": entry['rtl_code'],
                "natural_spec": entry['natural_spec']
            })            

            entry['complexity'] = output['complexity_level']

            append_jsonl(output_file, entry)



if __name__ == '__main__':
    main()