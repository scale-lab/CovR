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

"""Count lines of code in design
"""

import os 
import sys 
import json
import argparse
from statistics import median

from core.data_generation.utils import read_jsonl

def _to_float_or_none(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().rstrip("%")
        if cleaned == "":
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def collect_coverage_values(entries, dims):
    coverage_by_dim = {dim: [] for dim in dims}
    for entry in entries:
        coverage_stats = entry.get("coverage_stats")
        if not isinstance(coverage_stats, dict):
            continue

        for dim in dims:
            value = _to_float_or_none(coverage_stats.get(dim))
            if value is not None:
                coverage_by_dim[dim].append(value)
    return coverage_by_dim


def print_coverage_medians(entries, dims):
    coverage_by_dim = collect_coverage_values(entries, dims)
    parts = []
    for dim in dims:
        values = coverage_by_dim[dim]
        med_value = median(values) if values else None
        med_text = f"{med_value:.2f}" if med_value is not None else "N/A"
        parts.append(f"{dim}={med_text}")
    print("  Median Coverage: {" + ", ".join(parts) + "}")


def count_samples_with_dim(entries, dim):
    count = 0
    for entry in entries:
        coverage_stats = entry.get("coverage_stats")
        if not isinstance(coverage_stats, dict):
            continue
        value = _to_float_or_none(coverage_stats.get(dim))
        if value is not None:
            count += 1
    return count


def count_loc(code: str) -> int:
    if not code:
        return 0

    lines = code.splitlines()
    count = 0
    for line in lines:
        stripped = line.strip()
        if stripped == "":
            continue
        if stripped.startswith("//"):   # single-line comments
            continue
        if stripped.startswith("/*") and stripped.endswith("*/"):
            continue
        count += 1

    return count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dirs', type=list[str], help="Path to directory containing JSON data", required=False, default=[
    "./output/data_generation_verithoughts_second_half",
    "./output/data_generation_verithoughts", 
    "./output/data_generation_pyra", 
    "./output/data_generation_llm4cov",
    "./output/data_generation_verireason",
    "./output/data_generation_veritriplets"])

    args = parser.parse_args()

    input_dirs = args.input_dirs 
    coverage_dims = ["toggle", "line", "cond", "branch", "fsm"]

    met_data_paths = []
    not_met_data_paths = []

    
    for dir in input_dirs:
        rtl_counts = []
        testbench_counts = []

        met_coverage = os.path.join(dir, "train_dataset_met_coverage.jsonl")
        not_met_coverage = os.path.join(dir, "train_dataset_not_met_coverage.jsonl")
        met_data = read_jsonl(met_coverage)
        not_met_data = read_jsonl(not_met_coverage)

        data = met_data + not_met_data

        for entry in data: 
            if not entry["testbench"] or not entry["rtl_code"]:
                continue 

            rtl_count = count_loc(entry['rtl_code'])
            tb_count = count_loc(entry['testbench'])

            # if tb_count > 17:
            #     print(entry["natural_spec"])
            #     print("-------------------------")
            #     print(entry["testbench"])
            #     print("-------------------------")
            #     print(entry["coverage_stats"])
            #     print("-------------------------")
            #     print(entry["rtl_code"])
            #     print("-------------------------")
            #     print(entry["tb_reasoning_gen_output"])
            #     # import sys 
            #     # sys.exit(0)

            # if tb_count <= 10 :
            #     # print(entry["testbench"])
            #     continue

          
            rtl_counts.append(rtl_count)
            testbench_counts.append(tb_count)

        if len(rtl_counts) == 0:
            print(f"No entries found in {dir}, skipping.")
            continue

        rtl_min = min(rtl_counts)
        rtl_med = median(rtl_counts)
        rtl_max = max(rtl_counts)

        tb_min = min(testbench_counts)
        tb_med = median(testbench_counts)
        tb_max = max(testbench_counts)

        print("\nDirectory:", dir)
        print(f"  RTL LOC       : {{Min={rtl_min}, Median={rtl_med}, Max={rtl_max}}}")
        print(f"  Testbench LOC : {{Min={tb_min}, Median={tb_med}, Max={tb_max}}}")
        print_coverage_medians(data, coverage_dims)
        fsm_count = count_samples_with_dim(data, "fsm")
        print(f"  FSM Samples   : {fsm_count}/{len(data)}")
        print(f"  Designs Count: {len(data)}")
        print("-" * 60)
    

    #count the merged data 
    merged_json = "./output/data_merged/all_merged.jsonl"
    data = read_jsonl(merged_json)
    rtl_counts = []
    testbench_counts = []
    for entry in data: 
        rtl_count = count_loc(entry['rtl_code'])
        tb_count = count_loc(entry['testbench'])
    
        if tb_count <= 10 :
            # print(entry["testbench"])
            continue

        rtl_counts.append(rtl_count)
        testbench_counts.append(tb_count)
        
    rtl_min = min(rtl_counts)
    rtl_med = median(rtl_counts)
    rtl_max = max(rtl_counts)

    tb_min = min(testbench_counts)
    tb_med = median(testbench_counts)
    tb_max = max(testbench_counts)

    print("\nDirectory:", "CovR Train")
    print(f"  RTL LOC       : {{Min={rtl_min}, Median={rtl_med}, Max={rtl_max}}}")
    print(f"  Testbench LOC : {{Min={tb_min}, Median={tb_med}, Max={tb_max}}}")
    print_coverage_medians(data, coverage_dims)
    fsm_count = count_samples_with_dim(data, "fsm")
    print(f"  FSM Samples   : {fsm_count}/{len(data)}")
    print(f"  Designs Count: {len(data)}")
    print("-" * 60)

    # count the eval dataset 

    eval_json = "./data/eval_dataset.jsonl"
    data = read_jsonl(eval_json)
    rtl_counts = []
    testbench_counts = []
    for entry in data: 
        rtl_count = count_loc(entry['rtl_code'])
        tb_count = count_loc(entry['testbench'])
        
        rtl_counts.append(rtl_count)
        testbench_counts.append(tb_count)

    rtl_min = min(rtl_counts)
    rtl_med = median(rtl_counts)
    rtl_max = max(rtl_counts)

    tb_min = min(testbench_counts)
    tb_med = median(testbench_counts)
    tb_max = max(testbench_counts)

    print("\nDirectory:", "CovR Test")
    print(f"  RTL LOC       : {{Min={rtl_min}, Median={rtl_med}, Max={rtl_max}}}")
    print(f"  Testbench LOC : {{Min={tb_min}, Median={tb_med}, Max={tb_max}}}")
    print_coverage_medians(data, coverage_dims)
    fsm_count = count_samples_with_dim(data, "fsm")
    print(f"  FSM Samples   : {fsm_count}/{len(data)}")
    print(f"  Designs Count: {len(data)}")
    print("-" * 60)


if __name__ == "__main__":
    main()
