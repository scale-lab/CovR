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

"""Standardize JSON entry format and data types. Each entry should have the following keys: 
- timestamp
- dataset: str
- sample_id: int
- model: str
- natural_spec: str
- rtl_code: str
- top_module: str
- tb_prompt: str
- testbench: str
- coverage_stats: dict
- coverage_report: str
- enhancer_reasoning_trace: str
- tb_generator_reasoning_trace: str
- is_print_error: bool 
- error_statements: bool
"""
import os 
import sys 
import json
import glob
import argparse

from core.data_generation.utils import read_jsonl

STANDARD_KEYS = [
    "timestamp",
    "dataset",
    "sample_id",
    "model",
    "natural_spec",
    "rtl_code",
    "top_module",
    "tb_prompt",
    "testbench",
    "coverage_stats",
    "coverage_report",
    "enhancer_reasoning_trace",
    "tb_generator_reasoning_trace",
    "is_print_error",
    "error_statements"
]

def to_str_or_none(value):
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return str(value)


def to_int_or_none(value):
    if value is None:
        return 0
    if isinstance(value, str):
        return 0 
    return int(value)


def clean_value(value):
    if value is None:
        return {}  # or "" / False depending on expected type

    if isinstance(value, dict):
        return {k: clean_value(v) for k, v in value.items()}

    if isinstance(value, list):
        return [clean_value(v) for v in value]

    return value

def to_dict_or_none(value):
    if value is None:
        return {}
    
    for key in value.keys(): 
        if value[key] is None:
            value[key] = -1
    return value 


def standardize_entry(entry: dict) -> dict:
    e = dict(entry) 
    e["timestamp"] = to_str_or_none(e.get("timestamp"))
    e["dataset"] = to_str_or_none(e.get("dataset"))
    e["sample_id"] = to_int_or_none(e.get("sample_id"))
    e["model"] = to_str_or_none(e.get("model"))
    e["natural_spec"] = to_str_or_none(e.get("natural_spec"))
    e["rtl_code"] = to_str_or_none(e.get("rtl_code"))
    e["top_module"] = to_str_or_none(e.get("top_module"))
    e["tb_prompt"] = e.get("tb_prompt", "")
    e["testbench"] = to_str_or_none(e.get("testbench"))
    e["coverage_stats"] = to_dict_or_none(e.get("coverage_stats"))
    e["coverage_report"] = to_str_or_none(e.get("coverage_report"))
    e["enhancer_reasoning_trace"] = e.get("enhancer_reasoning_trace", "")
    e["tb_generator_reasoning_trace"] = e.get("tb_generator_reasoning_trace", "")
    e["tb_reasoning_gen_output"] =  e.get("tb_reasoning_gen_output", "")
    e["is_print_error"] = e.get("is_print_error", False)
    e["error_statements"] = e.get("error_statements", False)
    e["is_self_checking"] = e.get("is_self_checking", False)
    e["complexity"] = e.get("complexity", "")
    e["rtl_reasoning_trace"] = e.get("rtl_reasoning_trace", "")

    for key in STANDARD_KEYS:
        if key not in e:
            e[key] = ""

    return e



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dirs', type=list[str], help="Path to directory containing JSON data", required=False, default=[ "./output/deduplication"])
    # parser.add_argument('--input_dirs', type=list[str], help="Path to directory containing JSON data", required=False, default=[ "./output/data_generation_pyra", "./output/data_generation_verithoughts",  "./output/data_generation_verithoughts_second_half",  "./output/data_generation_rtlseek",   "./output/data_generation_verireason",  "./output/data_generation_veritriplets"])
    args = parser.parse_args()

    input_dirs = args.input_dirs 

    met_data_paths = []
    not_met_data_paths = []

    jsonl_files = []

    for dir in input_dirs:
        jsonl_files.extend(glob.glob(f"{dir}/*.jsonl", recursive=False))
    
    for path in jsonl_files: 
        data = read_jsonl(path)
        standardized = [standardize_entry(entry) for entry in data]

        # overwrite original jsonl 
        with open(path, "w", encoding="utf-8") as f:
            for obj in standardized:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")



if __name__ == "__main__":
    main()
