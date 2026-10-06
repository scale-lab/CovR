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

"""Filters  a JSON dataset and outputs JSON with updated coverage numebrs
"""
import os
import json
import tqdm
import shutil 
import argparse
import tempfile

from core.metrics.vcs import run_testbench_vcs, run_urg, report_coverage_vcs
from core.data_generation.utils import read_jsonl, remove_entry_from_jsonl


def update_coverage(json_path, output_dir):

    data = read_jsonl(json_path)
    sim_dir = os.path.join(output_dir, "./sim")

    violations = 0
    updated = []

    for entry in tqdm.tqdm(data): 
        top_module_name = entry["top_module"]
        print(top_module_name)
        testbench_path = os.path.join(output_dir, f"{entry['top_module']}_tb.v")
        with open(testbench_path, "w") as f:
            f.write(entry["testbench"])

        rtl_code_path = os.path.join(output_dir, f"{entry['top_module']}.v")
        with open(rtl_code_path, "w") as f:
            f.write(entry["rtl_code"])
        
        correct, result = run_testbench_vcs(
            verilog_files=[os.path.abspath(rtl_code_path)],
            testbench=os.path.abspath(testbench_path),
            output_dir=sim_dir,
            top_module_name=top_module_name
        )

        correct, report_file = run_urg(
            testbench=os.path.abspath(testbench_path),
            output_dir=sim_dir,
            top_module_name=top_module_name
        )

        coverage = report_coverage_vcs(
            report_file=report_file
        )

        for key, value in coverage.items():
            if value: 
                if value < 95:
                    print(coverage)
                    print("Violated Coverage: ", violations)
                    violations = violations + 1

        with open(report_file, 'r') as f: 
            coverage_report = f.read()

        entry["coverage_stats"] = coverage 
        entry["coverage_report"] = coverage_report
        updated.append(entry)

    base_dir = os.path.dirname(os.path.abspath(json_path))
    base_name = os.path.basename(json_path)
    fd, tmp_path = tempfile.mkstemp(prefix=base_name + ".", suffix=".tmp", dir=base_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmpf:
            for e in updated:
                tmpf.write(json.dumps(e, ensure_ascii=False) + "\n")
        os.replace(tmp_path, json_path)  # atomic on same filesystem
    except Exception:
        try:
            shutil.move(tmp_path, json_path)
        except Exception:
            # Best-effort cleanup
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    print(f"Updated {len(updated)} entries. Coverage violations (<95): {violations}")


def filter_coverage(met_path, not_met_path, failed_path, output_dir):
    
    met_data = read_jsonl(met_path)
    sim_dir = os.path.join(output_dir, "./sim")
    
    updated = []

    for entry in tqdm.tqdm(met_data): 
        testbench_path = os.path.join(output_dir, f"{entry['top_module']}_tb.v")
        with open(testbench_path, "w") as f:
            f.write(entry["testbench"])

        rtl_code_path = os.path.join(output_dir, f"{entry['top_module']}.v")
        with open(rtl_code_path, "w") as f:
            f.write(entry["rtl_code"])
        
        correct, result = run_testbench_vcs(
            verilog_files=[os.path.abspath(rtl_code_path)],
            testbench=os.path.abspath(testbench_path),
            output_dir=sim_dir
        )

        correct, report_file = run_urg(
            testbench=testbench_path,
            output_dir=sim_dir
        )

        coverage_stats = report_coverage_vcs(
            report_file=report_file
        )

         
        metrics = ['line', 'toggle', 'cond', 'fsm', 'branch']
        coverage_met = all(
            (coverage_stats[m] is None) or (coverage_stats[m] > 95)
            for m in metrics if m in coverage_stats
        )

        print(coverage_stats)
        print(coverage_met)
        
        
        if not coverage_met: 
            print("Removing sample id: ", entry["sample_id"])
            # write the entry to the not met path
            with open(not_met_path, 'a', encoding='utf-8') as f_not_met:
                f_not_met.write(json.dumps(entry) + "\n")
                f_not_met.flush()

            remove_entry_from_jsonl(met_path, entry["sample_id"])
            

def filter_not_met_coverage(met_path, not_met_path, failed_path, output_dir):
    
    not_met_data = read_jsonl(not_met_path)
    print(len(not_met_data))
    sim_dir = os.path.join(output_dir, "./sim")
    
    updated = []

    for entry in tqdm.tqdm(not_met_data): 
        testbench_path = os.path.join(output_dir, f"{entry['top_module']}_tb.v")
        with open(testbench_path, "w") as f:
            f.write(entry["testbench"])

        rtl_code_path = os.path.join(output_dir, f"{entry['top_module']}.v")
        with open(rtl_code_path, "w") as f:
            f.write(entry["rtl_code"])
        
        correct, result = run_testbench_vcs(
            verilog_files=[os.path.abspath(rtl_code_path)],
            testbench= os.path.abspath(testbench_path),
            output_dir=sim_dir
        )

        if not correct: 
            print("Removing sample id: ", entry["sample_id"])
            with open(failed_path, 'a', encoding='utf-8') as f_not_met:
                f_not_met.write(json.dumps(entry) + "\n")
                f_not_met.flush()

            remove_entry_from_jsonl(not_met_path, entry["sample_id"]) 

        correct, report_file = run_urg(
            testbench=testbench_path,
            output_dir=sim_dir
        )

        coverage_stats = report_coverage_vcs(
            report_file=report_file
        )

       
        metrics = ['line', 'toggle', 'cond', 'fsm', 'branch']
        coverage_met = all(
            (coverage_stats[m] is None) or (coverage_stats[m] > 95)
            for m in metrics if m in coverage_stats
        )

        print(coverage_stats)
        print(coverage_met)

        if coverage_met: 
            print("Removing sample id: ", entry["sample_id"])
            # write the entry to the not met path
            with open(met_path, 'a', encoding='utf-8') as f_met:
                f_met.write(json.dumps(entry) + "\n")
                f_met.flush()

            remove_entry_from_jsonl(not_met_path, entry["sample_id"])



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help="Path to directory containing JSON data", required=False, default="./output/data_generation_verireason")
    parser.add_argument('--output_dir', type=str, help="Path to output directory", required=False, default="./output/data_generation_verithoughts_filtered")
    args = parser.parse_args()

    data_path = args.data_path 
    output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)

    failed_path = os.path.join(data_path, "train_dataset_failed.jsonl")
    met_path = os.path.join(data_path, "train_dataset_met_coverage.jsonl")
    not_met_path = os.path.join(data_path, "train_dataset_not_met_coverage.jsonl")

    
    # update_coverage(json_path=met_path, output_dir=tempfile.mkdtemp())
    update_coverage(json_path=not_met_path, output_dir=tempfile.mkdtemp())

    # if os.path.exists(not_met_path): 
    #     update_coverage(json_path=not_met_path, output_dir=output_dir)

    # filter_coverage(
    #     met_path=met_path, 
    #     not_met_path=not_met_path, 
    #     failed_path=failed_path, 
    #     output_dir=tempfile.mkdtemp()
    # )

    # if os.path.exists(not_met_path): 
    #     filter_not_met_coverage(
    #         met_path=met_path,
    #         not_met_path=not_met_path,
    #         failed_path=failed_path,
    #         output_dir=output_dir
    #     )


if __name__ == '__main__':
    main()