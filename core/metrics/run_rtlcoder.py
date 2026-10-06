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

"""Report Coverage on VerilogEval
"""
import os 
import re 
import sys 
import json 
import tqdm
import argparse
from datetime import datetime

from core.coverage import run_covered, run_testbench, report_coverage
from data.extract_verilogeval import fix_dump_vars

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', type=str, help="Path to JSON dataset", default='/oscar/data/sreda/mabdelat/LLM4TB/data/RTLCoder_dataset_tb.json')
    parser.add_argument('--output_dir', type=str, help="Path to output data dir", default="./output")

    args = parser.parse_args()

    json_path = args.json 
    output_dir = args.output_dir

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tmp_output_dir = os.path.abspath(os.path.join(output_dir, f"tmp_{timestamp}"))
    os.makedirs(tmp_output_dir, exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    

    avg_stats = {
        'line': 0,
        'toggle': 0,
        'combinational': 0,
        'fsm': 0        
    }

    num_tasks = 0
    for task_id, record in records.items():
        ref_module_code = record.get("unopt", "")
        testbench_code = record.get("testbench", "")

        task_dir = os.path.join(tmp_output_dir, task_id)
        os.makedirs(task_dir, exist_ok=True)

        ref_module_file = os.path.join(task_dir, f"{task_id}_ref.v")
        with open(ref_module_file, "w", encoding="utf-8") as mf:
            mf.write(ref_module_code)

        testbench_code = fix_dump_vars(testbench_code, "testbench.DUT")
        tb_file = os.path.join(task_dir, f"{task_id}_tb.v")
        with open(tb_file, "w", encoding="utf-8") as tf:
            tf.write(testbench_code)

        correct, tb_result = run_testbench(
            verilog_files = [ref_module_file],
            testbench=tb_file,
            singularity_image="/oscar/data/sreda/mabdelat/LLM4TB/sing.sif",
            output_dir=task_dir
        ) 

        if correct: 
            covered_result, report_file = run_covered(
                verilog_files = [ref_module_file],
                testbench=tb_file,
                vcd_file=os.path.join(task_dir, "wave.vcd"),
                top_module_name="opt_model",
                top_module_tree_path="testbench.DUT",
                singularity_image="/oscar/data/sreda/mabdelat/LLM4TB/sing.sif",
                output_dir=task_dir
            )

            if covered_result: 
                stats = report_coverage(
                    report_file=report_file
                )
                for key in avg_stats.keys(): 
                    avg_stats[key] += stats[key]

                num_tasks += 1 

    for key in avg_stats.keys(): 
        avg_stats[key] = avg_stats[key] / num_tasks
    
    print(avg_stats)


if __name__ == '__main__':
    main()
