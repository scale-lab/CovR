"""Report Coverage on VerilogEval
"""
import os 
import sys 
import json 
import tqdm
import argparse
from datetime import datetime

from core.metrics.coverage import run_covered, run_testbench, report_coverage
from core.metrics.vcs import run_testbench_vcs, run_vcs, run_urg, run_sim, report_coverage_vcs
# from data.extract_verilogeval import fix_dump_vars
from core.metrics.utils import get_top_module
from core.data_generation.utils import compute_avg_coverage


def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', type=str, help="Path to JSON dataset", default='/oscar/data/sreda/mabdelat/LLM4TB/data/correctbench.json')
    parser.add_argument('--output_dir', type=str, help="Path to output data dir", default="./output")

    args = parser.parse_args()

    json_path = args.json 
    output_dir = args.output_dir

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tmp_output_dir = os.path.abspath(os.path.join(output_dir, f"tmp_{timestamp}"))
    os.makedirs(tmp_output_dir, exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    
    avg_stats = {
        'line': 0,
        'cond' : 0,
        'toggle': 0,
        'fsm': 0,
        'branch': 0        
    }

    num_tasks = 0
    cov = 0.0
    for record in records:
        task_id = record.get("task_id", "unknown_task")
        ref_module_code = record.get("ref_module", "")
        top_module_code = record.get("rtl_code", "")
        testbench_code = record.get("testbench", "")
        print(f"Processing Task: {task_id}")

        task_dir = os.path.join(tmp_output_dir, task_id)
        os.makedirs(task_dir, exist_ok=True)
        print(f"Task directory: {task_dir}")

        ref_module_file = os.path.join(task_dir, f"{task_id}_ref.v")
        with open(ref_module_file, "w", encoding="utf-8") as mf:
            mf.write(ref_module_code)

        top_module_file = os.path.join(task_dir, f"{task_id}_top.v")
        with open(top_module_file, "w", encoding="utf-8") as mf:
            mf.write(top_module_code)

        tb_file = os.path.join(task_dir, f"{task_id}_tb.v")
        with open(tb_file, "w", encoding="utf-8") as tf:
            tf.write(testbench_code)

        # testbench_name = get_top_module(tb_file)
        # testbench_code = fix_dump_vars(testbench_code, f"{testbench_name}.uut")
    
        # with open(tb_file, "w", encoding="utf-8") as tf:
        #     tf.write(testbench_code)

        # tb_result = run_testbench(
        #     verilog_files = [top_module_file, ref_module_file],
        #     testbench=tb_file,
        #     singularity_image="/oscar/data/sreda/mabdelat/LLM4TB/sing.sif",
        #     output_dir=task_dir
        # ) 

        top_module_name = get_top_module(ref_module_file)

        # covered_result, report_file = run_covered(
        #     verilog_files = [top_module_file, ref_module_file],
        #     testbench=tb_file,
        #     vcd_file=os.path.join(task_dir, "wave.vcd"),
        #     top_module_name=top_module_name,
        #     top_module_tree_path=f"{testbench_name}.uut",
        #     singularity_image="/oscar/data/sreda/mabdelat/LLM4TB/sing.sif",
        #     output_dir=task_dir
        # )
        correct, tb_compile_res = run_testbench_vcs(
            testbench=tb_file,
            top_module_name=top_module_name, 
            verilog_files = [top_module_file, ref_module_file],
            output_dir=task_dir
        )

        if "Error" in tb_compile_res or "failed" in tb_compile_res or "error" in tb_compile_res or "Failed" in tb_compile_res:
            correct = False
            print(f"Skipping task {task_id} due to testbench compilation error.")
            continue 
        
        print("TB Compile correct: ", correct)
        print("TB Compile result: ", tb_compile_res)
        # if covered_result: 
        #     stats = report_coverage(
        #         report_file=report_file
        #     )
        if correct: 
            correct, report_file = run_urg(
                top_module_name=top_module_name,
                output_dir=task_dir,
                testbench=tb_file
            )

            print("URG correct: ", correct)
            print("URG report file: ", report_file)

            # if correct:
            coverage = report_coverage_vcs(
                report_file=report_file
            )
            
            print("Pred Metrics: ", coverage)

            avg_coverage = compute_avg_coverage(coverage)
            cov += avg_coverage
            # print(stats)

            # for key in avg_stats.keys(): 
            #     if coverage[key] is not None:
            #         avg_stats[key] += coverage[key]

            num_tasks += 1 

    # for key in avg_stats.keys(): 
    #     avg_stats[key] = avg_stats[key] / num_tasks
    avg = cov/num_tasks if num_tasks > 0 else 0.0
    print(f"Average Coverage over {num_tasks} tasks: {avg:.4f}")
    # print(avg_stats)

    avg_all = cov/len(records) if len(records) > 0 else 0.0
    print(f"Average Coverage over all {len(records)} tasks: {avg_all:.4f}")


if __name__ == '__main__':
    main()
