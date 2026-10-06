"""Makes sure that testbenches are self checking with Error: print statements 
"""

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


def get_max_sample_id(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    max_id = -1
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                sample_id = int(data["sample_id"])
                max_id = max(max_id, sample_id)
            except json.JSONDecodeError:
                continue
    return max_id


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Model name', default='gpt-4o-mini-2024-07-18')
    parser.add_argument("--resume", action="store_true", help="Resume from the last checkpoint", default=True) 
    parser.add_argument('--dataset', type=str, help='Path to directory containing dataset', default=None)
    parser.add_argument('--output', type=str, help='Path to output direcotyr', default='./output/data_generation')

    args = parser.parse_args()
 
    model = args.model 
    resume = args.resume
    dataset = args.dataset
    output = args.output 

    os.makedirs(output, exist_ok=True)
    met_path_output = os.path.join(output, "train_dataset_met_coverage_self_checking.jsonl")
    not_met_path_output = os.path.join(output, "train_dataset_not_met_coverage_self_checking.jsonl")
    failed_path =  os.path.join(output, "train_dataset_self_checking_failed.jsonl")
    
    max_ids = [
        # get_max_sample_id(met_path_output),
        get_max_sample_id(not_met_path_output),
        # get_max_sample_id(failed_path)
    ]
    
    met_ids = get_sample_ids(met_path_output)
    not_met_ids = get_sample_ids(not_met_path_output)

    overall_max = max([i for i in max_ids if i is not None], default=0)
     
    input_met_path = os.path.join(dataset, "train_dataset_met_coverage.jsonl")
    input_not_met_path = os.path.join(dataset, "train_dataset_not_met_coverage.jsonl")
    
    # data = read_jsonl(input_met_path)
    data = read_jsonl(input_not_met_path)

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

    runner = Runner(
        config=config 
    )


    with open(met_path_output, 'a', encoding='utf-8') as f_met, \
        open(not_met_path_output, 'a', encoding='utf-8') as f_not_met, \
        open(failed_path, 'a', encoding='utf-8') as f_fail:

        for entry in tqdm.tqdm(data):
            sid = int(entry["sample_id"])
 
            # if int(sid) <= int(overall_max):
            #     print("Skipping already proceesed sample: ", sid)
            #     continue  # skip already processed samples

            if int(sid) in not_met_ids:
                print("Skipping already proceesed sample: ", sid)
                continue 

            coverage_stats = entry["coverage_stats"]

            output = runner.forward({
                "rtl_code": entry["rtl_code"], 
                "language": "Verilog",
                "natural_spec": entry["natural_spec"],
                "top_module": entry["top_module"],
                "testbench": entry["testbench"]
            })

            testbench = output["testbench"]
            
            if output.get("coverage_stats", None):
                coverage_stats = output["coverage_stats"]
                coverage_report = output["coverage_report"]
                avg_coverage = compute_avg_coverage(output["coverage_stats"])
                before_avg_coverage = compute_avg_coverage(entry["coverage_stats"])
                if avg_coverage < before_avg_coverage:
                    print("WARNING: Coverage is less!!!")
            else:
                coverage_stats = entry["coverage_stats"]
                coverage_report = entry["coverage_report"]
            
            metrics = ['line', 'toggle', 'cond', 'fsm', 'branch']
            coverage_met = all(
                (coverage_stats[m] is None) or (coverage_stats[m] > 95)
                for m in metrics if m in coverage_stats
            )

            print(output["is_self_checking"])
            print(output["is_print_error"])

            if output["is_self_checking"] and output["is_print_error"]: 
                record = {
                    "testbench": output["testbench"],
                    "tb_prompt": entry["tb_prompt"],
                    "coverage_stats": coverage_stats,
                    "coverage_report": coverage_report,
                    "coverage_met": coverage_met,
                    "enhancer_reasoning_trace": entry["enhancer_reasoning_trace"],
                    "tb_generator_reasoning_trace": entry["tb_generator_reasoning_trace"],
                    "is_self_checking": output["is_self_checking"],
                    "tb_reasoning_gen_output": entry.get("tb_reasoning_gen_output", ""),
                    "is_print_error": output["is_print_error"]
                }
                write_record(
                    dataset_id=entry["dataset"],
                    model=entry["model"],
                    sample_id=entry["sample_id"],
                    rtl_code=entry["rtl_code"],
                    natural_spec=entry["natural_spec"],
                    top_module=entry["top_module"],
                    output=record,
                    f_fail=f_fail,
                    f_met=f_met,
                    f_not_met=f_not_met
                )
            elif output.get("testbench_runs", False): 
                record = {
                    "testbench": output["testbench"],
                    "tb_prompt": entry["tb_prompt"],
                    "coverage_stats": coverage_stats,
                    "coverage_report": coverage_report,
                    "coverage_met": coverage_met,
                    "enhancer_reasoning_trace": entry["enhancer_reasoning_trace"],
                    "tb_generator_reasoning_trace": entry["tb_generator_reasoning_trace"],
                    "tb_reasoning_gen_output": entry.get("tb_reasoning_gen_output", ""),
                    "is_self_checking": output["is_self_checking"],
                    "is_print_error": output["is_print_error"]
                }
                write_record(
                    dataset_id=entry["dataset"],
                    model=entry["model"],
                    sample_id=entry["sample_id"],
                    rtl_code=entry["rtl_code"],
                    natural_spec=entry["natural_spec"],
                    top_module=entry["top_module"],
                    output=record,
                    f_fail=f_fail,
                    f_met=f_met,
                    f_not_met=f_not_met
                )
            else:
                # write it to failed
                entry["is_self_checking"] = output["is_self_checking"]
                entry["is_print_error"] = output["is_print_error"]
                f_fail.write(json.dumps(entry) + "\n")
                f_fail.flush()

            print(output["is_self_checking"])
            print(output["is_print_error"])


    
if __name__ == '__main__':
    main()