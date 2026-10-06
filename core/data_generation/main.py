
import os 
import sys
import ast
import json
import tqdm
import datetime
import argparse

from datasets import load_dataset

from core.data_generation.runner import DataGenRunner, DataGenNodes
from core.data_generation.config.config import FlowConfigs, LMConfigs
from core.data_generation.top_parser import TopParser
from core.data_generation.utils import read_jsonl, _max_sample_id, compute_avg_coverage, remove_entry_from_jsonl


def _get_fsm_coverage(entry):
    coverage_stats = entry.get("coverage_stats", {})
    value = coverage_stats.get("fsm") if isinstance(coverage_stats, dict) else None

    if value is None:
        return float("inf")

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        cleaned = value.strip().rstrip("%")
        if not cleaned:
            return float("inf")
        try:
            return float(cleaned)
        except ValueError:
            return float("inf")

    return float("inf")


def write_record(
    dataset_id: str,
    model: str,
    sample_id: str,
    rtl_code,
    natural_spec, 
    top_module,
    output,
    f_fail,
    f_met,
    f_not_met,
    complexity = None
):
    
    testbench = output.get("testbench", None)
    testbenches = output.get("testbenches", [testbench])
    coverage_stats = output.get("coverage_stats", None)
    coverage_stats_list = output.get("coverage_stats_list", [coverage_stats])
    coverage_report = output.get("coverage_report", None)
    coverage_reports = output.get("coverage_reports", [coverage_report])
    coverage_met_list = output.get("coverage_met_list", None)
    tbgen_prompt = output.get("tbgen_prompt", None)
    coverage_met = output.get("coverage_met", False)
    enhancer_reasoning_trace = output.get("enhancer_reasoning_trace", None)
    tb_generator_reasoning_trace = output.get("tb_generator_reasoning_trace", None)
    is_self_checking = output.get("is_self_checking", None)
    is_print_error = output.get("is_print_error", None)
    testbench_style = output.get("testbench_style", "self_checking")

    number_samples = len(coverage_reports)

    if number_samples > 0 : 
        # pick testbench with the highest coverage 
        best_idx = -1
        best_coverage = 0.0 
        for i in range(0, number_samples):
            avg_coverage = compute_avg_coverage(coverage_stats_list[i])
            if avg_coverage > best_coverage: 
                best_coverage = avg_coverage 
                best_idx = i 
        
        testbench = testbenches[best_idx]
        coverage_report = coverage_reports[best_idx]
        coverage_stats = coverage_stats_list[best_idx]
        print("Picking Tesbtnech generated at iteration: ", best_idx)


    print("Current sample id is: ", sample_id)
    record = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "dataset": dataset_id,
        "complexity": complexity,
        "sample_id": sample_id,
        "model": model,
        "natural_spec": natural_spec,
        "rtl_code": rtl_code,
        "top_module": top_module,
        "testbench_style": testbench_style,
        "tb_prompt": tbgen_prompt,         
        "testbench": testbench,
        "coverage_stats": coverage_stats,
        "coverage_report": coverage_report,
        "enhancer_reasoning_trace": enhancer_reasoning_trace,
        "tb_generator_reasoning_trace": tb_generator_reasoning_trace,
        "is_self_checking": is_self_checking,
        "is_print_error": is_print_error
    }

    if testbench is None or coverage_stats is None:
        f_fail.write(json.dumps(record) + "\n")
        f_fail.flush()
    elif coverage_met:
        f_met.write(json.dumps(record) + "\n")
        f_met.flush()
    else:
        f_not_met.write(json.dumps(record) + "\n")
        f_not_met.flush()
            

def run_dataset(
    runner,
    top_module_parser,
    model: str,
    dataset: str,
    start_idx: int,
    resume: bool,
    met_path: str ,
    not_met_path: str,
    failed_path: str
):
    
    if dataset == "verithoughts": 
        rtl_code_key = "ground_truth"
        natural_spec_key = "question"
        dataset_id = "wilyub/VeriThoughtsTrainSet"
    elif dataset == "pyra": 
        rtl_code_key = "code"
        natural_spec_key = "description"
        dataset_id = "bnadimi/PyraNet-Verilog"
    else:
        print(f"[ERROR] Invalid dataset name {dataset}")
        sys.exit(0)

    ds = load_dataset(dataset_id)["train"]


    if resume:
        max_met  = _max_sample_id(met_path, dataset_id)
        max_not  = _max_sample_id(not_met_path, dataset_id)
        max_fail = _max_sample_id(failed_path, dataset_id)
        last_done = max(max_met, max_not, max_fail)
       
        if not start_idx: 
            start_idx = last_done + 1
        
        print(f"[RESUME] Resuming at index {start_idx}")
        if start_idx >= len(ds):
            print("All samples already processed.")
            return
    else:
        start_idx = 0 

    unique_complexity = set()

    count = {
        'Basic': 0,
        'Expert': 0,
        'Advanced': 0,
        'Intermediate': 0
    }
    with open(met_path, 'a', encoding='utf-8') as f_met, \
        open(not_met_path, 'a', encoding='utf-8') as f_not_met, \
        open(failed_path, 'a', encoding='utf-8') as f_fail:
        
        for i in tqdm.tqdm(range(start_idx, len(ds))):
            sample = ds[i]
            rtl_code = sample[rtl_code_key]
            natural_spec = sample[natural_spec_key]
            complexity = None 

            if dataset == "pyra":
                natural_spec = str(natural_spec).strip()
                natural_spec= natural_spec.replace('\\"', '"')
                json_obj = json.loads(natural_spec)
                natural_spec = json_obj["description"]
                complexity = json_obj["complexity"]
                unique_complexity.add(complexity)
                count[complexity] += 1
                # skip basic examples
                print("Sample Complexity is: ", complexity)
                if complexity == 'Basic' or complexity == 'Intermediate':
                    print("Skipping basic sample: ", i)
                    continue
        
            top_module = top_module_parser.forward({
                "rtl_code": rtl_code
            })

            output = runner.forward({
                "rtl_code": rtl_code,
                "natural_spec": natural_spec,
                "language": "Verilog",
                "top_module": top_module,
                "testbench_style": runner.config.testbench_style
            })
            print("Current sample id is: ", i)
            write_record(
                dataset_id=dataset_id,
                model=model,
                sample_id=i,
                rtl_code=rtl_code,
                natural_spec=natural_spec, 
                top_module=top_module,
                output=output,
                f_fail=f_fail,
                f_met=f_met,
                f_not_met=f_not_met,
                complexity=complexity
            )


def run_failed_coverage(
    runner,
    model,
    met_path: str ,
    met_path_iter2: str ,
    not_met_path: str,
    not_met_path_iter2: str,
    failed_path: str
):

    data = read_jsonl(failed_path)

    with open(met_path_iter2, 'a', encoding='utf-8') as f_met, \
        open(not_met_path_iter2, 'a', encoding='utf-8') as f_not_met, \
        open(failed_path, 'a', encoding='utf-8') as f_fail:
        
        for entry in tqdm.tqdm(data):
            output = runner.forward({
                    "rtl_code": entry["rtl_code"], 
                    "language": "Verilog",
                    "natural_spec": entry["natural_spec"],
                    "top_module": entry["top_module"],
                    "testbench": entry["testbench"],
                    "coverage_report": entry["coverage_report"],
                    "coverage_stats": entry["coverage_stats"],
                }, 
                start_at=DataGenNodes.Compiler
            )
            
            if output.get('covered_executes', False): 
                remove_entry_from_jsonl(failed_path, entry['sample_id'])
                write_record(
                    dataset_id=entry["dataset"],
                    model=model,
                    sample_id=entry["sample_id"],
                    rtl_code= entry["rtl_code"],
                    natural_spec=entry["natural_spec"], 
                    top_module=entry["top_module"],
                    output=output,
                    f_fail=f_fail,
                    f_met=f_met,
                    f_not_met=f_not_met
                )


def run_coverage_not_met(
    runner,
    model,
    met_path: str ,
    met_path_iter2: str,
    not_met_path: str,
    not_met_path_iter2: str,
    failed_path: str
):
    data = read_jsonl(not_met_path)
    data_sorted = sorted(
        data,
        key=_get_fsm_coverage
    )

    with open(met_path_iter2, 'a', encoding='utf-8') as f_met, \
        open(not_met_path_iter2, 'a', encoding='utf-8') as f_not_met, \
        open(failed_path, 'a', encoding='utf-8') as f_fail:
    
        for entry in tqdm.tqdm(data_sorted):
            try: 
                output = runner.forward({
                        "rtl_code": entry["rtl_code"], 
                        "language": "Verilog",
                        "natural_spec": entry["natural_spec"],
                        "top_module": entry["top_module"],
                        "testbench": entry["testbench"],
                        "coverage_report": entry["coverage_report"],
                        "coverage_stats": entry["coverage_stats"],
                        "testbench_style": entry.get("testbench_style", runner.config.testbench_style),
                    }, 
                    start_at=DataGenNodes.Enhancer
                )
            except Exception as e:
                print(f"[ERROR] Failed to enhance sample id {entry['sample_id']} with error: {str(e)}")
                continue    
                
            avg_coverage = compute_avg_coverage(entry['coverage_stats'])
            new_avg_coverage = compute_avg_coverage(output['coverage_stats'])
            
            if new_avg_coverage > avg_coverage: 
                print("[Success] Improved Coverage")
                remove_entry_from_jsonl(not_met_path, entry['sample_id'])
                write_record(
                    dataset_id=entry["dataset"],
                    model=model,
                    sample_id=entry["sample_id"],
                    rtl_code= entry["rtl_code"],
                    natural_spec=entry["natural_spec"], 
                    top_module=entry["top_module"],
                    output=output,
                    f_fail=f_fail,
                    f_met=f_met,
                    f_not_met=f_not_met
                )



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Model name', default='gpt-4o-mini-2024-07-18')
    parser.add_argument("--resume", action="store_true", help="Resume from the last checkpoint", default=True) 
    parser.add_argument('--dataset', type=str, help='Dataset name: (verithoughts/pyra)', default=None)
    parser.add_argument('--start_idx', type=int, help='Dataset name: (verithoughts/pyra)', default=None)
    parser.add_argument("--run_not_met", action="store_true", help="Run designs not meeting coverage", default=False) 
    parser.add_argument("--run_failed", action="store_true", help="Run failed designs", default=False) 
    parser.add_argument('--output', type=str, help='Path to output direcotyr', default='./output/data_generation')
    parser.add_argument(
        "--testbench_style",
        type=str,
        choices=["self_checking", "stimulus_only"],
        default="stimulus_only",
        help="Select whether the workflow generates self-checking or stimulus-only testbenches.",
    )
    parser.add_argument('--max_seq_length', type=int, help='Path to output direcotyr', default=32000)
    parser.add_argument('--max_new_tokens', type=int, help='Path to output direcotyr', default=10000)

    args = parser.parse_args()

    model = args.model 
    resume = args.resume
    dataset = args.dataset
    start_idx = args.start_idx
    output = args.output 
    run_not_met = args.run_not_met 
    run_failed = args.run_failed 
    testbench_style = args.testbench_style
    max_seq_length = args.max_seq_length
    max_new_tokens = args.max_new_tokens


    os.makedirs(output, exist_ok=True)
    met_path = os.path.join(output, "train_dataset_met_coverage.jsonl")
    not_met_path = os.path.join(output, "train_dataset_not_met_coverage.jsonl")
    failed_path = os.path.join(output, f"train_dataset_failed_cleaned.jsonl")
    
    config = FlowConfigs(
        lm_config=LMConfigs(
            prompter_lm = model,
            fixer_lm = model,
            tb_generator_lm = model,
            enhancer_lm= model,
            checker_lm = model,
            reflector_lm = model,
            curator_lm = model,
            max_seq_length = max_seq_length,
            max_new_tokens = max_new_tokens
        ),
        max_refine_iters=4,
        max_enhance_iters=4,
        graph_recursion_limit=40,
        testbench_style=testbench_style,
        data_gen_mode=True,
        wo_reflector=False,
        wo_curator=False
    )

    runner = DataGenRunner(
        config=config
    )

    top_module_parser = TopParser(
        config = config 
    )

    if dataset and not run_not_met and not run_failed: 
        print("Running on dataset: ", dataset)
        run_dataset(
            runner=runner,
            top_module_parser=top_module_parser,
            model=model,
            dataset=dataset,
            start_idx=start_idx,
            resume=resume,
            met_path=met_path ,
            not_met_path=not_met_path,
            failed_path=failed_path
        )
    elif run_not_met: 
        print("Trying to enhance coverage on: ", not_met_path)
        not_met_path_iter_2 = os.path.join(output, "train_dataset_not_met_coverage_iter_2.jsonl")
        met_path_iter_2 = os.path.join(output, "train_dataset_met_coverage_iter_2.jsonl")
        run_coverage_not_met(
            runner=runner,
            model=model,
            met_path=met_path ,
            met_path_iter2=met_path_iter_2,
            not_met_path=not_met_path,
            not_met_path_iter2=not_met_path_iter_2,
            failed_path=failed_path
        )
    
    elif run_failed: 
        print("Running on failed designs: ", failed_path)
        not_met_path_iter_2 = os.path.join(output, "train_dataset_not_met_coverage_iter_3.jsonl")
        met_path_iter_2 = os.path.join(output, "train_dataset_met_coverage_iter_3.jsonl")
        run_failed_coverage(
            runner=runner,
            model=model,
            met_path=met_path ,
            met_path_iter2=met_path_iter_2 ,
            not_met_path=not_met_path,
            not_met_path_iter2=not_met_path_iter_2,
            failed_path=failed_path
        )
    else:
        print("Please specify a dataset to run on or either pass run_not_met or run_failed arguments")

   
            

if __name__ == '__main__':
    main()
