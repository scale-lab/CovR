"""Clean dataset from invalid testbenches or rtl code
"""
import os 
import json 
import tqdm
import argparse

from core.data_generation.utils import read_jsonl, remove_entry_from_jsonl
from core.data_generation.config.config import FlowConfigs, LMConfigs
from core.data_generation.self_checking.checker import Checker
from core.data_generation.compiler import Compiler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Model name', default='gpt-4o-mini-2024-07-18')
    parser.add_argument('--input_dir', type=str, help='Model name', default='gpt-4o-mini-2024-07-18')
    args = parser.parse_args() 

    model = args.model
    input_dir = args.input_dir

    # met_json = os.path.join(input_dir, "train_dataset_met_coverage.jsonl")
    # not_met_json = os.path.join(input_dir, "train_dataset_not_met_coverage.jsonl" )
    
    output_file = os.path.join(input_dir, "self_checking_running_samples.jsonl")

    all_merged_json =  os.path.join(input_dir, "all_merged.jsonl")
    met_merged_json = os.path.join(input_dir, "met_merged.jsonl")
    not_met_merged_json = os.path.join(input_dir, "not_met_merged.jsonl")

    config = FlowConfigs(
        lm_config=LMConfigs(
            prompter_lm = model,
            fixer_lm = model,
            tb_generator_lm = model,
            enhancer_lm = model,
            checker_lm = model,
            refiner_lm = model
        )
    ) 

    checker = Checker(
        config=config
    )

    compiler = Compiler(
        config=config
    )

    for input_file in [all_merged_json, met_merged_json, not_met_merged_json]: 
        data = read_jsonl(input_file)
        for i, entry in enumerate(tqdm.tqdm(data)): 
        
            sample_id = entry["sample_id"]
            rtl_code = entry["rtl_code"]
            testbench = entry["testbench"]
            top_module = entry["top_module"]

            try: 
                output = checker.forward({
                    "testbench": testbench
                })

                if output["is_self_checking"] and output["is_print_error"]: 
                    compiler_output = compiler.forward({
                        "testbench": testbench,
                        "rtl_code": rtl_code,
                        "top_module": top_module,
                        "language": "Verilog"
                    })

                    if compiler_output["testbench_runs"]: 
                        with open(output_file, "a") as f: 
                            f.write(json.dumps(entry) + "\n")
            except: 
                with open(output_file, "a") as f: 
                    f.write(json.dumps(entry) + "\n")

            # if "Error:" in testbench: 
            #     pass 
            # else:
            #     print("Sample id: ", sample_id, "is not self_checking")
            #     print(testbench)
            #     # remove_entry_from_jsonl(input_file, sample_id=sample_ids)

            #     with open(output_file, "a") as f: 
            #         f.write(json.dumps(entry) + "\n")


if __name__ == '__main__':
    main()