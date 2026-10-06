"""Remove the VCD dump files from the testbenches
"""

import os 
import re
import json 
import argparse

from core.data_generation.utils import read_jsonl
from core.data_generation.config.config import FlowConfigs, LMConfigs   
from core.data_generation.compiler import Compiler
from core.metrics.vcs import run_vcs

def remove_vcd_dump_statements(testbench: str) -> str:
    if not testbench:
        return testbench

    cleaned = testbench

    # Remove VCD-related lines
    patterns = [
        r'^\s*\$dumpfile\s*\([^;]*\)\s*;\s*$',
        r'^\s*\$dumpvars\s*\([^;]*\)\s*;\s*$',
        r'^\s*\$dumpvars\s*;\s*$',
        r'^\s*\$dumpon\s*;\s*$',
        r'^\s*\$dumpoff\s*;\s*$',
    ]

    for pattern in patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.MULTILINE)

    # Remove empty initial blocks like:
    # initial begin
    #
    # end
    cleaned = re.sub(
        r'initial\s+begin\s*(?:\n|\r\n|\r|\s)*end',
        '',
        cleaned,
        flags=re.MULTILINE
    )

    # Collapse excessive blank lines
    cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)

    return cleaned.strip() + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Model name', default='gpt-4o-mini-2024-07-18')
    parser.add_argument('--input_dir', type=str, help='Model name', default='gpt-4o-mini-2024-07-18')
    parser.add_argument('--output_dir', type=str, help='Model name', default='gpt-4o-mini-2024-07-18')

    args = parser.parse_args() 

    model = args.model
    input_dir = args.input_dir
    output_dir = args.output_dir

    all_json =  os.path.join(input_dir, "all_merged.jsonl")
    met_json = os.path.join(input_dir, "met_merged.jsonl")
    not_met_json = os.path.join(input_dir, "not_met_merged.jsonl" )

    os.makedirs(output_dir, exist_ok=True)

    output_all_json = os.path.join(output_dir, "all_merged.jsonl")
    output_met_json = os.path.join(output_dir, "met_merged.jsonl" )
    output_not_met_json = os.path.join(output_dir, "not_met_merged.jsonl" )

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

    compiler = Compiler(
        config=config
    )
            
    for input_file, output_file in zip([all_json, met_json, not_met_json], [output_all_json, output_met_json, output_not_met_json]): 
        data = read_jsonl(input_file)
        for entry in data:
            testbench = entry["testbench"]
            rtl_code = entry["rtl_code"]
            top_module = entry["top_module"]
            testbench_cleaned = remove_vcd_dump_statements(entry.get("testbench", ""))

            # run compiler
            # compiler_output = compiler.forward({
            #     "testbench": testbench_cleaned,
            #     "rtl_code": rtl_code,
            #     "top_module": top_module,
            #     "language": "Verilog"
            # })
            old_tb_file_path = os.path.join(config.output_dir, f"testbench_{top_module}.v")
            with open(old_tb_file_path, "w") as f:
                f.write(testbench)
            
            tb_file_path = os.path.join(config.output_dir, f"testbench_{top_module}.v")
            with open(tb_file_path, "w") as f:
                f.write(testbench_cleaned)
            
            rtl_file_path = os.path.join(config.output_dir, f"rtl_{top_module}.v")
            with open(rtl_file_path, "w") as f:
                f.write(rtl_code)
            

            old_correct, _ = run_vcs(
                verilog_files=[rtl_file_path],
                testbench=old_tb_file_path,
                top_module_name=top_module,
                output_dir=config.output_dir
            ) 

            correct, result = run_vcs(
                verilog_files=[rtl_file_path],
                testbench=tb_file_path,
                top_module_name=top_module,
                output_dir=config.output_dir
            ) 

            if correct or not old_correct:
                entry["testbench"] = testbench_cleaned
            else:
                print("Tesbtench failed")
                print(testbench_cleaned)
                import sys 
                sys.exit(0)
            # write to output jsonl 
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")


if __name__ == '__main__':
    main()