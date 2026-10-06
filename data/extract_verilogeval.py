import os 
import re 
import json

def normalize_VerilogEval_dataset(dataset_root, output_path):

    normalized_data = []
    problems_file = os.path.join(dataset_root, "problems.txt")

    task_counter = 0

    with open(problems_file, 'r') as f:
        for line in f:
            problem_name = line.strip()
            if not problem_name:
                continue
            task_counter += 1

            prompt_file = os.path.join(dataset_root, f"{problem_name}_prompt.txt")
            code_file   = os.path.join(dataset_root, f"{problem_name}_ref.sv")
            testbench_file   = os.path.join(dataset_root, f"{problem_name}_test.sv")

            with open(prompt_file, 'r') as pf:
                prompt = pf.read().strip()

            with open(code_file, 'r') as cf:
                verilog_code = cf.read().strip()

            top_module_code = re.sub(r'\bmodule\s+\w+', "module TopModule", verilog_code)
            with open(testbench_file, 'r') as cf:
                testbench = cf.read().strip()

            testbench = fix_dump_vars(testbench=testbench)
            task = {
                "task_id": problem_name,
                "task_number": task_counter,
                "description": prompt,
                "ref_module": verilog_code,
                "top_module": top_module_code,
                "testbench": testbench
            }

            normalized_data.append(task)

    with open(output_path, "w", encoding="utf-8") as out_f:
        for item in normalized_data:
            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"VerilogEval normalization complete. Saved to {output_path}")




def fix_dump_vars(testbench: str, module: str) -> str:
    # Clean previous dump directives
    tb = re.sub(r'\$dumpfile\s*\([^)]*\)\s*;\s*', '', testbench)
    tb = re.sub(r'\$dumpvars\s*\([^)]*\)\s*;\s*', '', tb)

    # Remove empty initial blocks like "initial begin end"
    tb = re.sub(r'initial\s+begin\s*end\s*', '', tb, flags=re.IGNORECASE)

    new_block = (
        "    // inserted by fix_dump_vars\n"
        "    initial begin\n"
        f"        $dumpfile(\"wave.vcd\");\n"
        f"        $dumpvars(0, {module});\n"
        "    end\n\n"
    )

    # Find complete module blocks (non-greedy to the matching endmodule)
    mod_block = re.compile(r'(?ms)^\s*module\b.*?^\s*endmodule\b')
    matches = list(mod_block.finditer(tb))
    if not matches:
        # Fallback: append at end (no module found)
        return tb + "\n" + new_block

    # Assume last module is the testbench
    last = matches[-1]
    block_text = tb[last.start():last.end()]

    # Find the end of the module header (first ';' after 'module ...')
    header_pat = re.compile(r'(?ms)^\s*module\b.*?;\s*')
    header_match = header_pat.search(block_text)
    if not header_match:
        # If header couldn't be parsed, append at the end of the block
        insert_pos = last.end()
        return tb[:insert_pos] + "\n" + new_block + tb[insert_pos:]

    # Insert right after the header semicolon
    insert_pos = last.start() + header_match.end()
    return tb[:insert_pos] + new_block + tb[insert_pos:]


def main():
    normalize_VerilogEval_dataset(
        dataset_root="/oscar/data/sreda/mabdelat/verilog-eval/dataset_spec-to-rtl",
        output_path="/oscar/data/sreda/mabdelat/LLM4TB/data/verilog_eval.json"
    ) 


if __name__ == '__main__':
    main()