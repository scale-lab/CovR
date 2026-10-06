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


"""Runs covered to report testbench coverage.
"""
import os 
import re 
import json
import shutil
import subprocess 
import argparse
import tempfile 
from pathlib import Path
from typing import Dict


def get_tb_top_module(design_path: str) -> str:
    tb_top = None
    module_pattern = re.compile(r'^\s*module\s+([A-Za-z_]\w*)')

    with open(design_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = module_pattern.search(line)
            if m:
                tb_top = m.group(1)      

    if tb_top is None:
        print("[Error] No module found in", design_path)
    else:
        print("Testbench top module:", tb_top)

    return tb_top


def fix_vcd(
    vcd_file: str 
): 
    vcd_path = Path(vcd_file)
    if not vcd_path.exists():
        raise FileNotFoundError(f"VCD file not found: {vcd_file}")
   
    text = vcd_path.read_text(encoding="utf-8", errors="ignore")
    pattern = re.compile(r'^\s*\$comment\b[\s\S]*?^\s*\$end\s*$', flags=re.MULTILINE)
    cleaned = pattern.sub("", text)
    vcd_path.write_text(cleaned, encoding="utf-8")


def run_covered(
    verilog_files: list[str],
    testbench: str,
    vcd_file: str,
    output_dir: str,
    top_module_name: str,
    top_module_tree_path: str,
    singularity_image: str = "sing.sif",
):
    covered_executable = shutil.which("covered")

    os.makedirs(output_dir, exist_ok=True)
    
    fix_vcd(vcd_file=vcd_file)

    file_name = os.path.basename(testbench)
    cdd_file = os.path.join(output_dir, f"{file_name}.cdd")
    report_file =  os.path.join(output_dir, f"{file_name}.txt")
    
    verilog_args = [arg for f in verilog_files for arg in ("-v", f)]

    if covered_executable:
        score_cmd = ["covered", "score", "-t", top_module_name,  "-i", top_module_tree_path, "-vcd", vcd_file] + verilog_args + ["-o", cdd_file]
        report_cmd = ["covered", "report", "-d", "v", "-o", report_file, cdd_file] 

    else:
        score_cmd = ["singularity", "exec", singularity_image, "covered", "score", "-t", top_module_name,  "-i", top_module_tree_path, "-vcd", vcd_file, "-v", testbench] + verilog_args + ["-o", cdd_file]
        report_cmd = ["singularity", "exec", singularity_image, "covered", "report", "-d", "d", "-o", report_file, cdd_file] 

    try: 
        score_result = subprocess.run(score_cmd, capture_output=True, text=True)

        if score_result.stderr:
            print(f"[WARNING] Running score command {testbench}: {score_result.stderr}") 
    except Exception as e: 
        print(f"[ERROR]: {e}")
        return False, ""

    try: 
        report_result = subprocess.run(report_cmd, capture_output=True, text=True)

        if report_result.stderr: 
            print(f"[Error] Running report command {testbench}: {report_result.stderr}") 
            return False, ""  
    except Exception as e: 
        print(f"[ERROR]: {e}")
        return False, ""
    
    return True, report_file


def run_testbench(
    verilog_files: list[str],
    testbench: str,
    output_dir: str,
    singularity_image: str = "sing.sif",
): 
    iverilog_executable = shutil.which("iverilog")
    
    os.makedirs(output_dir, exist_ok=True)

    file_name = os.path.basename(testbench)
    out_file = os.path.join(f"{file_name}.out")

    if iverilog_executable:
        compile_cmd = ["iverilog", "-g2012", "-o", out_file, testbench] + verilog_files
        run_cmd = ["vvp", out_file]
    else:
        compile_cmd = [
            "singularity", 
            "exec",
            "-B", "/oscar/data:/oscar/data",
            "-B", f"{os.getcwd()}:{os.getcwd()}",
            "-W", output_dir,                           
            singularity_image, 
            "iverilog", 
            "-g2012", 
            "-o", out_file, 
            testbench
        ] + verilog_files
        run_cmd = ["singularity", "exec", singularity_image, "vvp", out_file]

    compile_result = subprocess.run(
        compile_cmd, 
        cwd=output_dir,
        capture_output=True, 
        text=True
    )
    if compile_result.stderr:
        print(f"[Error] Running iverilog: {compile_result.stderr + compile_result.stdout}")
        return False, compile_result.stderr +  compile_result.stdout
    
    try:
        run_result = subprocess.run(
            run_cmd, 
            cwd=output_dir,
            capture_output=True, 
            text=True, 
            timeout=60*4
        )
    except subprocess.TimeoutExpired:
        print(f"Simulation timed out after {60*4} seconds")
        return False, f"Simulation timed out after {60*4} seconds"

    if run_result.stderr:
        print(f"[Error] Running vvp: {run_result.stderr + run_result.stdout}")
        return False, run_result.stderr +  run_result.stdout
    
 
    if run_result.stderr or "ERROR" in run_result.stdout or "Error" in run_result.stdout:
        print(f"[Error] Running vvp: {run_result.stderr + run_result.stdout}")
        return False, run_result.stderr +  run_result.stdout
    
    return True, run_result.stdout


def run_ivl(
    verilog_files: list[str],
    testbench: str,
    output_dir: str,
    singularity_image: str = "sing.sif",
):
    iverilog_executable = shutil.which("iverilog")
    
    os.makedirs(output_dir, exist_ok=True)

    file_name = os.path.basename(testbench)
    out_file = os.path.join(f"{file_name}.out")

    if iverilog_executable:
        compile_cmd = ["iverilog", "-g2012", "-o", out_file, testbench] + verilog_files
    else:
        compile_cmd = [
            "singularity", 
            "exec",
            "-B", "/oscar/data:/oscar/data",
            "-B", f"{os.getcwd()}:{os.getcwd()}",
            "-W", output_dir,                           
            singularity_image, 
            "iverilog", 
            "-g2012", 
            "-o", out_file, 
            testbench
        ] + verilog_files

    compile_result = subprocess.run(
        compile_cmd, 
        cwd=output_dir,
        capture_output=True, 
        text=True
    )
    if compile_result.stderr:
        print(f"[Error] Running iverilog: {compile_result.stderr + compile_result.stdout}")
        return False, compile_result.stderr +  compile_result.stdout 

    return True, compile_result.stdout


def run_vvp(
    testbench,
    output_dir,
    singularity_image  
):
    iverilog_executable = shutil.which("iverilog")

    file_name = os.path.basename(testbench)
    out_file = os.path.join(f"{file_name}.out")

    if iverilog_executable:
        run_cmd = ["vvp", out_file]
    else:
       
        run_cmd = ["singularity", "exec", singularity_image, "vvp", out_file]

    try:
        run_result = subprocess.run(
            run_cmd, 
            cwd=output_dir,
            capture_output=True, 
            text=True, 
            timeout=60*4
        )
    except subprocess.TimeoutExpired:
        print(f"Simulation timed out after {60*4} seconds")
        return False, f"Simulation timed out after {60*4} seconds"

    if run_result.stderr or "ERROR" in run_result.stdout or "Error" in run_result.stdout:
        print(f"[Error] Running vvp: {run_result.stderr + run_result.stdout}")
        return False, run_result.stderr +  run_result.stdout
    
    return True, run_result.stdout 



def report_coverage(report_file: str) -> Dict[str, float]:
    report_path = Path(report_file)
    
    if not report_path.exists():
        raise FileNotFoundError(f"Report file not found: {report_file}")

    text = report_path.read_text(encoding="utf-8", errors="ignore")

    line_re   = re.compile(r'Accumulated.*?(\d+/\s*\d+/\s*\d+)\s+(\d+)%', re.MULTILINE)
    toggle_re = re.compile(r'Accumulated.*?(\d+/\s*\d+/\s*\d+)\s+(\d+)%.*?(\d+/\s*\d+/\s*\d+)\s+(\d+)%', re.MULTILINE|re.DOTALL)

    coverage = {}

    line_section = re.search(r'LINE COVERAGE RESULTS.*?Accumulated.*', text, re.DOTALL)
    if line_section:
        m = re.search(r'Accumulated.*?(\d+/\s*\d+/\s*\d+)\s+(\d+)%', line_section.group())
        if m:
            coverage['line'] = float(m.group(2))

    toggle_section = re.search(r'TOGGLE COVERAGE RESULTS.*?Accumulated.*', text, re.DOTALL)
    if toggle_section:
        m = re.search(r'Accumulated.*?(\d+/\s*\d+/\s*\d+)\s+(\d+)%', toggle_section.group())
        if m:
            coverage['toggle'] = float(m.group(2))

    comb_section = re.search(r'COMBINATIONAL LOGIC COVERAGE RESULTS.*?Accumulated.*', text, re.DOTALL)
    if comb_section:
        m = re.search(r'Accumulated.*?(\d+/\s*\d+/\s*\d+)\s+(\d+)%', comb_section.group())
        if m:
            coverage['combinational'] = float(m.group(2))

    fsm_section = re.search(r'FINITE STATE MACHINE COVERAGE RESULTS.*?Accumulated.*', text, re.DOTALL)
    if fsm_section:
        m = re.search(r'Accumulated.*?(\d+/\s*\d+/\s*\d+)\s+(\d+)%', fsm_section.group())
        if m:
            coverage['fsm'] = float(m.group(2))

    fsm_match = re.search(
        r'FINITE STATE MACHINE COVERAGE RESULTS.*?Accumulated.*?(\d+)\s*/\s*(\d+)\s*/\s*(\d+)',
        text,
        flags=re.DOTALL
    )

    has_fsm = False
    if fsm_match:
        total_states = int(fsm_match.group(3))
        if total_states > 0:
            has_fsm = True

    coverage['fsm_present'] = has_fsm

    return coverage



def main():
        
    tb_result = run_testbench(
        verilog_files = ["/oscar/data/sreda/mabdelat/verilog-eval/dataset_spec-to-rtl/Prob021_mux256to1v_ref.sv", "/oscar/data/sreda/mabdelat/LLM4TB/data/Prob021_mux256to1v_ref.sv"],
        testbench="/oscar/data/sreda/mabdelat/verilog-eval/dataset_spec-to-rtl/Prob021_mux256to1v_test.v",
        singularity_image="/oscar/data/sreda/mabdelat/LLM4TB/sing.sif",
        output_dir="./output"
    ) 

    covered_result, report_file = run_covered(
        verilog_files = ["/oscar/data/sreda/mabdelat/verilog-eval/dataset_spec-to-rtl/Prob021_mux256to1v_ref.sv", "/oscar/data/sreda/mabdelat/LLM4TB/data/Prob021_mux256to1v_ref.sv"],
        testbench="/oscar/data/sreda/mabdelat/verilog-eval/dataset_spec-to-rtl/Prob021_mux256to1v_test.v",
        vcd_file="/oscar/data/sreda/mabdelat/LLM4TB/output/wave.vcd",
        singularity_image="/oscar/data/sreda/mabdelat/LLM4TB/sing.sif",
        output_dir="./output"

    )

    stats = report_coverage(
        report_file=report_file
    )

    print(stats)


if __name__ == '__main__':
    main()