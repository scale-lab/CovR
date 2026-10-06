"""Runs VCS and Verdi 
"""

import re 
import os 
import math
import argparse
import shutil
import subprocess
import time

from core.data_generation.utils import compute_avg_coverage

BANNER_BLOCK_RE = re.compile(
    r"(?is)"                   
    r"^[ \t]*Chronologic\s+VCS.*?"  
    r"solvnetplus\.synopsys\.com.*?\n",  
)

def strip_vcs_banner(text: str) -> str:
    return BANNER_BLOCK_RE.sub("", text)



VCS_BANNER_RE = re.compile(
    r"(?is)"  
    r"^Info:\s*\[VCS_SAVE_RESTORE_INFO\].*?"       
    r"VCS\s+Coverage\s+Metrics\s+Release.*?by\s+Synopsys\s+Inc\..*?\n",  #
)
FOOTER_RE = re.compile(
    r"(?ims)"
    r"^-{3,}.*?\r?\n"                                         # dashed separator
    r".*?VCS\s+Coverage\s+Metrics:.*?\r?\n"                   # the "VCS Coverage Metrics:" line
    r"(?:.*?\r?\n)*?"                                         # the block body (coverage status, report title, etc.)
    r"^Time:.*?\r?\n"                                         # Time line
    r"^(?:CPU\s*Time:.*?\r?\n)?"                              # optional CPU Time
    r"^(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\s+\d{4}\s*\r?\n?",
)

def strip_sim_banner(text: str) -> str:
    out = VCS_BANNER_RE.sub("", text)
    out = FOOTER_RE.sub("", out)
    lines = [ln for ln in out.splitlines() if ln.strip()]
    return "\n".join(lines) + "\n"

def run_vcs(
    verilog_files,
    testbench: str,
    top_module_name: str,
    output_dir: str,
    timeout = 60 * 4
):
    try:
        with open(testbench, "r", encoding="utf-8", errors="replace") as f:
            testbench_content = f.read()
    except OSError as exc:
        return False, f"Could not read testbench: {exc}"

    if not testbench_content.strip():
        return False, "Empty testbench"

    vcs_executable = shutil.which("vcs")
    
    if not vcs_executable:
        print("[Error] VCS not found in PATH. Please load Synopsys VCS before running.")
        return False, "VCS not found in PATH"

    os.makedirs(output_dir, exist_ok=True)

    file_name = os.path.basename(testbench)    
    simv_path =  f"{file_name}.simv"
    cov_db_dir =  f"covdb_{top_module_name}"

    compile_cmd = [
        "vcs", 
        "+vcs+lic+wait",
        "-full64",
        "-sverilog",
        "-cm", "line+cond+fsm+tgl+branch",
        "-debug_access+all", 
        "-cm_dir", cov_db_dir, 
        testbench, 
        "-o", 
        simv_path
    ] + verilog_files
    start_time = time.perf_counter()

    try: 
        compile_result = subprocess.run(
            compile_cmd,
            cwd=output_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,   
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout
        )
        # print(compile_result.stderr +  compile_result.stdout )
    except subprocess.TimeoutExpired:
        print(f"Simulation timed out after {timeout} seconds")
        return False, f"Simulation timed out after {timeout} seconds"

    if compile_result.returncode != 0 or compile_result.stderr or "Error-" in compile_result.stdout:
        # print(f"[Error] Running vcs: {compile_result.stderr + compile_result.stdout}")
        return False, compile_result.stderr +  compile_result.stdout 

    return True, compile_result.stdout



def run_sim(
    out_file,
    top_module_name,
    output_dir,
    timeout = 60 * 4
):
    if not os.path.exists(out_file):
        print("[ERROR] out file doesn't exist!")
        return False, "[ERROR] out file doesn't exist!"

    cov_db_dir = os.path.join(output_dir, f"covdb_{top_module_name}")

    run_cmd = [f"{out_file}", "+vcs+lic+wait" , "-cm", "line+cond+fsm+tgl+branch"] 
    start_time = time.perf_counter()
    
    try:
        compile_result = subprocess.run(
            run_cmd, 
            cwd=output_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,   
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout
        )
        # print(compile_result.stderr +  compile_result.stdout )
    except subprocess.TimeoutExpired:
        print(f"[ERROR] Running simv timed out after {timeout} seconds")
        return False, f"Running the simulation timed out after {timeout} seconds."
    except PermissionError as exc:
        return False, f"Running the simulation failed with permission error: {exc}"

    if compile_result.returncode != 0 or compile_result.stderr:
        print(f"[Error] Running simulation: {compile_result.stderr + compile_result.stdout}")
        return False, compile_result.stderr + compile_result.stdout

    return True, compile_result.stdout


def run_testbench_vcs(
    verilog_files,
    testbench: str,
    output_dir: str,
    top_module_name =  None
):
    correct, result = run_vcs(
        verilog_files=verilog_files,
        testbench=testbench,
        top_module_name=top_module_name,
        output_dir=output_dir
    ) 
    
    result = strip_vcs_banner(result)

    if not correct:
        return correct, result
    
    file_name = os.path.basename(testbench)    
    simv_path = os.path.join(output_dir, f"{file_name}.simv")

    correct, result = run_sim(
        out_file=simv_path,
        output_dir=output_dir,
        top_module_name=top_module_name
    )
    
    result = strip_sim_banner(result)

    return correct, result


def _generate_heir(testbench: str, output_path: str):

    with open(testbench, "r") as f:
        content = f.read()

    tb_match = re.search(r"module\s+(\w+)", content)
    testbench_module = tb_match.group(1) if tb_match else None

    # Parse DUT instance (e.g., "pipelined_multiplier uut (")
    inst_match = re.search(r"(\w+)\s+(\w+)\s*\(", content)
    instance_name = inst_match.group(2) if inst_match else None


    template = f"""
// Exclude coverage for module testbench
-module {testbench_module}

// Include coverage for instance 
+tree {testbench_module}.{instance_name}
"""

    with open(output_path, "w") as f:
        f.write(template)

    return output_path


def run_urg(testbench, top_module_name, output_dir, timeout=60*5):
    urg_exe = shutil.which("urg")
    if not urg_exe:
        return False, "[ERROR] urg not found. Is Synopsys env loaded?"

    vdb_path = os.path.join(output_dir, f"covdb_{top_module_name}.vdb")
    if not os.path.exists(vdb_path):
        return False, f"[ERROR] Coverage DB not found at {vdb_path}"

    report_dir = os.path.join(output_dir, f"cov_report_{top_module_name}")
    os.makedirs(report_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"heir_{top_module_name}.txt")
    hier_file = _generate_heir(testbench, output_path)

    urg_cmd = [
        "urg",
        "-dir", vdb_path,
        "-format", "both",           
        "-hier", hier_file,
        "-report", report_dir,
    ]
    start_time = time.perf_counter()

    try: 
        res = subprocess.run(
            urg_cmd, 
            capture_output=True, 
            text=True,
            timeout=timeout
        )
    except subprocess.TimeoutExpired:
        # print(f"[ERROR] Running URG timed out after {timeout} seconds. ")
        return False, f"Running URG timed out after {timeout} seconds."

    if res.returncode != 0:
        return False, f"[Error] urg failed:\n{res.stderr}{res.stdout}"

    report_file = os.path.join(report_dir, "modinfo.txt")
    
    return True, report_file


def report_coverage_vcs(report_file: str):
    def _to_float(x: str):
        return None if x == "--" else float(x)

    pattern = re.compile(
        r"SCORE\s+LINE\s+COND\s+TOGGLE\s+FSM\s+BRANCH(?:\s+NAME)?\s*\n"
        r"\s*(?P<score>--|[\d.]+)\s+"
        r"(?P<line>--|[\d.]+)\s+"
        r"(?P<cond>--|[\d.]+)\s+"
        r"(?P<toggle>--|[\d.]+)\s+"
        r"(?P<fsm>--|[\d.]+)\s+"
        r"(?P<branch>--|[\d.]+)"
    )

    with open(report_file, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    matches = list(pattern.finditer(text))
    if not matches:
        raise Exception("INVALID Coverage Report: ", text)
        # return {"line": None, "cond": None, "toggle": None, "fsm": None, "branch": None}

    last = matches[-1]
    
    coverage = {
        "line": _to_float(last.group("line")),
        "cond": _to_float(last.group("cond")),
        "toggle": _to_float(last.group("toggle")),
        "fsm": _to_float(last.group("fsm")),
        "branch": _to_float(last.group("branch"))
    }

    avg_score = _to_float(last.group("score"))
    computed = round(compute_avg_coverage(coverage), 2) 
    assert math.isclose(computed, avg_score, rel_tol=1e-3, abs_tol=0.02), (
        f"Mismatch!\ncomputed={computed}\nexpected={avg_score}",
        f"Coverage report is: {text}",
        f"Parsed stats are: {coverage}"
    )

    return coverage


def main():
    parser = argparse.ArgumentParser(description="Run VCS compile/sim and optional URG coverage parsing.")
    parser.add_argument("--rtl", action="append", required=True, help="RTL Verilog/SystemVerilog file. Repeat for multiple files.")
    parser.add_argument("--testbench", required=True, help="Generated testbench file.")
    parser.add_argument("--top-module", required=True, help="Top DUT module name.")
    parser.add_argument("--output-dir", default="./sim", help="Directory for VCS/URG artifacts.")
    parser.add_argument("--no-urg", action="store_true", help="Skip URG coverage reporting.")
    args = parser.parse_args()

    correct, result = run_testbench_vcs(
        verilog_files=[os.path.abspath(path) for path in args.rtl],
        testbench=os.path.abspath(args.testbench),
        top_module_name=args.top_module,
        output_dir=os.path.abspath(args.output_dir)
    )
    print(correct)
    print(result)

    if not correct or args.no_urg:
        return

    urg_ok, report_file = run_urg(
        output_dir=os.path.abspath(args.output_dir),
        top_module_name=args.top_module,
        testbench=os.path.abspath(args.testbench),
    )

    if not urg_ok:
        print(report_file)
        return

    coverage = report_coverage_vcs(
        report_file=report_file
    )
    print(coverage)



if __name__ == '__main__':
    main()
