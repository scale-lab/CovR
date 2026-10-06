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

"""Computes pass@k (syntax & functionality) and cov@k (testbench coverage across four dims: line, toggle, fsm, and average)
"""

import os 
import json 
import time
import tempfile
import argparse
import shutil
import numpy as np 

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import partial

from core.metrics.coverage import run_ivl, run_vvp
from core.metrics.vcs import run_testbench_vcs, run_vcs, run_urg, run_sim, report_coverage_vcs
from core.metrics.coverage import run_covered
from core.utils import get_logger, write_answers, write_ground_truth
from core.data_generation.utils import read_jsonl


def estimator(n: int, c: int, k: int) -> float:
    """
    Calculates 1 - comb(n - c, k) / comb(n, k).
    """
    if n - c < k:
        return 1.0 
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def cov_estimator(e: list, n: int, k: int):
    lbd = [k / n]
    k_ = k - 1
    for r in range(n - 1, k_, -1):
        lbd.append(lbd[-1] * (1 - k_ / r))
    lbd = np.flip(lbd)
    e = np.sort(e)[k_ :]
    return (lbd * e).sum()


def count_correct_tb(
    samples,
    rtl_code,
    top_module_name,
    output_dir
):
    num_correct_syntax = 0
    num_correct_func = 0

    for sample_tb in samples:
        correct_syntax, _ = run_vcs(
            verilog_files=[rtl_code],
            testbench=sample_tb,
            top_module_name=top_module_name,
            output_dir=output_dir
        )
        
        print(correct_syntax)
        file_name = os.path.basename(sample_tb)    
        simv_path = os.path.join(output_dir, f"{file_name}.simv")

        print(simv_path)
        if correct_syntax: 
            num_correct_syntax += 1 

            with open(sample_tb, "r") as f:
                testbench_content = f.read()

                correct_func, _ = run_sim(
                    out_file=simv_path,
                    top_module_name=top_module_name,
                    output_dir=output_dir
                )
         
        else:
            correct_func = False

        if correct_func: 
            num_correct_func += 1

    return num_correct_syntax, num_correct_func


def _check_one_tb(sample_tb, rtl_code, top_module_name):
    output_dir = tempfile.mkdtemp(prefix="out_pass_")
    try:
        correct_syntax, _ = run_vcs(
            verilog_files=[rtl_code],
            testbench=sample_tb,
            top_module_name=top_module_name,
            output_dir=output_dir
        )

        file_name = os.path.basename(sample_tb)
        simv_path = os.path.join(output_dir, f"{file_name}.simv")

        if correct_syntax:
            correct_func, _ = run_sim(
                out_file=simv_path,
                top_module_name=top_module_name,
                output_dir=output_dir
            )
        else:
            correct_func = False

        return correct_syntax, correct_func
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def count_correct_tb_parallel(
    samples,
    rtl_code,
    top_module_name,
    *,
    backend: str = "thread",
    max_workers: int = os.cpu_count() - 1,
    chunksize: int = 1
):
    if max_workers is None or max_workers < 1:
        max_workers = 1

    worker = partial(
        _check_one_tb,
        rtl_code=rtl_code,
        top_module_name=top_module_name
    )
    Executor = ProcessPoolExecutor if backend == "process" else ThreadPoolExecutor

    num_correct_syntax = 0
    num_correct_func = 0

    with Executor(max_workers=max_workers) as ex:
        if backend == "process":
            results_iter = ex.map(worker, samples, chunksize=chunksize)
        else:
            results_iter = ex.map(worker, samples)

        for correct_syntax, correct_func in results_iter:
            if correct_syntax:
                num_correct_syntax += 1
            if correct_func:
                num_correct_func += 1

    return num_correct_syntax, num_correct_func



def compute_pass_at_k(
    answers: list[list],
    rtl_code: list,
    top_module_name: list,
    k_list: list
):
    """
    Compute pass@k value; syntax and functional correctness only at different K values
    """

    avg_pass_at_k_syntax = [0] * len(k_list)
    avg_pass_at_k_func = [0] * len(k_list)

    num_problems = len(answers)
    for i, answer in enumerate(answers):
        pass_at_k_syntax, pass_at_k_func = _compute_one_problem(
            answer=answer,
            rtl_code=rtl_code[i],
            top_module_name=top_module_name[i],
            k_list=k_list,
            output_dir=tempfile.mkdtemp(prefix="out_"),
            parallel=False
        )

        for j, _ in enumerate(k_list): 
            avg_pass_at_k_syntax[j] += pass_at_k_syntax[j]
            avg_pass_at_k_func[j] += pass_at_k_func[j]

    for i in range(len(k_list)): 
        avg_pass_at_k_syntax[i] = avg_pass_at_k_syntax[i] / num_problems
        avg_pass_at_k_func[i] = avg_pass_at_k_func[i] / num_problems
    
    return avg_pass_at_k_syntax, avg_pass_at_k_func


def _compute_one_problem(
    answer,
    rtl_code,
    top_module_name,
    k_list,
    output_dir,
    *,
    parallel=False,
    backend: str = "thread",
    max_workers: int = os.cpu_count() - 1,
    chunksize: int = 1
):
    """
    Returns a list[float] of pass@k for this single problem (one entry per k).
    """
    out_dir = output_dir
    try:
        num_samples = len(answer)
        if parallel:
            num_correct_syntax, num_correct_func = count_correct_tb_parallel(
                samples=answer,
                rtl_code=rtl_code,
                top_module_name=top_module_name,
                backend=backend,
                max_workers=max_workers,
                chunksize=chunksize
            )
        else:
            num_correct_syntax, num_correct_func = count_correct_tb(
                samples=answer,
                rtl_code=rtl_code,
                top_module_name=top_module_name,
                output_dir=out_dir
            )
        # Compute pass@k for all ks for this problem
        out_syntax = []
        out_func = []
        for k in k_list:
            pass_at_k_syntax = estimator(
                n=num_samples,
                c=num_correct_syntax,
                k=k
            )
            pass_at_k_func = estimator(
                n=num_samples,
                c=num_correct_func,
                k=k
            )
            out_syntax.append(pass_at_k_syntax)
            out_func.append(pass_at_k_func)

        return out_syntax, out_func
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def compute_pass_at_k_parallel(
    answers: list[list],
    rtl_code: list,
    top_module_name: list,
    k_list: list,
    *,
    backend: str = "thread",     # "process" for CPU-bound, "thread" for I/O/subprocess-heavy
    max_workers: int = os.cpu_count() -1 ,
    chunksize: int = 1           
):
    """
    Parallel pass@k over samples for each problem.

    Returns list[float] (avg pass@k per k in k_list).
    """
    print("MAX workers is : ", max_workers)
    assert len(answers) == len(rtl_code), "Mismatched lengths"
    num_problems = len(answers)
    if num_problems == 0:
        return [0.0 for _ in k_list]

    sums_syntax = [0.0] * len(k_list)
    sums_func = [0.0] * len(k_list)

    for answer, rtl, top in zip(answers, rtl_code, top_module_name):
        per_problem = _compute_one_problem(
            answer=answer,
            rtl_code=rtl,
            top_module_name=top,
            k_list=k_list,
            output_dir=tempfile.mkdtemp(prefix="out_"),
            parallel=True,
            backend=backend,
            max_workers=max_workers,
            chunksize=chunksize
        )
        for i, (v_syntax, v_func) in enumerate(zip(per_problem[0], per_problem[1])):
            sums_syntax[i] += v_syntax
            sums_func[i] += v_func

    avg_pass_syntax  = [s / num_problems for s in sums_syntax]
    avg_pass_func  = [s / num_problems for s in sums_func]

    return avg_pass_syntax, avg_pass_func



def get_max_cov(
    samples,
    rtl_code,
    top_module_name,
    n,
    k_list
):
    
    max_cov_score_list = []
    for i in range(len(k_list)): 
        max_cov_score = {
            'line': 0,
            'cond': 0,
            'toggle': 0,
            'fsm': 0,
            'branch': 0
        }
        max_cov_score_list.append(max_cov_score)

    e_list = {
        'line': [],
        'cond': [],
        'toggle': [],
        'fsm': [],
        'branch': []
    }

    for sample in samples:
        output_dir = tempfile.mkdtemp(prefix="out_cov_")
        try:
            eff_score = compute_coverage_score_normalized(
                pred=sample,
                rtl_code=rtl_code,
                top_module_name=top_module_name,
                output_dir=output_dir
            )
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)
        
        for key in eff_score.keys(): 
            e_list[key].append(eff_score[key])

    for i, k in enumerate(k_list): 
        for key in e_list:
            if e_list[key] is None: 
                max_cov_score_list[i][key] = None 
            else: 
                max_cov_score_list[i][key] = cov_estimator(
                    e=e_list[key],
                    n=n,
                    k=k
                )

    return max_cov_score_list



def _score_one_sample(sample, rtl_code, top_module_name):
    output_dir = tempfile.mkdtemp(prefix="out_cov")
    try:
        return compute_coverage_score_normalized(
            pred=sample,
            rtl_code=rtl_code,
            top_module_name=top_module_name,
            output_dir=output_dir
        )
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
   


def get_max_cov_parallel(
    samples,
    rtl_code,
    top_module_name,
    n,
    k_list,
    *,
    backend: str = "thread",     # "process" for CPU-bound; "thread" for I/O/subprocess-heavy
    max_workers: int = os.cpu_count() - 1,
    chunksize: int = 8            # helpful for ProcessPool on large arrays
):
    max_cov_score_list = [{k: 0 for k in ("line", "cond", "toggle", "fsm", "branch")} for _ in k_list]
    e_list = {k: [] for k in  ("line", "cond", "toggle", "fsm", "branch")}

    print("MAX workers is : ", max_workers)

    # Choose executor
    Executor = ProcessPoolExecutor if backend == "process" else ThreadPoolExecutor
    if max_workers is None:
        if backend == "process":
            max_workers = os.cpu_count() or 1
        else:
            # threads can exceed core count if you mostly wait on simulators/I/O
            max_workers = 4 * (os.cpu_count() or 1)

    worker = partial(
        _score_one_sample,
        rtl_code=rtl_code,
        top_module_name=top_module_name
    )

    with Executor(max_workers=max_workers) as ex:
        if backend == "process":
            results_iter = ex.map(worker, samples, chunksize=chunksize)
        else:
            results_iter = ex.map(worker, samples)

        for eff_score in results_iter:
            if not eff_score:
                continue  # skip failures
            for key, val in eff_score.items():
                e_list[key].append(val)

    for i, k in enumerate(k_list):
        for key, vals in e_list.items():
            if all(v is None for v in vals):
                max_cov_score_list[i][key] = None
            else: 
                effective_vals = [0.0 if v is None else v for v in vals]
                max_cov_score_list[i][key] = cov_estimator(e=effective_vals, n=n, k=k)


    return max_cov_score_list


def compute_cov_at_k(
    answers: list[list],
    rtl_code: list,
    top_module_name: list,
    k_list: int,
    parallel : bool = True,
    max_workers: int = os.cpu_count() - 1   
):
    """
    Compute cov@k value; Coverage + Functional correctness
    """

    avg_cov_at_k_list = []
    for i in range(len(k_list)): 
        avg_cov_at_k = {
            'line': 0,
            'cond': 0,
            'toggle': 0,
            'fsm': 0,
            'branch': 0
        }
        avg_cov_at_k_list.append(avg_cov_at_k)

    num_problems = len(answers)
    for i, answer in enumerate(answers):
        num_samples = len(answer)
       
        if parallel: 
            cov_score_list = get_max_cov_parallel(
                samples=answer,
                rtl_code=rtl_code[i],
                top_module_name=top_module_name[i],
                n=num_samples,
                k_list=k_list,
                backend="thread",                  # <-- important for Yosys/OpenSTA
                max_workers=max_workers     # leave one core free
            )
        else:
            cov_score_list = get_max_cov(
                samples=answer,
                rtl_code=rtl_code[i],
                top_module_name=top_module_name[i],
                n=num_samples,
                k_list=k_list
            )

        for idx in range(len(k_list)): 
            for key in cov_score_list[idx].keys(): 
                if cov_score_list[idx][key] is None: 
                    avg_cov_at_k_list[idx][key] = None 
                else: 
                    avg_cov_at_k_list[idx][key] += cov_score_list[idx][key]

    for idx in range(len(k_list)): 
        for key in avg_cov_at_k_list[idx].keys(): 
            if avg_cov_at_k_list[idx][key] is not None: 
                avg_cov_at_k_list[idx][key] = avg_cov_at_k_list[idx][key] / num_problems
    
    return avg_cov_at_k_list 


def compute_coverage_score_normalized(
    pred,
    rtl_code,
    top_module_name,
    output_dir
):
    
    scores = {
        'line': None, 
        'cond': None, 
        'toggle': None, 
        'fsm': None, 
        'branch': None
    }

    with open(pred, "r") as f:
        testbench_content = f.read()

    correct, _ = run_testbench_vcs(
        testbench=pred,
        top_module_name=top_module_name,
        verilog_files=[rtl_code],
        output_dir=output_dir,
    ) 
 
    if correct: 
        correct, report_file = run_urg(
            top_module_name=top_module_name,
            output_dir=output_dir,
            testbench=pred
        )

        coverage = report_coverage_vcs(
            report_file=report_file
        )
        
        print("Pred Metrics: ", coverage)

        scores['toggle'] = coverage['toggle'] / 100.0 if coverage['toggle'] is not None else None
        scores['fsm'] = coverage['fsm'] / 100.0   if coverage['fsm'] is not None else None
        scores['cond'] = coverage['cond'] / 100.0   if coverage['cond'] is not None else None
        scores['line'] = coverage['line'] / 100.0   if coverage['line'] is not None else None
        scores['branch'] = coverage['branch'] / 100.0  if coverage['branch'] is not None else None

        print("Scores are: ", scores)

    return scores




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', type=str, help="Path to JSON eval set", default="./data/eval_dataset.jsonl")
    parser.add_argument('--output_dir', type=str, help="Path to output directory", default="./output")
    args = parser.parse_args() 

    eval = args.eval
    output_dir = args.output_dir 

    logger = get_logger('metrics.log')
    
    tmp_dir = tempfile.mkdtemp()
    data = read_jsonl(eval)

    for problem in data: 
        if problem["dataset"] == 'verilog_eval':
            continue 

        logger.info(f"CYAN: Running Problem {problem['sample_id']}")

        rtl_code_file = os.path.join(tmp_dir, f"rtl_code.v")
        with open(rtl_code_file, "w") as file:
            file.write(problem["rtl_code"])
        
        testbench_path = os.path.join(tmp_dir, f"testbench.v")
        with open(testbench_path, "w") as file:
            file.write(problem["testbench"])
        
        start_time = time.time()
        pass_at_k_syntax, pass_at_k_func = compute_pass_at_k(
            answers=[[testbench_path, testbench_path, testbench_path, testbench_path, testbench_path, testbench_path, testbench_path]],
            rtl_code=[rtl_code_file],
            top_module_name=[problem["top_module"]],
            k_list=[1, 5, 10]
        )
        total_time = time.time() - start_time 

        logger.info(f"YELLOW: Pass@k-Syntax is {pass_at_k_syntax}, Pass@k-Func {pass_at_k_func} Total Time {total_time}")

        start_time = time.time()
        pass_at_k_syntax, pass_at_k_func  = compute_pass_at_k_parallel(
            answers=[[testbench_path, testbench_path, testbench_path, testbench_path, testbench_path, testbench_path, testbench_path]],
            rtl_code=[rtl_code_file],
            top_module_name=[problem["top_module"]],
            k_list=[1, 5, 10],
            backend="thread",
            max_workers=os.cpu_count() - 1     
        )
        total_time = time.time() - start_time 

        logger.info(f"YELLOW: Pass@k-Syntax is {pass_at_k_syntax}, Pass@k-Func {pass_at_k_func} Total Time {total_time}")

        start_time = time.time()


        cov_at_k = compute_cov_at_k(
            answers=[[testbench_path, testbench_path, testbench_path, testbench_path, testbench_path, testbench_path, testbench_path]],
            rtl_code=[rtl_code_file],
            top_module_name=[problem["top_module"]],
            k_list=[1, 5, 10],
            parallel=False
        )
        total_time = time.time() - start_time 

        logger.info(f"YELLOW: cov@k is {cov_at_k}, Total Time {total_time}")
        logger.info(f" Total Time {total_time}")

        start_time = time.time()
        cov_at_k = compute_cov_at_k(
            answers=[[testbench_path, testbench_path, testbench_path, testbench_path, testbench_path, testbench_path, testbench_path]],
            rtl_code=[rtl_code_file],
            top_module_name=[problem["top_module"]],
            k_list=[1, 5, 10],
            parallel=False
        )
        total_time = time.time() - start_time 

        logger.info(f"YELLOW: cov@k is {cov_at_k}, Total Time (Parallel) {total_time}")
        logger.info(f" Total Time (Parallel) {total_time}")

        
if __name__ == '__main__':
    main()
