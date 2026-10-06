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

"""Runs iterative feedback loop without reflector/curator to enhance tb based on coverage report
"""
import os 
import sys
import ast
import json
import tqdm
import time 
import atexit
import argparse
import shutil
import tempfile
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.utils import get_logger
from core.data_generation.runner import DataGenRunner, DataGenNodes
from core.data_generation.config.config import FlowConfigs, LMConfigs
from core.data_generation.top_parser import TopParser
from core.data_generation.utils import read_jsonl
from core.train.metrics import compute_pass_at_k_parallel
from core.train.metrics import compute_cov_at_k, compute_pass_at_k
from core.data_generation.utils import compute_avg_coverage
from core.train.utils import write_text_answers, write_answers, write_ground_truth, write_json, plot_cov_pass_each_k
from core.train.evaluator import Evaluator
from core.metrics.utils import get_top_module 


def clone_flow_config(config, output_dir):
    return FlowConfigs(
        lm_config=config.lm_config,
        max_refine_iters=config.max_refine_iters,
        max_enhance_iters=config.max_enhance_iters,
        graph_recursion_limit=config.graph_recursion_limit,
        output_dir=output_dir,
        testbench_style=config.testbench_style,
        wo_reflector=config.wo_reflector,
        wo_curator=config.wo_curator,
        data_gen_mode=config.data_gen_mode,
    )


def pick_best_testbench(output):
    testbench = output.get("testbench")
    testbenches = output.get("testbenches", [])
    coverage_stats_list = output.get("coverage_stats_list", [])

    if not testbenches or not coverage_stats_list:
        return testbench, output.get("coverage_stats"), -1

    n = min(len(testbenches), len(coverage_stats_list))
    if n == 0:
        return testbench, output.get("coverage_stats"), -1

    best_idx = max(
        range(n),
        key=lambda i: compute_avg_coverage(coverage_stats_list[i])
    )

    return testbenches[best_idx], coverage_stats_list[best_idx], best_idx

def get_target_percentage(entry):
    if "target_percentage" in entry:
        return entry["target_percentage"]

    dut_instance = entry.get("dut_instance_name", "")
    for target in entry.get("targets", []):
        if target.get("inst_name") == dut_instance:
            return target.get("target_percentage", 100.0)

    return 100.0


def compute_target_normalized_avg_coverage(cov_entry, target_percentage=100.0):
    avg_coverage = compute_avg_coverage(cov_entry) or 0.0
    target_fraction = max(float(target_percentage), 1e-9) / 100.0
    return min(avg_coverage / target_fraction, 1.0)


def main():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--prompter-model', type=str, default='gpt-4o-mini-2024-07-18')
    parser.add_argument('--fixer-model', type=str, default='gpt-4o-mini-2024-07-18')
    parser.add_argument('--tb-model', type=str, default='gpt-4o-mini-2024-07-18')
    parser.add_argument('--enhancer-model', type=str, default='gpt-4o-mini-2024-07-18')
    parser.add_argument('--checker-model', type=str, default='gpt-4o-mini-2024-07-18')
    parser.add_argument('--reflector-model', type=str, default='gpt-4o-mini-2024-07-18')
    parser.add_argument('--curator-model', type=str, default='gpt-4o-mini-2024-07-18')
    parser.add_argument('--dataset', type=str, help='Path to jsonl file containing dataset', default="./data/eval_dataset.jsonl")
    parser.add_argument('--answers', type=str, help='Path to jsonl file containing answers', default=None)
    parser.add_argument('--output', '--output_dir', dest='output', type=str, help='Path to output directory', default='./output-eval/baselines/self_reflection_loop')
    parser.add_argument('--start_idx', type=int, help='Path to output direcotyr', default=0)
    parser.add_argument('--iters', type=int, help='number of self refinement iterations', default=5)
    parser.add_argument('--num_samples', type=int, help='Number of trajectories to sample', default=5)
    parser.add_argument('--sample-workers', type=int, help='Number of trajectories to run concurrently per problem', default=1)
    parser.add_argument('--problem-workers', type=int, help='Number of problems to run concurrently', default=1)
    parser.add_argument('--trajectory-work-root', type=str, help='Root directory for per-trajectory VCS/coverage scratch files', default='/tmp')
    parser.add_argument('--keep-trajectory-work-dir', action='store_true', help='Keep per-trajectory VCS/coverage scratch files for debugging')
    parser.add_argument('--generate-only', action=argparse.BooleanOptionalAction, default=False, help='Only generate answers JSONL and skip final saved-answer evaluation')
    parser.add_argument('--seed', type=int, help='Sampling seed for reproducible endpoint generation', default=None)
    parser.add_argument("--k", type=list, help="K values", default=[1, 5, 10])
    parser.add_argument('--max_seq_length', type=int, help='Path to output direcotyr', default=32000)
    parser.add_argument('--max_new_tokens', type=int, help='Path to output direcotyr', default=10000)
    parser.add_argument('--wo_reflector', '--wo-reflector', action='store_true', help='Disable reflector module')
    parser.add_argument('--wo_curator', '--wo-curator', action='store_true', help='Disable curator module')
    parser.add_argument("--lora", "--generator-lora", dest="lora", type=str, help="Path to finetuned LoRA adapter for the testbench generator", default=None)
    parser.add_argument("--api-base", "--agent-api-base", dest="api_base", type=str, default=None, help="OpenAI-compatible API base URL for non-generator agents, e.g. http://localhost:8000/v1")
    parser.add_argument("--api-key", "--agent-api-key", dest="api_key", type=str, default="EMPTY", help="API key for the non-generator OpenAI-compatible endpoint")
    parser.add_argument("--generator-api-base", type=str, default=None, help="OpenAI-compatible API base URL for the testbench generator, e.g. http://localhost:8001/v1")
    parser.add_argument("--generator-api-key", type=str, default=None, help="API key for the generator OpenAI-compatible endpoint")
    parser.add_argument("--enhancer-api-base", type=str, default=None, help="OpenAI-compatible API base URL for the enhancer, e.g. http://localhost:8001/v1")
    parser.add_argument("--enhancer-api-key", type=str, default=None, help="API key for the enhancer OpenAI-compatible endpoint")

    args = parser.parse_args()

    answers = args.answers
    dataset = args.dataset 
    output_dir = args.output 
    lora = args.lora
    k_list = args.k
    iters = args.iters
    start_idx = args.start_idx 
    max_seq_length = args.max_seq_length
    max_new_tokens = args.max_new_tokens
    num_samples = args.num_samples
    sample_workers = args.sample_workers
    problem_workers = args.problem_workers
    trajectory_work_root = args.trajectory_work_root
    keep_trajectory_work_dir = args.keep_trajectory_work_dir
    generate_only = args.generate_only
    base_seed = args.seed

    if num_samples < 1:
        raise ValueError("--num_samples must be at least 1")

    if sample_workers < 1:
        raise ValueError("--sample-workers must be at least 1")

    if problem_workers < 1:
        raise ValueError("--problem-workers must be at least 1")

    if lora and args.generator_api_base:
        raise ValueError(
            "Use either --generator-lora for in-process local loading or "
            "--generator-api-base for a remote/local vLLM server, not both. "
            "For two-GPU server mode, load the LoRA with vllm serve and set "
            "--tb-model to the served LoRA model name."
        )

    if lora and not os.path.isdir(lora):
        raise FileNotFoundError(f"LoRA adapter directory does not exist: {lora}")
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_dir, f"{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    logger = get_logger(f"eval_{args.enhancer_model.replace('/', '_')}.log")

    config = FlowConfigs(
        lm_config=LMConfigs(
            prompter_lm=args.prompter_model,
            fixer_lm=args.fixer_model,
            tb_generator_lm=args.tb_model,
            enhancer_lm=args.enhancer_model,
            checker_lm=args.checker_model,
            reflector_lm=args.reflector_model,
            curator_lm=args.curator_model,
            max_seq_length=args.max_seq_length,
            max_new_tokens=args.max_new_tokens,
            temperature=0.1,
            use_vllm=True,
            generator_lora=lora,
            api_base=args.api_base,
            api_key=args.api_key,
            generator_api_base=args.generator_api_base,
            generator_api_key=args.generator_api_key,
            enhancer_api_base=args.enhancer_api_base,
            enhancer_api_key=args.enhancer_api_key,
            seed=base_seed,
        ),
        max_refine_iters=iters,
        max_enhance_iters=iters,
        graph_recursion_limit=50,
        data_gen_mode=False,
        wo_reflector=args.wo_reflector,
        wo_curator=args.wo_curator
    )
    trajectory_work_dir = tempfile.mkdtemp(
        prefix=f"llm4tb_self_reflection_{timestamp}_",
        dir=trajectory_work_root,
    )
    os.makedirs(trajectory_work_dir, exist_ok=True)
    print("Trajectory work dir: ", trajectory_work_dir)
    trajectory_work_dir_cleaned = False

    def cleanup_trajectory_work_dir():
        nonlocal trajectory_work_dir_cleaned
        if trajectory_work_dir_cleaned:
            return
        trajectory_work_dir_cleaned = True

        if keep_trajectory_work_dir:
            print("Keeping trajectory work dir: ", trajectory_work_dir)
            return

        shutil.rmtree(trajectory_work_dir, ignore_errors=True)
        print("Cleaned trajectory work dir: ", trajectory_work_dir)

    atexit.register(cleanup_trajectory_work_dir)

    data = read_jsonl(dataset)
    target_percentage_by_problem = {
        entry["sample_id"]: get_target_percentage(entry)
        for entry in data
    }

    evaluator = Evaluator(
        output_dir=output_dir, 
        batch_size=2
    )

    if not answers: 
        output_path = os.path.join(output_dir, "answers_llm4dv.jsonl")

        def write_record(record):
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

        def run_problem(i, entry):

            problem = entry["sample_id"]
            rtl_code = entry["rtl_code"]
            top_module =  entry.get("top_module", "") 
            target_percentage = get_target_percentage(entry)

            if not top_module:
                top_module_parser = TopParser(
                    config=config
                )
                top_module = top_module_parser.forward({
                    "rtl_code": rtl_code
                })

            safe_problem = "".join(
                c if c.isalnum() or c in ("-", "_") else "_"
                for c in str(problem)
            )

            def make_trajectory_output_dir(sample_idx):
                return tempfile.mkdtemp(
                    prefix=f"{safe_problem}_sample_{sample_idx}_",
                    dir=trajectory_work_dir,
                )

            def make_worker_runner(worker_idx):
                trajectory_config = clone_flow_config(
                    config,
                    output_dir=tempfile.mkdtemp(
                        prefix=f"{safe_problem}_worker_{worker_idx}_",
                        dir=trajectory_work_dir,
                    ),
                )
                return DataGenRunner(config=trajectory_config)

            def run_trajectory(sample_idx, trajectory_runner):
                trajectory_runner.config.output_dir = make_trajectory_output_dir(sample_idx)
                output = trajectory_runner.forward({
                    "rtl_code": rtl_code,
                    "natural_spec": entry["natural_spec"],
                    "language": "Verilog",
                    "top_module": top_module,
                    "testbench_style": "stimulus_only"
                })
                testbench, coverage_stats, best_idx = pick_best_testbench(output)
                return {
                    "sample_idx": sample_idx,
                    "testbench": testbench,
                    "coverage_stats": coverage_stats,
                    "best_idx": best_idx,
                    "enhance_iters": output.get("enhance_iters", 0),
                    "fix_iters": output.get("fix_iters", 0),
                    "coverage_stats_candidates": output.get("coverage_stats_list", []),
                }

            start_time = time.time()

            if sample_workers == 1:
                trajectory_runner = make_worker_runner(0)
                trajectory_results = [
                    run_trajectory(sample_idx, trajectory_runner)
                    for sample_idx in range(num_samples)
                ]
            else:
                trajectory_results = []
                workers = min(sample_workers, num_samples)
                worker_sample_indices = [
                    list(range(worker_idx, num_samples, workers))
                    for worker_idx in range(workers)
                ]

                def run_worker(worker_idx, sample_indices):
                    trajectory_runner = make_worker_runner(worker_idx)
                    return [
                        run_trajectory(sample_idx, trajectory_runner)
                        for sample_idx in sample_indices
                    ]

                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = [
                        executor.submit(run_worker, worker_idx, sample_indices)
                        for worker_idx, sample_indices in enumerate(worker_sample_indices)
                    ]
                    for future in as_completed(futures):
                        trajectory_results.extend(future.result())

                trajectory_results.sort(key=lambda result: result["sample_idx"])

            end_time = time.time()
            total_time = end_time - start_time

            print("Problem: ", problem, "Total Time: ", total_time)

            answers_list = [result["testbench"] for result in trajectory_results]
            coverage_stats_list = [result["coverage_stats"] for result in trajectory_results]
            best_idx_list = [result["best_idx"] for result in trajectory_results]
            enhance_iters_list = [result["enhance_iters"] for result in trajectory_results]
            fix_iters_list = [result["fix_iters"] for result in trajectory_results]
            coverage_stats_candidates_list = [
                result["coverage_stats_candidates"] for result in trajectory_results
            ]

            scored_indices = [
                idx for idx, stats in enumerate(coverage_stats_list)
                if stats is not None
            ]
            if scored_indices:
                best_trajectory_idx = max(
                    scored_indices,
                    key=lambda idx: compute_avg_coverage(coverage_stats_list[idx])
                )
                best_coverage_stats = coverage_stats_list[best_trajectory_idx]
            else:
                best_trajectory_idx = -1
                best_coverage_stats = None

            print("[Final] Picking trajectory at idx: ", best_trajectory_idx)

            return i, {
                "problems": [problem],
                "rtl_code": [rtl_code],
                "top_module": [top_module],
                "target_percentage": [target_percentage],
                "answers": answers_list,
                "num_samples": len(answers_list),
                "total_time": total_time,
                "enhance_iters": max(enhance_iters_list) if enhance_iters_list else 0,
                "fix_iters": max(fix_iters_list) if fix_iters_list else 0,
                "enhance_iters_list": enhance_iters_list,
                "fix_iters_list": fix_iters_list,
                "coverage_stats": best_coverage_stats,
                "best_idx": best_trajectory_idx,
                "trajectory_best_idx_list": best_idx_list,
                "coverage_stats_list": coverage_stats_list,
                "coverage_stats_candidates_list": coverage_stats_candidates_list,
            }

        entries_to_run = [
            (i, entry)
            for i, entry in enumerate(data)
            if i >= start_idx
        ]

        if not entries_to_run:
            print("No problems to run after applying start_idx.")
        elif problem_workers == 1:
            for i, entry in tqdm.tqdm(entries_to_run):
                _, record = run_problem(i, entry)
                write_record(record)
        else:
            workers = min(problem_workers, len(entries_to_run))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(run_problem, i, entry)
                    for i, entry in entries_to_run
                ]
                for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
                    _, record = future.result()
                    write_record(record)

    else:
        output_path = answers 
        output_with_targets_path = os.path.join(output_dir, "answers_with_target_percentage.jsonl")
        with open(output_path, "r", encoding="utf-8") as src, open(output_with_targets_path, "w", encoding="utf-8") as dst:
            for line in src:
                if not line.strip():
                    continue
                record = json.loads(line)
                if "target_percentage" not in record:
                    record["target_percentage"] = [
                        target_percentage_by_problem.get(problem, 100.0)
                        for problem in record.get("problems", [])
                    ]
                elif not isinstance(record["target_percentage"], list):
                    record["target_percentage"] = [record["target_percentage"]]
                dst.write(json.dumps(record) + "\n")
        output_path = output_with_targets_path

    if generate_only:
        logger.info(
            f"CYAN: Generation-only mode completed. Answers saved to {output_path}"
        )
        cleanup_trajectory_work_dir()
        atexit.unregister(cleanup_trajectory_work_dir)
        return

    all_pass_at_k_by_problem, all_cov_at_k_by_problem  = evaluator.evaluate_json(
        json_path=output_path,
        k_list=k_list,
        logger=logger
    )

    plot_cov_pass_each_k(
        all_pass_at_k_by_problem, 
        all_cov_at_k_by_problem,
        k_list=k_list,
        save_dir=evaluator.results_dir,
        prefix=f"cov_pass",
        autoscale_ylim=False  # lets values exceed 1 if they do
    )
    
    write_json(
        n=num_samples,
        k_list=k_list,
        pass_at_k_by_problem=all_pass_at_k_by_problem,
        cov_at_k_by_problem=all_cov_at_k_by_problem,
        save_path=os.path.join(evaluator.results_dir, f"result_eval.json"),
    )

    num_problems = len(all_pass_at_k_by_problem)

    avg_pass_at_k_syntax = [
        sum(p["pass@k_syntax"][i] for p in all_pass_at_k_by_problem) / num_problems
        for i in range(len(k_list))
    ]
    avg_pass_at_k_func = [
        sum(p["pass@k_func"][i] for p in all_pass_at_k_by_problem) / num_problems
        for i in range(len(k_list))
    ]

    metrics = ["line", "cond", "toggle", "fsm", "branch"]
    avg_cov_at_k = [
        sum(
            compute_target_normalized_avg_coverage(
                p["cov@k"][i],
                p.get("target_percentage", 100.0)
            )
            for p in all_cov_at_k_by_problem
        ) / num_problems
        for i in range(len(k_list))
    ]

    for i, k_val in enumerate(k_list):
        logger.info(
            f"YELLOW: Overall pass@{k_val}_syntax] {avg_pass_at_k_syntax[i]}, pass@{k_val}_func] {avg_pass_at_k_func[i]} "
        )

        logger.info(
            f"YELLOW: Overall cov@{k_val}] {avg_cov_at_k[i]}"
        )

        logger.info("Evaluation completed.")

    cleanup_trajectory_work_dir()
    atexit.unregister(cleanup_trajectory_work_dir)

  

if __name__ == '__main__':
    main()
