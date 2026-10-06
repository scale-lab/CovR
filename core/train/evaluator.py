"""Reports pass@k and cov@k on the evaluation dataset
"""
import re
import os
import sys
import json
import tqdm
import argparse
import tempfile
import time
import torch
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from torch.utils.data import DataLoader
import torch.distributed as dist
from transformers import AutoTokenizer
from huggingface_hub import login
from accelerate import Accelerator

from core.train.model import get_model, HFModel
from core.train.instruct_dataset import InstructDataset
from core.train.metrics import compute_cov_at_k, compute_pass_at_k
from core.utils import get_logger
from core.train.instruct_dataset import HFDatasetWrapper
from core.train.utils import set_seed, write_text_answers, write_input_prompts, write_answers, write_ground_truth, write_json, plot_cov_pass_each_k
from core.train.metrics import compute_pass_at_k_parallel, compute_pass_at_k
from core.data_generation.utils import compute_avg_coverage

# from core.train.utils import write_answers, write_ground_truth, write_text_answers
# os.environ["NCCL_TIMEOUT"] = "1800"
# os.environ["NCCL_DEBUG"] = "INFO"
# os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

def _format_elapsed(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def _trim_before_context_end(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    m = re.search(r"</context>", text, flags=re.IGNORECASE)
    if m:
        return text[m.end():].lstrip()
    return text


def _extract_last_module_block(text: str):
    """Return the last complete Verilog module block in text.

    The fallback extractor must not match prose such as "the FSM module has..."
    as a module declaration. Requiring module/endmodule at the start of a line
    keeps the match aligned with real Verilog statements.
    """
    mod_block = re.compile(
        r"(?ms)^[ \t]*module[ \t]+[A-Za-z_][A-Za-z0-9_$]*\b.*?;.*?"
        r"^[ \t]*endmodule\b[^\n]*",
    )
    matches = list(mod_block.finditer(text))
    if not matches:
        return None
    return matches[-1].group(0).strip()


def _extract_testbrench(answers: list[str],  num_samples: int):
    batched_verilog_modules = []
    total_samples = len(answers)
    for b in range(0, total_samples, num_samples):
        verilog_modules = []
        for i in range(b, min(b + num_samples, total_samples)):
            sample = answers[i]
            verilog = None

            sample = _trim_before_context_end(sample)


            sample = re.sub(
                r'\bendmodule\b',
                r'\nendmodule\n\n',
                sample,
                flags=re.IGNORECASE,
            )
            match = re.search(r"CODE BEGIN\s*(.*?)\s*CODE END", sample, re.DOTALL)
            if match:
                verilog = match.group(1).strip()

            if not verilog:
                fence_matches = re.findall(
                    r"```(?:systemverilog|sv|verilog)\s*\n(.*?)```",
                    sample,
                    flags=re.DOTALL | re.IGNORECASE,
                )
                if fence_matches:
                    verilog = fence_matches[-1].strip()

            if not verilog:
                verilog = _extract_last_module_block(sample)

            if not verilog:
                verilog = sample.strip()

            verilog_modules.append(verilog)
        batched_verilog_modules.append(verilog_modules)

    return batched_verilog_modules


def _compute_target_normalized_avg_coverage(cov_entry, target_percentage=100.0):
    avg_coverage = compute_avg_coverage(cov_entry) or 0.0
    target_fraction = max(float(target_percentage), 1e-9) / 100.0
    return min(avg_coverage / target_fraction, 1.0)


class Evaluator:

    def __init__(self, output_dir, batch_size=8, max_workers=2, problem_workers=1) -> None:
        self.output_dir = output_dir

        self.output_dir = output_dir
        self.results_dir = os.path.join(output_dir, "results")
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.problem_workers = max(1, problem_workers)

        os.makedirs(self.results_dir, exist_ok=True)


    def evaluate_answers(
        self,
        answers,
        problems,
        rtl_code,
        target_percentages,
        top_module,
        num_samples,
        k_list,
        logger,
    ): 
        start_time = time.perf_counter()
        pass_at_k_by_problem = [None] * len(problems)
        cov_at_k_by_problem = [None] * len(problems)

        write_text_answers(
            tmp_dir=self.output_dir,
            problems=problems,
            answers=answers,
            num_samples=num_samples,
        )

        logger.info("YELLOW: Extracting Testbenches")
        batched_verilog_modules = _extract_testbrench(
            answers=answers, num_samples=num_samples
        )
        logger.info(f"CYAN: Extracted {len(batched_verilog_modules)} Verilog")

        zero_cov_at_k = [
            {"line": 0.0, "cond": 0.0, "toggle": 0.0, "fsm": 0.0, "branch": 0.0}
            for _ in k_list
        ]
        problem_indices = []

        for j, modules in enumerate(batched_verilog_modules):
            if any(module is None for module in modules):
                logger.info(
                    f"RED: Failed to extract verilog from {batched_verilog_modules[j]}"
                )
                pass_at_k_by_problem[j] = {
                    "problem": problems[j],
                    "pass@k_syntax": [0.0] * len(k_list),
                    "pass@k_func": [0.0] * len(k_list),
                    "pass@k_avg": [0.0] * len(k_list),
                }
            
                cov_at_k_by_problem[j] = {"problem": problems[j], "cov@k": zero_cov_at_k}
            else:
                problem_indices.append(j)

        problem_workers = min(self.problem_workers, len(problem_indices))
        sample_workers = self.max_workers
        if problem_workers > 1:
            sample_workers = max(1, self.max_workers // problem_workers)
            logger.info(
                f"CYAN: Evaluating {len(problem_indices)} problems with "
                f"{problem_workers} problem workers and {sample_workers} sample workers each"
            )

        def _evaluate_one_problem(j):
            problem_start_time = time.perf_counter()
            logger.info(
                f"YELLOW: Problem {problems[j]}, {j}, {len(batched_verilog_modules)}"
            )
            logger.info(
                f"CYAN: Problem {problems[j]} target_percentage: {target_percentages[j]}"
            )

            testbench_files = write_answers(
                tmp_dir=self.output_dir,
                problem=problems[j],
                answers=batched_verilog_modules[j],
            )

            rtl_code_file_path = write_ground_truth(
                tmp_dir=self.output_dir,
                problem=problems[j],
                ground_truth=[rtl_code[j]],
            )

            pass_at_k_list_syntax, pass_at_k_list_func = compute_pass_at_k_parallel(
                answers=[testbench_files],
                rtl_code=rtl_code_file_path,
                top_module_name=[top_module[j]],
                k_list=k_list,
                max_workers=sample_workers
            )

            cov_at_k_list = compute_cov_at_k(
                answers=[testbench_files],
                rtl_code=rtl_code_file_path,
                top_module_name=[top_module[j]],
                k_list=k_list,
                parallel=True,
                max_workers=sample_workers
            )

            pass_result = {
                "problem": problems[j], 
                "pass@k_syntax": pass_at_k_list_syntax, 
                "pass@k_func": pass_at_k_list_func,
                "pass@k_avg": [(s + f) / 2 for s, f in zip(pass_at_k_list_syntax, pass_at_k_list_func)]
            }

            cov_result = {
                "problem": problems[j], 
                "target_percentage": target_percentages[j],
                "cov@k": cov_at_k_list 
            }
            logger.info(
                f"CYAN: Problem {problems[j]} evaluation time: "
                f"{_format_elapsed(time.perf_counter() - problem_start_time)}"
            )
            return j, pass_result, cov_result

        if problem_workers <= 1:
            for j in problem_indices:
                _, pass_result, cov_result = _evaluate_one_problem(j)
                pass_at_k_by_problem[j] = pass_result
                cov_at_k_by_problem[j] = cov_result
        else:
            with ThreadPoolExecutor(max_workers=problem_workers) as ex:
                futures = [ex.submit(_evaluate_one_problem, j) for j in problem_indices]
                for future in as_completed(futures):
                    j, pass_result, cov_result = future.result()
                    pass_at_k_by_problem[j] = pass_result
                    cov_at_k_by_problem[j] = cov_result
            
        logger.info(
            f"CYAN: evaluate_answers total time: "
            f"{_format_elapsed(time.perf_counter() - start_time)}"
        )
        return pass_at_k_by_problem, cov_at_k_by_problem


    def evaluate_json(self, json_path, k_list, logger):
        start_time = time.perf_counter()
        all_pass_at_k_by_problem = []
        all_cov_at_k_by_problem = []

        with open(json_path, "r", encoding="utf-8") as f:
            
            for _, line in enumerate(f, start=1):
                line = line.strip()
                record = json.loads(line)

                pass_at_k_by_problem, cov_at_k_by_problem = self.evaluate_answers(
                    answers=record["answers"],
                    problems=record["problems"],
                    rtl_code=record["rtl_code"],
                    top_module=record["top_module"],
                    num_samples=record["num_samples"],
                    target_percentages=record["target_percentage"],
                    logger=logger,
                    k_list=k_list
                )
                
                all_pass_at_k_by_problem.extend(pass_at_k_by_problem)
                all_cov_at_k_by_problem.extend(cov_at_k_by_problem)

        logger.info(
            f"CYAN: evaluate_json total time for {json_path}: "
            f"{_format_elapsed(time.perf_counter() - start_time)}"
        )
        return all_pass_at_k_by_problem, all_cov_at_k_by_problem

    def evaluate(
        self,
        dataloader,
        model: HFModel,
        num_samples: int,
        k_list: list,
        temperature: float,
        logger,
        debug: bool = True
    ):
        eval_start_time = time.perf_counter()
        total_generation_time = 0.0
        total_inline_eval_time = 0.0
        results_for_csv = []

        avg_pass_at_k_syntax = [0.0] * len(k_list)
        avg_pass_at_k_func = [0.0] * len(k_list)
        avg_cov_at_k = []
        for _ in k_list:
            avg_cov_at_k.append(
                {"line": 0, "cond": 0, "toggle": 0, "fsm": 0, "branch": 0}
            )

        all_pass_at_k_by_problem = []
        all_cov_at_k_by_problem = []
        total_num_problems_seen = 0

        rank = 0
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
        output_path = os.path.join(self.output_dir, f"answers_rank{rank}.jsonl")

        for batch_idx, batch in enumerate(dataloader):
            batch_start_time = time.perf_counter()
            problems = batch["sample_id"]
            rtl_code = batch["rtl_code"]
            top_module = batch["top_module"]
            target_percentages = batch.get("target_percentage")

            logger.info(f"CYAN: Running Problems {problems}")

            # write_input_prompts(
            #     tmp_dir=self.output_dir,
            #     problems=problems,
            #     prompts=batch["prompt"]
            # )

            generation_start_time = time.perf_counter()
            answers = model.message_batch(
                prompts=batch["prompt"],
                temperature=temperature,
                num_samples=num_samples,
            )
            generation_time = time.perf_counter() - generation_start_time
            total_generation_time += generation_time
            logger.info(
                f"CYAN: Batch {batch_idx} generation time: "
                f"{_format_elapsed(generation_time)} "
                f"({len(problems)} problems, {len(answers)} answers)"
            )

            total_num_problems_seen += len(problems)

            if debug: 
                inline_eval_start_time = time.perf_counter()
                pass_at_k_by_problem, cov_at_k_by_problem = self.evaluate_answers(
                    answers=answers,
                    problems=problems,
                    rtl_code=rtl_code,
                    top_module=top_module,
                    num_samples=num_samples,
                    logger=logger,
                    k_list=k_list,
                    target_percentages=target_percentages
                )
                inline_eval_time = time.perf_counter() - inline_eval_start_time
                total_inline_eval_time += inline_eval_time
                logger.info(
                    f"CYAN: Batch {batch_idx} inline evaluation time: "
                    f"{_format_elapsed(inline_eval_time)}"
                )

                all_pass_at_k_by_problem.extend(pass_at_k_by_problem)
                all_cov_at_k_by_problem.extend(cov_at_k_by_problem)
                        
            # dump answers to json file 
            with open(output_path, "a", encoding="utf-8") as f:
                record = {
                    "problems": problems,
                    "rtl_code": rtl_code,
                    "top_module": top_module,
                    "target_percentage": target_percentages.tolist(),
                    "answers": answers,
                    "num_samples": num_samples
                }
                f.write(json.dumps(record) + "\n")
            logger.info(
                f"CYAN: Batch {batch_idx} total time: "
                f"{_format_elapsed(time.perf_counter() - batch_start_time)}"
            )
            
        result = {
            "pass_at_k_syntax": avg_pass_at_k_syntax,
            "pass_at_k_func": avg_pass_at_k_func,
            "cov_at_k": avg_cov_at_k,
            "pass_at_k_by_problem": all_pass_at_k_by_problem,
            "cov_at_k_by_problem": all_cov_at_k_by_problem,
        }

        logger.info(
            f"CYAN: Generation total time: {_format_elapsed(total_generation_time)}"
        )
        logger.info(
            f"CYAN: Inline evaluation total time: {_format_elapsed(total_inline_eval_time)}"
        )
        logger.info(
            f"CYAN: Evaluator.evaluate total time: "
            f"{_format_elapsed(time.perf_counter() - eval_start_time)}"
        )
        return result


def main():
    main_start_time = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Model name: mistral, llama3.1:8b, mixtral, gpt3.5", required=False, default="llama3.1:8b")
    parser.add_argument("--lora", type=str, help="Path to finetuned lora", default=None)
    parser.add_argument("--eval", type=str, help="Path to evaluation set", default="./data/eval_dataset.jsonl")
    parser.add_argument("--output_dir", type=str, help="Path to output directory", default="./output-eval")
    parser.add_argument("--num_samples", type=int, help="number of samples to generate (n)", default=10)
    parser.add_argument("--batch_size", type=int, help="Batch Size", default=8)
    parser.add_argument("--temperature", type=float, help="Batch Size", default=0.1)
    parser.add_argument("--max_seq_length", type=int, help="Batch Size", default=32000)
    parser.add_argument("--max_new_tokens", type=int, help="Batch Size", default=10000)
    parser.add_argument("--max_workers", type=int, help="Batch Size", default=3)
    parser.add_argument("--problem_workers", type=int, help="Number of problems to evaluate in parallel", default=1)
    parser.add_argument("--run_name", type=str, help="run name", default=None)
    parser.add_argument("--answers", type=str, help="Path to JSON answer", default=None)
    parser.add_argument("--debug", dest="debug", action="store_true", help="Enable debug mode")
    parser.add_argument('--use_sloth', action='store_const', const=True, help="Use sloth models for faster finetuning", default=False)
    parser.add_argument('--multi_gpu', action='store_const', const=True, help="Use Multiple GPUs: fully sharded data parallel mode (You have to launch this with accelerate!)", default=False)
    parser.add_argument("--k", type=int, nargs="+", help="K values", default=[1, 5, 10]) 
    parser.add_argument("--with_reasoning", action="store_true", help="to use reasoning prompt in evaluation")
    parser.add_argument("--use_llm4cov_prompt", action="store_true", help="to use LLM4Cov prompt in evaluation")
    parser.add_argument('--use_vllm', action='store_const', const=True, help="Use VLLM", default=False)
    parser.add_argument("--generate_only", action=argparse.BooleanOptionalAction, default=True, help="Only generate answers JSONL and skip Synopsys/VCS evaluation.")

    args = parser.parse_args()

    model_name = args.model
    eval = args.eval
    lora = args.lora
    max_workers = args.max_workers
    problem_workers = args.problem_workers
    max_seq_length = args.max_seq_length
    max_new_tokens = args.max_new_tokens
    k = args.k
    debug = args.debug
    run_name = args.run_name
    answers = args.answers
    use_sloth = args.use_sloth
    temperature = args.temperature
    multi_gpu = args.multi_gpu
    num_samples = args.num_samples
    batch_size = args.batch_size
    output_dir = args.output_dir
    use_llm4cov_prompt = args.use_llm4cov_prompt
    use_vllm = args.use_vllm
    generate_only = args.generate_only

    set_seed(0, n_gpu=torch.cuda.device_count())
    
    output_dir = os.path.join(output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)

    logger = get_logger(f"eval_{model_name}.log")
    logger.info(f"CYAN: Model, {model_name}")
    logger.info("CYAN: Starting evaluator")

    access_token = os.getenv('HF_ACCESS_TOKEN')
    # login(token=access_token)

    if multi_gpu or generate_only:
        debug = False

    if run_name: 
        base_dir = os.path.join(output_dir, f"{model_name}_{run_name}")
    else:
        base_dir = os.path.join(output_dir, model_name)

    os.makedirs(base_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    evaluator = Evaluator(
        output_dir=output_dir, 
        batch_size=batch_size,
        max_workers=max_workers,
        problem_workers=problem_workers
    )

    if answers: 
        avg_pass_at_k_syntax = [0.0] * len(k)
        avg_pass_at_k_func = [0.0] * len(k)
        avg_cov_at_k = []
       
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return

        accelerator = Accelerator() if multi_gpu else None
        if not accelerator or accelerator.is_main_process: 
            all_pass_at_k_by_problem, all_cov_at_k_by_problem = evaluator.evaluate_json(
                json_path=answers,
                k_list=k,
                logger=logger
            )
            num_problems = len(all_pass_at_k_by_problem)
            avg_pass_at_k_syntax = [
                sum(p["pass@k_syntax"][i] for p in all_pass_at_k_by_problem) / num_problems
                for i in range(len(k))
            ]
            avg_pass_at_k_func = [
                sum(p["pass@k_func"][i] for p in all_pass_at_k_by_problem) / num_problems
                for i in range(len(k))
            ]

            plot_cov_pass_each_k(
                all_pass_at_k_by_problem, 
                all_cov_at_k_by_problem,
                k_list=k,
                save_dir=evaluator.results_dir,
                prefix=f"cov_pass",
                autoscale_ylim=False  
            )

            write_json(
                n=num_samples,
                k_list=k,
                pass_at_k_by_problem=all_pass_at_k_by_problem,
                cov_at_k_by_problem=all_cov_at_k_by_problem,
                save_path=os.path.join(evaluator.results_dir, f"result_eval.json"),
            )

            metrics = ["line", "cond", "toggle", "fsm", "branch"]
            avg_cov_at_k = [
                sum(
                    _compute_target_normalized_avg_coverage(
                        p["cov@k"][i],
                        p.get("target_percentage", 100.0)
                    )
                    for p in all_cov_at_k_by_problem
                ) / num_problems
                for i in range(len(k))
            ]
            
            for i, k_val in enumerate(k):
                logger.info(
                    f"YELLOW: Overall pass@{k_val}_syntax:  {avg_pass_at_k_syntax[i]}, pass@{k_val}_func:  {avg_pass_at_k_func[i]} "
                )
                logger.info(
                    f"YELLOW: Overall cov@{k_val}: {avg_cov_at_k[i]}"
                )

            logger.info("Evaluation completed.")
    else: 
        model_load_start_time = time.perf_counter()
        model = get_model(
            name=model_name, 
            lora_path=lora, 
            merge_lora=True,
            max_seq_length=max_seq_length,
            max_new_tokens=max_new_tokens,
            use_sloth=use_sloth,
            multi_gpu=multi_gpu,
            use_vllm=use_vllm
        )
        logger.info(
            f"CYAN: Model load time: "
            f"{_format_elapsed(time.perf_counter() - model_load_start_time)}"
        )

        dataset_start_time = time.perf_counter()
        if "gpt" not in model_name and "deepseek-v3" not in model_name: 
            tokenizer = AutoTokenizer.from_pretrained(
                model.model_instruct_ids[model_name], 
                token=access_token, 
            ) 
        else:
            tokenizer = None 

        dataset = InstructDataset(
            train_json=None,     
            test_json=eval,
            train_split=0.8,
            valid_split=0.2,
            model=model,
            tokenizer=tokenizer,
            shuffle=True,
            with_reasoning=args.with_reasoning if args.with_reasoning else False,
            use_llm4cov_prompt=use_llm4cov_prompt
        )    
        
        print(dataset.dataset['test'])
        wrapped_dataset = HFDatasetWrapper(dataset.dataset['test'])
        dataloader = DataLoader(wrapped_dataset , batch_size=batch_size, num_workers=0)
        logger.info(
            f"CYAN: Dataset/tokenizer setup time: "
            f"{_format_elapsed(time.perf_counter() - dataset_start_time)}"
        )

        if multi_gpu: 
            dataloader = model.accelerator.prepare(dataloader)
        
        generation_eval_start_time = time.perf_counter()
        result = evaluator.evaluate(
            dataloader=dataloader,
            model=model,
            num_samples=num_samples,
            k_list=k,
            temperature=args.temperature,
            debug=debug,
            logger=logger,
        )
        logger.info(
            f"CYAN: Answer generation loop time: "
            f"{_format_elapsed(time.perf_counter() - generation_eval_start_time)}"
        )

        if multi_gpu:
            model.accelerator.wait_for_everyone()

        if generate_only:
            logger.info(
                f"CYAN: Generation-only mode completed. Answers saved under {output_dir}"
            )
            return

        avg_pass_at_k_syntax = [0.0] * len(k)
        avg_pass_at_k_func = [0.0] * len(k)
        avg_cov_at_k = []
       
        if (not multi_gpu) or model.accelerator.is_main_process:
            saved_answer_eval_start_time = time.perf_counter()
            all_pass_at_k_by_problem = []
            all_cov_at_k_by_problem = []

            world_size = model.accelerator.num_processes if multi_gpu else 1
            for r in range(world_size):
                p = os.path.join(output_dir, f"answers_rank{r}.jsonl")
                if os.path.exists(p):
                    pass_p, cov_p = evaluator.evaluate_json(
                        p,
                        k_list=k,
                        logger=logger
                    )
                    all_pass_at_k_by_problem.extend(pass_p)
                    all_cov_at_k_by_problem.extend(cov_p)
            logger.info(
                f"CYAN: Saved-answer evaluation time: "
                f"{_format_elapsed(time.perf_counter() - saved_answer_eval_start_time)}"
            )

            num_problems = len(all_pass_at_k_by_problem)
            avg_pass_at_k_syntax = [
                sum(p["pass@k_syntax"][i] for p in all_pass_at_k_by_problem) / num_problems
                for i in range(len(k))
            ]
            avg_pass_at_k_func = [
                sum(p["pass@k_func"][i] for p in all_pass_at_k_by_problem) / num_problems
                for i in range(len(k))
            ]

            plot_cov_pass_each_k(
                all_pass_at_k_by_problem, 
                all_cov_at_k_by_problem,
                k_list=k,
                save_dir=evaluator.results_dir,
                prefix=f"cov_pass",
                autoscale_ylim=False  
            )
            
            write_json(
                n=num_samples,
                k_list=k,
                pass_at_k_by_problem=all_pass_at_k_by_problem,
                cov_at_k_by_problem=all_cov_at_k_by_problem,
                save_path=os.path.join(evaluator.results_dir, f"result_eval.json"),
            )

            metrics = ["line", "cond", "toggle", "fsm", "branch"]
            avg_cov_at_k = [
                sum(
                    _compute_target_normalized_avg_coverage(
                        p["cov@k"][i],
                        p.get("target_percentage", 100.0)
                    )
                    for p in all_cov_at_k_by_problem
                ) / num_problems
                for i in range(len(k))
            ]
            
            for i, k_val in enumerate(k):
                logger.info(
                    f"YELLOW: Overall pass@{k_val}_syntax] {avg_pass_at_k_syntax[i]}, pass@{k_val}_func] {avg_pass_at_k_func[i]} "
                )

                logger.info(
                    f"YELLOW: Overall cov@{k_val}] {avg_cov_at_k[i]}"
                )

            logger.info("Evaluation completed.")
   
    logger.info(
        f"CYAN: End-to-end evaluator time: "
        f"{_format_elapsed(time.perf_counter() - main_start_time)}"
    )


if __name__ == "__main__":
    main()
