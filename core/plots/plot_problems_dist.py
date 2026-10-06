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

import os
import json
import argparse
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")

from core.train.utils import plot_cov_pass_each_k

AXIS_LABEL_FONTSIZE = 11
TICK_LABEL_FONTSIZE = 9
LEGEND_FONTSIZE = 10


def read_json(save_path: str, k_list: List[int], problem_prefix: str = ""):

    with open(save_path, "r") as f:
        combined = json.load(f)

    pass_at_k_by_problem = []
    cov_at_k_by_problem = []

    for prob_name, entry in combined.items():
        # skip the aggregated stats
        if prob_name == "overall":
            continue

        plot_prob_name = f"{problem_prefix}{prob_name}" if problem_prefix else prob_name
        pass_syntax_list = []
        pass_func_list = []
        cov_list = []

        for k in k_list:
            pass_syntax_list.append(entry.get(f"pass@{k}_syntax"))
            pass_func_list.append(entry.get(f"pass@{k}_func"))
            cov_list.append(entry.get(f"cov@{k}"))

        pass_at_k_by_problem.append({
            "problem": plot_prob_name,
            "pass@k_syntax": pass_syntax_list,
            "pass@k_func": pass_func_list,
        })

        cov_at_k_by_problem.append({
            "problem": plot_prob_name,
            "cov@k": cov_list,
        })

    return pass_at_k_by_problem, cov_at_k_by_problem


def add_avg_pass(pass_at_k_by_problem):
    for d in pass_at_k_by_problem:
        syntax_vals = d.get("pass@k_syntax", [])
        func_vals = d.get("pass@k_func", [])
        avg_vals = []
        for syntax_val, func_val in zip(syntax_vals, func_vals):
            if syntax_val is None or func_val is None:
                avg_vals.append(None)
            else:
                avg_vals.append((syntax_val + func_val) / 2.0)
        d["pass@k_avg"] = avg_vals


def plot_result_json(json_path: str, k_list: List[int], save_dir: str, prefix: str, problem_prefix: str = ""):
    pass_at_k_by_problem, cov_at_k_by_problem = read_json(
        json_path,
        k_list=k_list,
        problem_prefix=problem_prefix,
    )
    add_avg_pass(pass_at_k_by_problem)

    return plot_cov_pass_each_k(
        pass_at_k_by_problem,
        cov_at_k_by_problem,
        k_list=k_list,
        autoscale_ylim=False,
        save_dir=save_dir,
        prefix=prefix,
        cov_metric="avg",
        axis_label_fontsize=AXIS_LABEL_FONTSIZE,
        tick_label_fontsize=TICK_LABEL_FONTSIZE,
        legend_fontsize=LEGEND_FONTSIZE,
    )


def safe_name(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json",
        type=str,
        action="append",
        default=[],
        help="Optional extra result_eval.json path to plot separately. Can be passed multiple times.",
    )
    parser.add_argument("--k", nargs="+", type=int, help="K values", default=[1, 5, 10])
    parser.add_argument('--output_dir', type=str, help="Path to output directory ", required=False, default="./output/plot")
    
    args = parser.parse_args()

    k = args.k 
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    qwen_coder_verilog_eval_problems = {
        "vanilla": "/oscar/data/sreda/mabdelat/LLM4TB/output-eval/base_models/vllm/qwen_coder_7b_veval/qwen2.5-coder/qwen2.5-coder_qwen_coder_7b_veval/20260422_204906/results/result_eval.json",
        "sft": "/oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen2.5-coder_full_sft_r512_a512_full_data_without_reasoning/qwen2.5-coder_full_sft_r512_a512_full_data_without_reasoning_20260507_172807/test_veval/synopsys/qwen2.5-coder/qwen2.5-coder_qwen2.5-coder_eval_synopsys/20260508_112139/results/result_eval.json",
        "sft_reason": "/oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen2.5-coder_full_sft_r512_a512_completion_only_loss_full_data/qwen2.5-coder/qwen2.5-coder_test_veval_r512_comp_only_loss/20260505_140045/synopsys/qwen2.5-coder/qwen2.5-coder_test_veval_r512_comp_only_synopsys/20260505_154936/results/result_eval.json",
        "sft_reason_rl": "/oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen_coder_grpo_unsloth_grpo/checkpoint-600-test_veval/qwen2.5-coder/qwen2.5-coder_test_eval_ckp_600/20260511_154513/synopsys/qwen2.5-coder/qwen2.5-coder_test_eval_ckp_600_synopsys/20260511_202043/results/result_eval.json",
        "sft_reason_rl_agentic": "/oscar/data/sreda/mabdelat/LLM4TB/output-eval/baselines/self_reflection_loop/vllm/qwen_2.5_coder_7b_lora_veval/20260517_142642/results/result_eval.json"
    }
    qwen_coder_cvdp_problems = {
        "vanilla": "/oscar/data/sreda/mabdelat/LLM4TB/output-eval/base_models/vllm/qwen_coder_7b_cvdp/qwen2.5-coder/qwen2.5-coder_qwen_coder_7b_cvdp/20260423_090158/results/result_eval.json",
        "sft": "/oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen2.5-coder_full_sft_r512_a512_full_data_without_reasoning/qwen2.5-coder_full_sft_r512_a512_full_data_without_reasoning_20260507_172807/test_cvdp/synopsys/qwen2.5-coder/qwen2.5-coder_qwen2.5-coder_eval_synopsys/20260508_111830/results/result_eval.json",
        "sft_reason": "/oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen2.5-coder_full_sft_r512_a512_completion_only_loss_full_data/qwen2.5-coder/qwen2.5-coder_test_cvdp_r512_comp_only_loss/20260505_140105/synopsys/qwen2.5-coder/qwen2.5-coder_test_cvdp_r512_comp_only_synopsys/20260505_154908/results/result_eval.json",
        "sft_reason_rl": "/oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen_coder_grpo_unsloth_grpo/checkpoint-600/qwen2.5-coder/qwen2.5-coder_grpo_qwen2.5_coder_cvdp_target_cov/20260527_110020/synopsys/qwen2.5-coder/qwen2.5-coder_grpo_qwen2.5_coder_cvdp_target_cov_synopsys/20260527_123713/results/result_eval.json",
        "sft_reason_rl_agentic": "/oscar/data/sreda/mabdelat/LLM4TB/output-eval/baselines/self_reflection_loop/vllm/qwen_2.5_coder_7b_lora_cvdp/20260527_082301/synopsys/tb_lora/tb_lora_tb_lora_eval_synopsys/20260527_105845/results/result_eval.json"
    }   

    qwen_4b_verilog_eval_problems = {
        "vanilla": "/oscar/data/sreda/mabdelat/LLM4TB/output-eval/base_models/vllm/qwen_coder_4b_veval/qwen4b-instruct/qwen4b-instruct_qwen_coder_4b/20260422_201325/results/result_eval.json",
        "sft": "/oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen4b_instruct_full_sft_r512_a512_full_data_without_reasoning/qwen4b_instruct_full_sft_r512_a512_full_data_without_reasoning_20260507_173003/test_veval/synopsys/qwen4b_instruct/qwen4b_instruct_test_veval_qwen4b_without_reasoning_synopsys/20260508_151446/results/result_eval.json",
        "sft_reason": "/oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen4b_instruct_full_sft_r512_a512_full_data/qwen4b_instruct_full_sft_r512_a512_full_data_20260507_173012/test_veval/qwen4b-instruct/qwen4b-instruct_qwen4b-instruct_eval/20260508_141107/synopsys/qwen4b-instruct/qwen4b-instruct_qwen4b-instruct_eval_synopsys/20260508_191455/results/result_eval.json",
        "sft_reason_rl": "/oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen3_4b_grpo_unsloth_1000_dataset_grpo/checkpoint-500/qwen4b-instruct/qwen4b-instruct_test_veval_ckp_500/20260515_155657/synopsys/qwen4b-instruct/qwen4b-instruct_test_veval_ckp_500_synopsys/20260515_192153/results/result_eval.json",
        "sft_reason_rl_agentic": "/oscar/data/sreda/mabdelat/LLM4TB/output-eval/baselines/self_reflection_loop/vllm/qwen4b-instruct_lora_veval/20260517_142644/results/result_eval.json"
    }
    qwen_4b_cvdp_problems = {
        "vanilla": "/oscar/data/sreda/mabdelat/LLM4TB/output-eval/base_models/vllm/qwen_coder_4b_cvdp/qwen4b-instruct/qwen4b-instruct_qwen_coder_4b_cvdp/20260423_215508/results/result_eval.json",
        "sft": "/oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen4b_instruct_full_sft_r512_a512_full_data_without_reasoning/qwen4b_instruct_full_sft_r512_a512_full_data_without_reasoning_20260507_173003/test_cvdp/synopsys/qwen4b_instruct/qwen4b_instruct_qwen4b_instruct_eval_synopsys/20260508_124246/results/result_eval.json",
        "sft_reason": "/oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen4b_instruct_full_sft_r512_a512_full_data/qwen4b_instruct_full_sft_r512_a512_full_data_20260507_173012/test_cvdp2/qwen4b-instruct/qwen4b-instruct_qwen4b-instruct_eval/20260509_133502/synopsys/qwen4b-instruct/qwen4b-instruct_qwen4b-instruct_eval_synopsys/20260509_155835/results/result_eval.json",
        "sft_reason_rl": "/oscar/data/sreda/mabdelat/LLM4TB/output/sft/qwen3_4b_grpo_unsloth_1000_dataset_grpo/checkpoint-500/qwen4b-instruct/qwen4b-instruct_grpo_qwen4b_instruct_cvdp_target_cov/20260526_223318/synopsys/qwen4b-instruct/qwen4b-instruct_grpo_qwen4b_instruct_cvdp_target_cov_synopsys/20260527_024857/results/result_eval.json",
        "sft_reason_rl_agentic": "/oscar/data/sreda/mabdelat/LLM4TB/output-eval/baselines/self_reflection_loop/vllm/qwen4b-instruct_lora_cvdp/20260527_082402/synopsys/qwen4b-tb-lora/qwen4b-tb-lora_qwen4b-tb-lora_eval_synopsys/20260527_113140/results/result_eval.json"
    }   

    result_groups = {
        "qwen_coder": {
            "verilogeval": qwen_coder_verilog_eval_problems,
            "cvdp": qwen_coder_cvdp_problems,
        },
        "qwen_4b": {
            "verilogeval": qwen_4b_verilog_eval_problems,
            "cvdp": qwen_4b_cvdp_problems,
        },
    }

    for model_name, dataset_results in result_groups.items():
        for dataset_name, config_paths in dataset_results.items():
            for config_name, json_path in config_paths.items():
                if not json_path:
                    print(f"Skipping empty path for {model_name}/{dataset_name}/{config_name}")
                    continue
                if not os.path.exists(json_path):
                    print(f"Skipping missing file for {model_name}/{dataset_name}/{config_name}: {json_path}")
                    continue

                save_dir = os.path.join(
                    output_dir,
                    safe_name(model_name),
                    safe_name(dataset_name),
                    safe_name(config_name),
                )
                prefix = f"cov_pass_{safe_name(model_name)}_{safe_name(dataset_name)}_{safe_name(config_name)}"
                plot_result_json(json_path, k, save_dir, prefix)

        verilog_eval_configs = dataset_results["verilogeval"]
        cvdp_configs = dataset_results["cvdp"]
        for config_name in sorted(set(verilog_eval_configs) & set(cvdp_configs)):
            verilog_eval_path = verilog_eval_configs[config_name]
            cvdp_path = cvdp_configs[config_name]
            if not verilog_eval_path or not cvdp_path:
                print(f"Skipping combined plot for {model_name}/{config_name}: empty path")
                continue
            if not os.path.exists(verilog_eval_path) or not os.path.exists(cvdp_path):
                print(f"Skipping combined plot for {model_name}/{config_name}: missing file")
                continue

            veval_pass, veval_cov = read_json(
                verilog_eval_path,
                k_list=k,
                problem_prefix="verilogeval/",
            )
            cvdp_pass, cvdp_cov = read_json(
                cvdp_path,
                k_list=k,
                problem_prefix="cvdp/",
            )
            combined_pass = veval_pass + cvdp_pass
            combined_cov = veval_cov + cvdp_cov
            add_avg_pass(combined_pass)

            save_dir = os.path.join(
                output_dir,
                safe_name(model_name),
                "verilogeval_cvdp_combined",
                safe_name(config_name),
            )
            prefix = f"cov_pass_{safe_name(model_name)}_verilogeval_cvdp_{safe_name(config_name)}"
            plot_cov_pass_each_k(
                combined_pass,
                combined_cov,
                k_list=k,
                autoscale_ylim=False,
                save_dir=save_dir,
                prefix=prefix,
                cov_metric="avg",
                axis_label_fontsize=AXIS_LABEL_FONTSIZE,
                tick_label_fontsize=TICK_LABEL_FONTSIZE,
                legend_fontsize=LEGEND_FONTSIZE,
            )

    for extra_json in args.json:
        if not os.path.exists(extra_json):
            print(f"Skipping missing extra JSON: {extra_json}")
            continue
        extra_name = safe_name(Path(extra_json).parent.parent.name or Path(extra_json).stem)
        save_dir = os.path.join(output_dir, "extra_json", extra_name)
        plot_result_json(
            extra_json,
            k,
            save_dir,
            prefix=f"cov_pass_{extra_name}",
        )


if __name__ == "__main__":
    main()
