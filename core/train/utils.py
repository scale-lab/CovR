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

import re 
import os
import json
import random
import glob
import shutil
import torch
import numpy as np
import matplotlib.pyplot as plt

from core.data_generation.utils import compute_avg_coverage

def plot_data_lengths(tokenized_train_dataset, tokenized_val_dataset, save_path):
    lengths = [len(x['input_ids']) for x in tokenized_train_dataset]
    # lengths += [len(x['input_ids']) for x in tokenized_val_dataset]

    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=50, alpha=0.7, color='blue')
    plt.xlabel('Length of input_ids')
    plt.ylabel('Frequency')
    plt.title('Distribution of Lengths of input_ids')
    # plt.xlim([0, 1500])
    plt.savefig(save_path)
    plt.cla()
    plt.close()

def plot_loss(log_history, save_path):
    train_loss = []
    valid_loss = []
    for element in log_history:
        if "loss" in element.keys():
            loss = element["loss"]
            train_loss.append(loss)
        elif "eval_loss" in element.keys():
            loss = element["eval_loss"]
            valid_loss.append(loss)
            
    plt.plot(np.arange(0, len(train_loss)), train_loss, marker = 'o', label='Train Loss')
    plt.plot(np.arange(0, len(valid_loss)), valid_loss, marker = 'o', label='Valid Loss')
    plt.legend()
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.savefig(save_path) 
    plt.cla()
    plt.close()


def print_trainable_parameters(model, logger):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logger.info(f"CYAN: trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")


def set_seed(seed, n_gpu=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def write_text_answers(tmp_dir, problems: list[str], answers: list[str], num_samples: int) -> list[str]:
    total_samples = len(answers)
    for i, sample in enumerate(answers):
        problem_idx = i // num_samples
        file_path = os.path.join(tmp_dir, f"{problems[problem_idx]}_ans_txt__{i}.txt")
        with open(file_path, "w") as file:
            file.write(sample)

def write_input_prompts(tmp_dir, problems: list[str], prompts: list[str]) -> list[str]:
    prompt_files = []
    for i, sample in enumerate(prompts):
        file_path = os.path.join(tmp_dir, f"{problems[i]}_prompt_txt__{i}.txt")
        prompt_files.append(os.path.abspath(file_path))
        with open(file_path, "w") as file:
            file.write(sample)

    return prompt_files


def write_answers(tmp_dir, problem: str, answers: list[str]) -> list[str]:
    answer_list = []
    for j, sample in enumerate(answers): 
        file_path = os.path.join(tmp_dir, f"{problem}_ans__{j}.v")
        answer_list.append(os.path.abspath(file_path))
        with open(file_path, "w") as file:
            if sample is None:
                sample = ""
            file.write(sample)

    return answer_list

def write_ground_truth(tmp_dir, problem, ground_truth: list[str]) -> list[str]:
    ground_truth_files = []
    for i, verilog in enumerate(ground_truth): 
        file_path = os.path.join(tmp_dir, f"rtl_code_{problem}_{i}.v")
        ground_truth_files.append(os.path.abspath(file_path))
        with open(file_path, "w") as file:
            file.write(verilog)
    
    return ground_truth_files



def write_json(n, k_list, pass_at_k_by_problem, cov_at_k_by_problem, save_path):
    combined = {}
    K = len(k_list)

    pass_syntax_sum = [0.0] * K
    pass_func_sum = [0.0] * K
    cov_sum  = [0.0] * K
    counts   = [0]   * K  # per-k denominators

    for problem_pass, problem_cov in zip(pass_at_k_by_problem, cov_at_k_by_problem):
        prob_name = problem_pass.get("problem") or problem_pass.get("problem")

        pass_val_syntax = problem_pass.get("pass@k_syntax")
        pass_val_func = problem_pass.get("pass@k_func")
        cov_val = problem_cov.get("cov@k")
        target_percentage = problem_cov.get("target_percentage", 100.0)
        target_fraction = max(float(target_percentage), 1e-9) / 100.0
        # eff_val  = problem_eff.get("eff@k")

        entry = {}
        entry["target_percentage"] = target_percentage

        for i, k in enumerate(k_list):
            p_syntax_i = pass_val_syntax[i] 
            p_func_i = pass_val_func[i] 
            p_cov_i = cov_val[i]
                     
            entry[f"pass@{k}_syntax"] = p_syntax_i
            entry[f"pass@{k}_func"] = p_func_i
            entry[f"cov@{k}"] = p_cov_i

            if p_syntax_i is not None:
                pass_syntax_sum[i] += p_syntax_i
                pass_func_sum[i] += p_func_i
                avg_cov_i = compute_avg_coverage(p_cov_i)
                cov_sum[i] += min(avg_cov_i / target_fraction, 1.0)

                counts[i]   += 1
            # if e_i is not None:
            #     eff_sum[i]  += e_i

        combined[prob_name] = entry

    # Compute overall using per-k counts
    overall = {}
    for i, k in enumerate(k_list):
        overall[f"pass@{k}_syntax"] = round(pass_syntax_sum[i] / counts[i], 4) 
        overall[f"pass@{k}_func"] = round(pass_func_sum[i] / counts[i], 4) 
        overall[f"cov@{k}"]  = round(cov_sum[i]  / counts[i], 4) 
    
    overall["n"] = n
    combined["overall"] = overall

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(combined, f, indent=4)


def _to_float_or_none(x):
    import numpy as np

    try:
        if isinstance(x, np.ndarray):
            if x.size == 0:
                return None
            v = float(x.item())  # safely extract single value
        elif isinstance(x, (np.generic,)):
            v = float(x)
        else:
            v = float(x)

        if np.isnan(v):  # np.isnan handles np.float64 fine
            return None
        return v
    except Exception:
        return None



def _extract_cov_value(ev_k, metric=None):
    if isinstance(ev_k, dict):
        if metric and metric in ev_k:
            return _to_float_or_none(ev_k[metric])
        nums = [_to_float_or_none(v) for v in ev_k.values()]
        nums = [v for v in nums if v is not None]
        return sum(nums) / len(nums) if nums else None
    return _to_float_or_none(ev_k)


def plot_cov_pass_each_k(
    pass_at_k_by_problem,
    cov_at_k_by_problem,
    k_list,
    save_dir,
    prefix="cov_pass",
    clip_to_unit=False,
    autoscale_ylim=True,
    cov_metric=None,
    pass_key="pass@k_func",
    cov_key="cov@k",
    axis_label_fontsize=9,
    tick_label_fontsize=7,
    legend_fontsize=8,
):
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    # ---- GLOBAL STYLE for ACM 2-column, small subfigures ----
    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": axis_label_fontsize,
        "xtick.labelsize": tick_label_fontsize,
        "ytick.labelsize": tick_label_fontsize,
        "legend.fontsize": legend_fontsize,
        "axes.spines.top": True,
        "axes.spines.right": True,
    })

    print("=== Debug Info ===")
    print("len(k_list):", len(k_list))
    if pass_at_k_by_problem:
        print("Sample pass_at_k_by_problem[0]:", pass_at_k_by_problem[0])
    if cov_at_k_by_problem:
        print("Sample cov_at_k_by_problem[0]:", cov_at_k_by_problem[0])

    os.makedirs(save_dir, exist_ok=True)

    pass_dict = {
        d["problem"]: d.get(pass_key, [])
        for d in pass_at_k_by_problem if "problem" in d
    }
    cov_dict = {
        d["problem"]: d.get(cov_key, [])
        for d in cov_at_k_by_problem if "problem" in d
    }

    def _compute_problem_avg_coverage(cov_entry):
        """
        User's definition:
        - ignore null values within the problem
        - if all coverage dimensions are null, coverage for that problem is 0
        """
        if cov_entry is None:
            return 0.0

        # If cov_metric is requested, extract just that metric
        if cov_metric is not None:
            val = _extract_cov_value(cov_entry, cov_metric)
            return 0.0 if val is None else float(val)

        # Otherwise compute average across available metrics
        if isinstance(cov_entry, dict):
            avg_coverage = 0.0
            count = 0
            for _, value in cov_entry.items():
                if value is not None:
                    avg_coverage += float(value)
                    count += 1
            if count != 0:
                avg_coverage = avg_coverage / count
            else:
                avg_coverage = 0.0
            return float(avg_coverage)

        # Fallback: try normal extractor
        val = _extract_cov_value(cov_entry, cov_metric)
        return 0.0 if val is None else float(val)

    saved_paths = []
    problem_set = set(pass_dict) & set(cov_dict)

    for k_idx, k_val in enumerate(k_list):
        rows = []
        for p in problem_set:
            pv = pass_dict.get(p, [])
            ev = cov_dict.get(p, [])

            if not (isinstance(pv, list) and len(pv) > k_idx):
                continue
            if not (isinstance(ev, list) and len(ev) > k_idx):
                continue

            p_k = _to_float_or_none(pv[k_idx])
            if p_k is None:
                continue

            # Fixed behavior:
            # - compute per-problem avg coverage
            # - if all coverage values are null => 0
            e_k = _compute_problem_avg_coverage(ev[k_idx])

            rows.append((p, float(p_k), float(e_k)))

        base_path = os.path.join(save_dir, f"{prefix}_k{k_val}")

        if not rows:
            fig, ax = plt.subplots(figsize=(2.2, 1.2))
            ax.text(0.5, 0.5, f"No valid data for k={k_val}",
                    ha='center', va='center', fontsize=8)
            ax.axis('off')
            plt.savefig(base_path + ".png", bbox_inches='tight', dpi=300)
            plt.savefig(base_path + ".pdf", bbox_inches='tight')
            plt.close()
            saved_paths.append(base_path + ".png")
            saved_paths.append(base_path + ".pdf")
            continue

        # ---- DATA PREP ----
        rows.sort(key=lambda t: t[2], reverse=True)
        problems  = [r[0] for r in rows]
        pass_vals = np.array([r[1] for r in rows], dtype=float)
        cov_vals  = np.array([r[2] for r in rows], dtype=float)

        if clip_to_unit:
            pass_vals = np.clip(pass_vals, 0.0, 1.0)
            cov_vals  = np.clip(cov_vals, 0.0, 1.0)

        # Fixed bug:
        # This is not "pass", it is the positive gap needed to reach pass
        gap_to_pass = np.maximum(0.0, pass_vals - cov_vals)
        x = np.arange(len(problems))

        # ---- PLOT ----
        fig, ax = plt.subplots(figsize=(5.2, 2.2))
        ax.bar(x, cov_vals, 0.7, label=f'cov@{k_val}', zorder=2)
        ax.bar(
            x,
            gap_to_pass,
            0.7,
            bottom=cov_vals,
            label=f'pass@{k_val}',
            zorder=3,
        )

        ax.set_xticks([])
        ax.set_xlim([-0.5, len(x) - 0.5])

        ax.set_ylabel("Metric", fontsize=axis_label_fontsize)
        ax.set_xlabel(f"Problems sorted by cov@{k_val}", fontsize=axis_label_fontsize)
        ax.tick_params(axis="both", labelsize=tick_label_fontsize)

        if autoscale_ylim:
            ymax = max(
                np.max(cov_vals) if len(cov_vals) else 0.0,
                np.max(cov_vals + gap_to_pass) if len(cov_vals) else 0.0
            )
            ax.set_ylim(0, min(1.02, ymax * 1.05 if ymax > 0 else 1.02))
        else:
            ax.set_ylim(0, 1.02)

        ax.legend(
            loc='upper right',
            frameon=True,
            fontsize=legend_fontsize,
            edgecolor="black",
            fancybox=True,
            framealpha=0.9,
        )

        ax.grid(True, axis='y', linestyle='--', alpha=0.35)

        plt.savefig(base_path + ".png", bbox_inches='tight', dpi=300)
        plt.savefig(base_path + ".pdf", bbox_inches='tight')

        plt.close()
        saved_paths.append(base_path + ".png")
        saved_paths.append(base_path + ".pdf")

    return saved_paths



def cleanup_sim_dir(sim_dir, keep_reports=True):
    if keep_reports:
        for path in glob.glob(os.path.join(sim_dir, "*.simv")):
            os.remove(path)
        for path in glob.glob(os.path.join(sim_dir, "*.simv.daidir")):
            shutil.rmtree(path, ignore_errors=True)
        shutil.rmtree(os.path.join(sim_dir, "csrc"), ignore_errors=True)
        for path in glob.glob(os.path.join(sim_dir, "covdb_*.vdb")):
            shutil.rmtree(path, ignore_errors=True)
        for report_dir in glob.glob(os.path.join(sim_dir, "cov_report_*")):
            shutil.rmtree(os.path.join(report_dir, "css"), ignore_errors=True)
            shutil.rmtree(os.path.join(report_dir, "js"), ignore_errors=True)
    else:
        shutil.rmtree(sim_dir, ignore_errors=True)
