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

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

plt.rcParams.update({
    "font.size": 8.0,
    "axes.titlesize": 9.0,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 6.6,
    "legend.title_fontsize": 7.2,
})

# -----------------------------
# Load JSON files
# -----------------------------
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

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

METRICS = ["line", "toggle", "cond", "branch", "fsm"]
METRIC_COLORS = {
    "line": "#5B7FA6",
    "cond": "#78AFA8",
    "toggle": "#D9BC55",
    "fsm": "#A77A9D",
    "branch": "#D66A62",
}
METRIC_LABELS = {
    "line": "Line",
    "cond": "Condition",
    "toggle": "Toggle",
    "fsm": "FSM",
    "branch": "Branch",
}
CONFIG_LABELS = {
    "vanilla": "Vanilla",
    "sft_reason": "SFT-w/Reason",
    "sft_reason_rl": "SFT-w/Reason+RL",
    "sft_reason_rl_agentic": "SFT-w/Reason+RL+Agentic",
}
BAR_CONFIG_LABELS = {
    "vanilla": "Vanilla",
    "sft_reason": "+SFT/W-Reason",
    "sft_reason_rl": "+RL",
    "sft_reason_rl_agentic": "+Agentic",
}
BAR_COLORS = {
    "Vanilla": "#4E79A7",
    "+SFT/W-Reason": "#59A14F",
    "+RL": "#B07AA1",
    "+Agentic": "#E15759",
}
BAR_METRIC_LABELS = {
    "line": "Line",
    "cond": "Cond.",
    "toggle": "Toggle",
    "fsm": "FSM",
    "branch": "Branch",
    "average": "Avg.",
}


def format_variant_label(label):
    return {
        "SFT-w/Reason": "+SFT-W/Reason",
        "SFT-w/Reason+RL": "+RL",
        "SFT-w/Reason+RL+Agentic": "+Agentic",
    }.get(label, label)


def extract_cov_values(data_dict, cov_key):
    values = {metric: [] for metric in METRICS}

    for problem, entry in data_dict.items():
        if problem == "overall":
            continue

        cov = entry.get(cov_key, {})
        if not isinstance(cov, dict):
            continue

        for metric in METRICS:
            value = cov.get(metric)
            if isinstance(value, (int, float)):
                values[metric].append(value)

    return values


def mean_combined_cov(verilog_eval_path, cvdp_path, cov_key):
    combined = {metric: [] for metric in METRICS}

    for path in [verilog_eval_path, cvdp_path]:
        cov_values = extract_cov_values(load_json(path), cov_key)
        for metric in METRICS:
            combined[metric].extend(cov_values[metric])

    return {
        metric: float(np.mean(combined[metric])) if combined[metric] else 0.0
        for metric in METRICS
    }


def build_summary(verilog_eval_paths, cvdp_paths, cov_key, config_labels=CONFIG_LABELS):
    summary = {}

    for config_name, label in config_labels.items():
        verilog_eval_path = verilog_eval_paths.get(config_name, "")
        cvdp_path = cvdp_paths.get(config_name, "")

        if not verilog_eval_path or not cvdp_path:
            print(f"Skipping {label}: empty VerilogEval or CVDP path")
            continue
        if not os.path.exists(verilog_eval_path) or not os.path.exists(cvdp_path):
            print(f"Skipping {label}: missing VerilogEval or CVDP result file")
            continue

        summary[label] = mean_combined_cov(verilog_eval_path, cvdp_path, cov_key)

    return summary


def plot_bar_summary(summary, model_label, cov_key, output_dir):
    labels = list(summary.keys())
    if not labels:
        print(f"No valid bar data for {model_label}; skipping plot")
        return

    categories = METRICS + ["average"]
    category_labels = [BAR_METRIC_LABELS[m] for m in categories]

    values_by_label = {}
    for label in labels:
        vals = [summary[label][m] for m in METRICS]
        values_by_label[label] = [100.0 * v for v in vals] + [100.0 * float(np.mean(vals))]

    style = {
        "font.family": "DejaVu Sans",
        "font.size": 7.2,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 6.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }

    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(3.45, 1.7))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        x = np.arange(len(categories))
        n = len(labels)
        bar_width = min(0.14, 0.66 / max(n, 1))
        offset_step = bar_width + 0.05
        offsets = (np.arange(n) - (n - 1) / 2) * offset_step

        # Highlight average column
        ax.axvspan(
            len(METRICS) - 0.5,
            len(METRICS) + 0.5,
            color="#F3F4F6",
            zorder=0,
        )
        ax.axvline(
            len(METRICS) - 0.5,
            color="#C9CDD3",
            linewidth=0.55,
            zorder=1,
        )

        handles = []

        for i, label in enumerate(labels):
            bars = ax.bar(
                x + offsets[i],
                values_by_label[label],
                width=bar_width,
                label=label,
                color=BAR_COLORS.get(label, "#777777"),
                edgecolor="#2A2A2A",
                linewidth=0.35,
                zorder=3,
            )

            handles.append(bars[0])

            # Label only average bars
            avg_bar = bars[-1]
            avg_val = values_by_label[label][-1]

            ax.text(
                avg_bar.get_x() + avg_bar.get_width() / 2,
                min(avg_val + 1.8, 98.0),
                f"{avg_val:.0f}",
                ha="center",
                va="bottom",
                fontsize=5.8,
                color="#222222",
                rotation=0,
                zorder=4,
            )

        ax.set_ylabel(f"{cov_key} (%)", labelpad=1.5)
        ax.set_ylim(0, 104)

        ax.set_xticks(x)
        ax.set_xticklabels(category_labels)

        ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
        ax.yaxis.set_major_formatter(
            mticker.PercentFormatter(xmax=100, decimals=0)
        )

        ax.grid(
            axis="both",
            color="#E2E6EA",
            linestyle="--",
            linewidth=0.45,
            zorder=0,
        )
        ax.set_axisbelow(True)

        for spine in ["left", "bottom", "top", "right"]:
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_color("#AEB4BB")
            ax.spines[spine].set_linewidth(0.5)

        ax.tick_params(
            axis="both",
            colors="#222222",
            length=2.0,
            width=0.5,
            pad=1.5,
        )

        legend = ax.legend(
            handles,
            labels,
            ncol=min(len(labels), 4),
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            frameon=True,
            fancybox=False,
            framealpha=1.0,
            borderpad=0.25,
            columnspacing=3.4,
            handlelength=1.05,
            handletextpad=0.35,
            borderaxespad=0.0,
        )

        legend.get_frame().set_edgecolor("#B8B8B8")
        legend.get_frame().set_linewidth(0.5)
        legend.get_frame().set_facecolor("white")

        fig.subplots_adjust(
            left=0.14,
            right=0.995,
            bottom=0.14,
            top=0.76,
        )

        os.makedirs(output_dir, exist_ok=True)

        safe_model_label = model_label.lower().replace(" ", "_")
        safe_cov_key = cov_key.replace("@", "")

        base_path = os.path.join(
            output_dir,
            f"{safe_model_label}_verilogeval_cvdp_{safe_cov_key}_metrics_progression_bar",
        )

        fig.savefig(
            base_path + ".png",
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.01,
        )

        fig.savefig(
            base_path + ".pdf",
            bbox_inches="tight",
            pad_inches=0.01,
        )

        plt.close(fig)

        print(f"Saved {base_path}.png and {base_path}.pdf")


def plot_summary(summary, model_label, cov_key, output_dir):
    x_labels = list(summary.keys())
    if not x_labels:
        print(f"No valid data for {model_label}; skipping plot")
        return

    print(f"\n=== {model_label}: Combined VerilogEval+CVDP Mean Coverage ({cov_key}) ===")
    for label in x_labels:
        print(f"\n{label}:")
        for metric in METRICS:
            print(f"  {metric}: {summary[label][metric]:.4f}")

    fig_width = 4.35 if len(x_labels) > 3 else 3.85
    fig, ax = plt.subplots(figsize=(fig_width, 2.15))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f2f2f2")

    matrix = np.array(
        [[summary[label][metric] for label in x_labels] for metric in METRICS],
        dtype=float,
    )
    plot_matrix = matrix * 100.0

    x = np.arange(len(x_labels))
    for metric_idx, metric in enumerate(METRICS):
        values = plot_matrix[metric_idx]
        ax.plot(
            x,
            values,
            label=METRIC_LABELS[metric],
            color=METRIC_COLORS[metric],
            linewidth=1.55,
            marker="o",
            markersize=4.2,
            markeredgecolor="white",
            markeredgewidth=0.7,
            alpha=0.98,
            zorder=3,
        )

    average_coverages = np.mean(matrix, axis=0) * 100.0
    ax.plot(
        x,
        average_coverages,
        label="Average",
        color="#333333",
        linewidth=1.45,
        linestyle="--",
        marker="D",
        markersize=3.8,
        markerfacecolor="#333333",
        markeredgecolor="white",
        markeredgewidth=0.7,
        zorder=4,
    )

    average_label_x_offsets = np.linspace(0.07, -0.07, len(average_coverages))
    for label_idx, average_coverage in enumerate(average_coverages):
        ax.text(
            x[label_idx] + average_label_x_offsets[label_idx],
            average_coverage - 1.7,
            f"{average_coverage:.1f}%",
            ha="center",
            va="top",
            color="#3a3a3a",
            fontsize=7.4,
            zorder=5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([format_variant_label(label) for label in x_labels], rotation=0)
    ax.set_xlabel("")
    ax.set_ylabel(f"{cov_key} (%)")
    y_min = max(0, np.floor((plot_matrix.min() - 5) / 10) * 10)
    ax.set_ylim(y_min, 100)
    ax.set_yticks(np.arange(y_min, 101, 10))
    ax.set_xlim(x[0] - 0.16, x[-1] + 0.16)
    ax.grid(axis="y", color="white", linestyle="-", linewidth=1.0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=3, width=0.8, color="#555555", pad=8)
    ax.tick_params(axis="y", colors="#444444")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#bbbbbb")
    ax.spines["bottom"].set_color("#bbbbbb")
    ax.legend(
        ncol=len(METRICS) + 1,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.06),
        borderaxespad=0.0,
        frameon=False,
        handlelength=1.1,
        columnspacing=0.42,
        handletextpad=0.32,
    )

    os.makedirs(output_dir, exist_ok=True)
    safe_model_label = model_label.lower().replace(" ", "_")
    safe_cov_key = cov_key.replace("@", "")
    base_path = os.path.join(
        output_dir,
        f"{safe_model_label}_verilogeval_cvdp_{safe_cov_key}_metrics_progression",
    )
    fig.subplots_adjust(left=0.11, right=0.995, bottom=0.18, top=0.78)
    plt.savefig(base_path + ".png", dpi=300, bbox_inches="tight", pad_inches=0.015, facecolor="white")
    plt.savefig(base_path + ".pdf", bbox_inches="tight", pad_inches=0.015, facecolor="white")
    plt.close(fig)
    print(f"Saved {base_path}.png and {base_path}.pdf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cov_key", type=str, default=None, help="Single coverage key to plot, e.g. cov@1, cov@5, cov@10")
    parser.add_argument("--cov_keys", nargs="+", default=None, help="Coverage keys to plot. Defaults to cov@10 and cov@1.")
    parser.add_argument("--output_dir", type=str, default="output/coverage_metrics")
    args = parser.parse_args()
    cov_keys = args.cov_keys or ([args.cov_key] if args.cov_key else ["cov@10", "cov@1"])

    model_groups = {
        "Qwen Coder": (qwen_coder_verilog_eval_problems, qwen_coder_cvdp_problems),
        "Qwen4B": (qwen_4b_verilog_eval_problems, qwen_4b_cvdp_problems),
    }

    for cov_key in cov_keys:
        for model_label, (verilog_eval_paths, cvdp_paths) in model_groups.items():
            summary = build_summary(verilog_eval_paths, cvdp_paths, cov_key)
            plot_summary(summary, model_label, cov_key, args.output_dir)
            bar_summary = build_summary(
                verilog_eval_paths,
                cvdp_paths,
                cov_key,
                config_labels=BAR_CONFIG_LABELS,
            )
            plot_bar_summary(bar_summary, model_label, cov_key, args.output_dir)


if __name__ == "__main__":
    main()
