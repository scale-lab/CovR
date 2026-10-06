"""Plot token counts for testbench and reasoning thoughts (seaborn overlay)."""
import json
import argparse
import tiktoken
from pathlib import Path

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches


def _iter_records(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            if isinstance(data, list):
                for r in data:
                    yield r
            elif isinstance(data, dict):
                yield data
        except json.JSONDecodeError:
            f.seek(0)
            for line in f:
                if line.strip():
                    yield json.loads(line)


def _collect_tb_reason_token_counts(path: Path):
    """
    Tokenizes testbench and reasoning strings and returns two lists of token counts.
    Uses cl100k_base (GPT-4/4o compatible) tokenizer.
    """
    enc = tiktoken.get_encoding("cl100k_base")
    tb_counts, reason_counts = [], []

    for rec in _iter_records(path):
        tb = rec.get("testbench")
        rsn = rec.get("best_reasoning_trace")

        if isinstance(tb, str):
            tb = tb.strip()
            if tb:
                tb_counts.append(len(enc.encode(tb)))

        if isinstance(rsn, str):
            rsn = rsn.strip()
            if rsn:
                reason_counts.append(len(enc.encode(rsn)))

    return tb_counts, reason_counts


def plot_tb_reason_token_histogram_seaborn(input_path: str,
                                           save_path: str = None,
                                           bins: int = 50,
                                           log_scale: bool = False,
                                           title: str = ""):
    path = Path(input_path)
    tb_counts, reason_counts = _collect_tb_reason_token_counts(path)

    if not tb_counts and not reason_counts:
        print("No tokenizable 'testbench' or 'tb_generator_reasoning_trace' fields found.")
        return

    rows = []
    for v in tb_counts:
        rows.append({"series": "Testbench", "tokens": v})
    for v in reason_counts:
        rows.append({"series": "Reasoning", "tokens": v})

    df = pd.DataFrame(rows)

    sns.set_theme(style="darkgrid")
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]
    sns.set_palette(colors)

    fig, ax = plt.subplots(figsize=(6.2, 3.6))

    sns.histplot(
        data=df,
        x="tokens",
        hue="series",
        bins=bins,
        element="poly",
        stat="count",
        common_norm=False,
        alpha=0.65,
        ax=ax,
        linewidth=1.8,
        legend=True,
        log_scale=log_scale,
    )

    ax.set_xlabel("Token Count", fontsize=24)
    ax.set_ylabel("Frequency", fontsize=24)
    ax.set_title(title, fontsize=18, pad=8)
    ax.tick_params(axis="both", which="major", labelsize=18)

    if not log_scale:
        ax.set_xlim(left=0)

    if ax.get_legend():
        leg = ax.get_legend()
        leg.set_frame_on(True)
        leg.get_frame().set_facecolor("white")
        leg.get_frame().set_alpha(0.95)
        leg.get_frame().set_edgecolor("#CCCCCC")
        leg.get_frame().set_linewidth(0.8)
        leg.set_loc("upper right")
        leg.set_title(None)
        for text in leg.get_texts():
            text.set_fontsize(18)

    fig.tight_layout()

    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(out.with_suffix(".pdf"), dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved token histograms to {out.with_suffix('.png').resolve()}")
    else:
        tmp = path.with_suffix(".tb_reason_tokens.hist.seaborn.png")
        fig.savefig(tmp, dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(tmp.with_suffix(".pdf"), dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved token histograms to {tmp.resolve()}")


def plot_tb_reason_token_histogram_cbrt(input_path: str,
                                        save_path: str = None,
                                        bins: int = 50,
                                        title: str = ""):
    path = Path(input_path)
    tb_counts, reason_counts = _collect_tb_reason_token_counts(path)

    if not tb_counts and not reason_counts:
        print("No tokenizable fields found.")
        return

    all_counts = tb_counts + reason_counts
    bin_edges = np.linspace(0, max(all_counts), bins + 1)

    def cbrt(x):
        return np.cbrt(x)

    def cbrt_inv(x):
        return np.sign(x) * (np.abs(x) ** 3)

    palette_map = {"Testbench": "#3498db", "Reasoning": "#e74c3c"}

    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots(figsize=(6.2, 3.6))

    # Reasoning first so Testbench (usually smaller) renders on top
    for label, vals in [("Reasoning", reason_counts), ("Testbench", tb_counts)]:
        if not vals:
            continue
        counts, edges = np.histogram(vals, bins=bin_edges)
        cbrt_counts = cbrt(counts.astype(float))
        ax.fill_between(
            np.repeat(edges, 2)[1:-1],
            np.repeat(cbrt_counts, 2),
            alpha=0.4,
            color=palette_map[label],
            linewidth=0,
        )
        ax.step(
            edges[:-1],
            cbrt_counts,
            where="post",
            color=palette_map[label],
            linewidth=1.6,
        )

    # Y ticks in original count space
    y_max_cbrt = ax.get_ylim()[1]
    candidates = [0, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    major_ticks_actual = [t for t in candidates if cbrt(t) <= y_max_cbrt * 1.05]
    idx = np.round(np.linspace(0, len(major_ticks_actual) - 1, min(6, len(major_ticks_actual)))).astype(int)
    major_ticks_actual = [major_ticks_actual[i] for i in idx]

    ax.set_yticks([cbrt(t) for t in major_ticks_actual])
    ax.set_yticklabels([str(t) for t in major_ticks_actual])
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)

    ax.set_xlabel("Token Count", fontsize=24)
    ax.set_ylabel("Frequency", fontsize=24)
    ax.tick_params(axis="both", which="major", labelsize=18)

    if title:
        ax.set_title(title, fontsize=18, pad=8)

    handles = [
        mpatches.Patch(facecolor=palette_map[label], edgecolor=palette_map[label],
                       linewidth=1.6, alpha=0.6, label=label)
        for label in ["Testbench", "Reasoning"]
    ]
    leg = ax.legend(handles=handles, loc="upper right", frameon=True, fontsize=16)
    if leg:
        frame = leg.get_frame()
        frame.set_facecolor("white")
        frame.set_alpha(0.95)
        frame.set_edgecolor("#CCCCCC")
        frame.set_linewidth(0.8)
        leg.set_title(None)

    if title:
        ax.set_title(title, fontsize=18, pad=8)

    fig.tight_layout()

    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        cbrt_out = out.parent / (out.stem + "_cbrt")
        fig.savefig(cbrt_out.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(cbrt_out.with_suffix(".pdf"), dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved cbrt token histogram to {cbrt_out.with_suffix('.png').resolve()}")
    else:
        tmp = path.with_suffix(".tb_reason_tokens.hist.cbrt.png")
        fig.savefig(tmp, dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(tmp.with_suffix(".pdf"), dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved cbrt token histogram to {tmp.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Plot TB & Reasoning token histograms from dataset JSON/JSONL.")
    parser.add_argument("--input", required=True, help="Path to dataset (.json or .jsonl)")
    parser.add_argument("--save-tokens", default=None, help="Output path (no extension) for token hist (.png/.pdf)")
    parser.add_argument("--tb-bins", type=int, default=50, help="Histogram bins (default 50)")
    parser.add_argument("--tb-log-scale", action="store_true", help="Use log scale on x-axis for seaborn plot")
    args = parser.parse_args()

    plot_tb_reason_token_histogram_seaborn(
        args.input,
        save_path=args.save_tokens,
        bins=args.tb_bins,
        log_scale=args.tb_log_scale,
        title="",
    )
    plot_tb_reason_token_histogram_cbrt(
        args.input,
        save_path=args.save_tokens,
        bins=args.tb_bins,
        title="",
    )


if __name__ == "__main__":
    main()