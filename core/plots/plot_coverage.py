"""
Plots coverage scatter distributions and an overlaid histogram from a JSON/JSONL dataset.

Each point = one design entry. Skips None values automatically.
"""

import argparse
import json
import math
import numpy as np 
from pathlib import Path

import seaborn as sns
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS_DEFAULT = ["line", "toggle", "combinational", "fsm", "branch"]


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


def _collect_metrics(path: Path, metrics):
    buckets = {m: [] for m in metrics}
    for rec in _iter_records(path):
        stats = rec.get("coverage_stats") or {}
        for m in metrics:
            v = stats.get(m)
            if isinstance(v, (int, float)) and v != 0:
                buckets[m].append(float(v))
    return buckets


def _to_percent_if_needed(vals):
    """If values look like fractions (<=1.5), convert to percent."""
    if not vals:
        return vals
    if max(vals) <= 1.5:
        return [v * 100.0 for v in vals]
    return vals


def plot_overlay_histogram_seaborn(input_path: str,
                                   metrics=None,
                                   save_path=None,
                                   bins=20,
                                   density=False,
                                   title="",
                                   log_scale=False):  # Added parameter

    path = Path(input_path)
    if metrics is None:
        metrics = METRICS_DEFAULT

    data = _collect_metrics(path, metrics)

    # Flatten into long-form DataFrame (required for seaborn)
    rows = []
    for m, values in data.items():
        for v in values:
            if isinstance(v, (int, float)):
                rows.append({"metric": m.capitalize(), "coverage": v})
    if not rows:
        print("No numeric coverage data to plot.")
        return

    df = pd.DataFrame(rows)

    # Detect whether conversion to percent is needed
    if df["coverage"].max() <= 1.5:
        df["coverage"] = df["coverage"] * 100.0

    sns.set_theme(
        context="paper",
        style="whitegrid",
        rc={
            "axes.edgecolor": "#333333",
            "axes.facecolor": "#F7F7F7",
            "axes.linewidth": 0.8,
            "font.family": "DejaVu Sans",
            "grid.color": "#DADADA",
            "grid.linewidth": 0.7,
        },
    )
    
    colors = ["#4C78A8", "#E45756", "#59A14F", "#F2A541", "#8E6BBE", "#72B7B2"]
    sns.set_palette(colors)

    fig, ax = plt.subplots(figsize=(5.7, 3.05))

    # Overlaid histograms
    sns.histplot(
        data=df,
        x="coverage",
        hue="metric",
        bins=bins,
        element="bars",
        multiple="layer",
        stat="percent" if density else "count",
        common_norm=False,
        alpha=0.5,
        ax=ax,
        edgecolor="white",
        linewidth=0.3,
        legend=True,
        log_scale=log_scale,  # Enable log scale
    )

    # Labels and title
    ax.set_xlabel("Coverage (%)", fontsize=11.5)
    ax.set_ylabel("Samples per bin (%)" if density else "Frequency", fontsize=11.5)
    ax.tick_params(axis="both", which="major", labelsize=9.5)

    # Legend styling - only if legend exists
    if ax.get_legend():
        leg = ax.get_legend()
        handles = getattr(leg, "legend_handles", None)
        if handles is None:
            handles = getattr(leg, "legendHandles")
        labels = [text.get_text() for text in leg.get_texts()]
        leg.remove()

        leg = ax.legend(
            handles=handles,
            labels=labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=len(labels),
            columnspacing=0.65,
            handlelength=1.0,
            handletextpad=0.35,
            borderaxespad=0.0,
            frameon=True,
            fontsize=9.2,
        )
        leg.set_frame_on(True)
        leg.get_frame().set_facecolor("white")
        leg.get_frame().set_alpha(0.95)
        leg.get_frame().set_edgecolor("#CCCCCC")
        leg.get_frame().set_linewidth(0.8)
        leg.set_title(None)
        for text in leg.get_texts():
            text.set_fontsize(9.2)

    # Axis limits
    if not log_scale:
        ax.set_xlim(0, 100)

    ax.grid(axis="x", visible=True)
    ax.grid(axis="y", visible=True, alpha=0.65)
    sns.despine(ax=ax)

    fig.tight_layout()

    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as both PNG and PDF
        png_path = out.with_suffix('.png')
        pdf_path = out.with_suffix('.pdf')
        
        fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor='white')
        fig.savefig(pdf_path, dpi=300, bbox_inches="tight", facecolor='white')
        
        print(f"Saved seaborn overlaid histogram to {png_path.resolve()}")
        print(f"Saved seaborn overlaid histogram to {pdf_path.resolve()}")
    else:
        # Save both formats with default naming
        tmp_png = path.with_suffix(".hist.seaborn.png")
        tmp_pdf = path.with_suffix(".hist.seaborn.pdf")
        
        fig.savefig(tmp_png, dpi=300, bbox_inches="tight", facecolor='white')
        fig.savefig(tmp_pdf, dpi=300, bbox_inches="tight", facecolor='white')
        
        print(f"Saved seaborn overlaid histogram to {tmp_png.resolve()}")
        print(f"Saved seaborn overlaid histogram to {tmp_pdf.resolve()}")



def plot_overlay_histogram_seaborn_log(
    input_path: str,
    metrics=None,
    save_path=None,
    bins=30,
    density=False,
    title="",
    zoom_range=(-1, 102),
    log_y=True,
):
    from pathlib import Path
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    path = Path(input_path)
    if metrics is None:
        metrics = METRICS_DEFAULT

    data = _collect_metrics(path, metrics)

    rows = []
    for m, values in data.items():
        values = _to_percent_if_needed(values)
        for v in values:
            if isinstance(v, (int, float)):
                rows.append({"metric": m.capitalize(), "coverage": v})

    if not rows:
        print("No numeric coverage data to plot.")
        return

    df = pd.DataFrame(rows)

    metric_order = [m.capitalize() for m in metrics if m.capitalize() in df["metric"].unique()]

    sns.set_theme(
        context="paper",
        style="whitegrid",
        rc={
            "axes.edgecolor": "#333333",
            "axes.facecolor": "#F7F7F7",
            "axes.linewidth": 0.8,
            "font.family": "DejaVu Sans",
            "grid.color": "#DADADA",
            "grid.linewidth": 0.7,
        },
    )

    colors = ["#4C78A8", "#E45756", "#59A14F", "#F2A541", "#8E6BBE", "#72B7B2"]
    palette_map = {metric: colors[i % len(colors)] for i, metric in enumerate(metric_order)}

    fig, ax = plt.subplots(figsize=(5.7, 3.05))

    # Draw in order (FSM last so it's visible)
    plot_order = [m for m in ["Toggle", "Line", "Cond", "Branch", "Fsm"] if m in metric_order]

    for m in plot_order:
        sub = df[df["metric"] == m]

        sns.histplot(
            data=sub,
            x="coverage",
            bins=bins,
            stat="density" if density else "count",
            element="bars",
            multiple="layer",
            alpha=0.5,
            common_norm=density,
            edgecolor="white",
            linewidth=0.3,
            color=palette_map[m],
            ax=ax,
            legend=False,
        )

    # -----------------------------
    # Labels (match first figure)
    # -----------------------------
    ax.set_xlabel("Coverage (%)", fontsize=11.5)
    if density:
        ylabel = "Density"
    else:
        ylabel = "Frequency (log scale)" if log_y else "Frequency"
    ax.set_ylabel(ylabel, fontsize=11.5)
    ax.tick_params(axis="both", which="major", labelsize=9.5)

    if zoom_range is not None:
        ax.set_xlim(*zoom_range)

    if log_y:
        ax.set_yscale("log")

    # -----------------------------
    # Legend (match styling)
    # -----------------------------
    handles = [
        Patch(facecolor=palette_map[label], edgecolor="none", alpha=0.65, label=label)
        for label in metric_order
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            linewidth=1.4,
            label="Median",
        )
    )
    leg = ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(handles),
        columnspacing=0.55,
        handlelength=1.0,
        handletextpad=0.3,
        borderaxespad=0.0,
        frameon=True,
        fontsize=8.8,
    )

    if leg:
        frame = leg.get_frame()
        frame.set_facecolor("white")
        frame.set_alpha(0.95)
        frame.set_edgecolor("#CCCCCC")
        frame.set_linewidth(0.8)
        leg.set_title(None)

        for text in leg.get_texts():
            text.set_fontsize(8.8)

    # -----------------------------
    # Median lines (cleaner)
    # -----------------------------
    label_y_levels = [0.88, 0.78, 0.68, 0.58, 0.50, 0.42]

    for i, m in enumerate(metric_order):
        sub = df[df["metric"] == m]["coverage"]
        median = sub.median()

        ax.axvline(
            median,
            color=palette_map[m],
            linestyle="--",
            linewidth=1.4,
            alpha=0.85
        )

        x_right = ax.get_xlim()[1]
        if median > x_right - 10:
            x_offset_points = -8
            ha = "right"
        else:
            x_offset_points = 8
            ha = "left"

        ax.annotate(
            f"{m} {median:.0f}%",
            xy=(median, label_y_levels[i % len(label_y_levels)]),
            xycoords=("data", "axes fraction"),
            xytext=(x_offset_points, 0),
            textcoords="offset points",
            color=palette_map[m],
            fontsize=8.8,
            va="center",
            ha=ha,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=0.25),
            zorder=4,
        )

    if title:
        ax.set_title(title, fontsize=11.5, pad=8)

    ax.grid(axis="x", visible=True)
    ax.grid(axis="y", visible=True, alpha=0.65)
    sns.despine(ax=ax)

    fig.tight_layout()

    # -----------------------------
    # Save
    # -----------------------------
    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(out.with_suffix(".pdf"), dpi=300, bbox_inches="tight", facecolor="white")
    else:
        fig.savefig(path.with_suffix(".hist_log.png"), dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(path.with_suffix(".hist_log.pdf"), dpi=300, bbox_inches="tight", facecolor="white")


def plot_overlay_histogram_seaborn_cbrt(
    input_path: str,
    metrics=None,
    save_path=None,
    bins=50,
    density=False,
    title="",
    zoom_range=(-1, 102),
    show_median=True,
):
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches

    path = Path(input_path)
    if metrics is None:
        metrics = METRICS_DEFAULT

    data = _collect_metrics(path, metrics)

    rows = []
    for m, values in data.items():
        values = _to_percent_if_needed(values)
        for v in values:
            if isinstance(v, (int, float)) and np.isfinite(v):
                rows.append({"metric": m.capitalize(), "coverage": v})

    if not rows:
        print("No numeric coverage data to plot.")
        return

    df = pd.DataFrame(rows)
    metric_order = [m.capitalize() for m in metrics if m.capitalize() in df["metric"].unique()]

    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]
    palette_map = {metric: colors[i % len(colors)] for i, metric in enumerate(metric_order)}

    def cbrt(x):
        return np.cbrt(x)

    import matplotlib
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    import seaborn as sns
    sns.set_theme(style="darkgrid")

    fig, ax = plt.subplots(figsize=(6.8, 4.0))

    bin_edges = np.linspace(0, 100, bins + 1)
    plot_order = [m for m in ["Toggle", "Branch", "Cond", "Line", "Fsm"] if m in metric_order]

    median_info = []

    for m in plot_order:
        sub = df[df["metric"] == m]["coverage"].values
        if len(sub) == 0:
            continue

        counts, edges = np.histogram(sub, bins=bin_edges, density=density)
        cbrt_counts = cbrt(counts.astype(float))

        ax.fill_between(
            np.repeat(edges, 2)[1:-1],
            np.repeat(cbrt_counts, 2),
            alpha=0.22,
            color=palette_map[m],
            linewidth=0,
            zorder=1,
        )

        ax.step(
            edges[:-1],
            cbrt_counts,
            where="post",
            color=palette_map[m],
            linewidth=2.0,
            zorder=2,
        )

        if show_median:
            median_val = float(np.median(sub))
            median_info.append((m, median_val))

    # Y-axis ticks shown in original scale
    if density:
        y_max_cbrt = ax.get_ylim()[1]
        y_candidates = [0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
        major_ticks_actual = [t for t in y_candidates if cbrt(t) <= y_max_cbrt * 1.05]
        if len(major_ticks_actual) > 0:
            ax.set_yticks([cbrt(t) for t in major_ticks_actual])
            ax.set_yticklabels([f"{t:g}" for t in major_ticks_actual])
    else:
        candidates = [0, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
        y_max_cbrt = ax.get_ylim()[1]
        major_ticks_actual = [t for t in candidates if cbrt(t) <= y_max_cbrt * 1.05]
        if len(major_ticks_actual) > 0:
            idx = np.round(
                np.linspace(0, len(major_ticks_actual) - 1, min(6, len(major_ticks_actual)))
            ).astype(int)
            major_ticks_actual = [major_ticks_actual[i] for i in idx]
            ax.set_yticks([cbrt(t) for t in major_ticks_actual])
            ax.set_yticklabels([str(t) for t in major_ticks_actual])

    ax.yaxis.set_minor_locator(ticker.NullLocator())

    ax.set_xlabel("Coverage (%)", fontsize=24)
    ylabel = "Density" if density else "Frequency"
    ax.set_ylabel(ylabel, fontsize=22)
    ax.tick_params(axis="both", which="major", labelsize=18)

    if zoom_range is not None:
        ax.set_xlim(*zoom_range)
    ax.set_ylim(bottom=0)

    # Draw median lines and stagger labels when percentages cluster together.
    if show_median and median_info:
        import matplotlib.transforms as transforms

        median_info_sorted = sorted(median_info, key=lambda item: item[1])
        x_left, x_right = ax.get_xlim()
        cluster_gap = 3.0
        label_y_levels = [0.94, 0.84, 0.74, 0.64, 0.54, 0.44]

        clusters = []
        for item in median_info_sorted:
            if not clusters or item[1] - clusters[-1][-1][1] > cluster_gap:
                clusters.append([item])
            else:
                clusters[-1].append(item)

        blended = transforms.blended_transform_factory(ax.transData, ax.transAxes)

        label_index = 0

        for cluster in clusters:
            cluster_mid = sum(item[1] for item in cluster) / len(cluster)
            text_side = "left" if cluster_mid > x_right - 12 else "right"

            for i, (m, median_val) in enumerate(cluster):
                y_frac = label_y_levels[label_index % len(label_y_levels)]
                label_index += 1
                if text_side == "left":
                    x_offset_points = -8
                    ha = "right"
                else:
                    x_offset_points = 8
                    ha = "left"

                if m == "Fsm":
                    x_offset_points = -8
                    ha = "right"

                # Keep labels inside the plot if a median is very close to the edge.
                if median_val < x_left + 3:
                    x_offset_points = 8
                    ha = "left"
                elif median_val > x_right - 3:
                    x_offset_points = -8
                    ha = "right"

                ax.axvline(
                    median_val,
                    color=palette_map[m],
                    linestyle="--",
                    linewidth=1.8,
                    alpha=0.95,
                    zorder=3,
                )

                ax.annotate(
                    f"{m} {median_val:.0f}%",
                    xy=(median_val, y_frac),
                    xycoords=blended,
                    xytext=(x_offset_points, 0),
                    textcoords="offset points",
                    color=palette_map[m],
                    fontsize=17,
                    rotation=0,
                    va="center",
                    ha=ha,
                    zorder=4,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=0.25),
                )

    handles = [
        mpatches.Patch(
            facecolor=palette_map[label],
            edgecolor=palette_map[label],
            linewidth=1.6,
            alpha=0.5,
            label=label,
        )
        for label in plot_order
    ]
    if show_median and median_info:
        handles.append(
            mlines.Line2D(
                [0],
                [0],
                color="black",
                linestyle="--",
                linewidth=1.8,
                label="Median",
            )
        )
    leg = ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.4, 1.02),
        ncol=len(handles),
        columnspacing=0.35,
        handlelength=0.7,
        handletextpad=0.2,
        borderaxespad=0.0,
        frameon=True,
        fontsize=18,
        labelspacing=0.18,         # ↓ vertical spacing

    )
    if leg:
        frame = leg.get_frame()
        frame.set_facecolor("white")
        frame.set_alpha(0.95)
        frame.set_edgecolor("#CCCCCC")
        frame.set_linewidth(0.8)
        leg.set_title(None)

    if title:
        ax.set_title(title, fontsize=24)

    fig.tight_layout()

    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(out.with_suffix(".pdf"), dpi=300, bbox_inches="tight", facecolor="white")
    else:
        fig.savefig(path.with_suffix(".hist_cbrt.png"), dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(path.with_suffix(".hist_cbrt.pdf"), dpi=300, bbox_inches="tight", facecolor="white")

    print(f"Saved cbrt histogram to {path.resolve()}")


def plot_coverage_boxplot_seaborn(
    input_path: str,
    metrics=None,
    save_path=None,
    title="",
    show_points=True,
    show_fliers=False,
    show_median_labels=False,
    max_points_per_metric=450,
):
    path = Path(input_path)
    if metrics is None:
        metrics = METRICS_DEFAULT

    data = _collect_metrics(path, metrics)
    metric_order = []
    plot_values = []
    for metric in metrics:
        values = _to_percent_if_needed(data.get(metric, []))
        values = [v for v in values if isinstance(v, (int, float)) and np.isfinite(v)]
        if values:
            metric_order.append(metric.capitalize())
            plot_values.append(values)

    if not plot_values:
        print("No numeric coverage data to plot.")
        return

    colors = ["#9BBF9E", "#C6A6D8", "#9EB3D6", "#D9A0A0", "#D8BE82", "#9ED4CD"]
    palette_map = {metric: colors[i % len(colors)] for i, metric in enumerate(metric_order)}

    sns.set_theme(
        context="paper",
        style="whitegrid",
        rc={
            "axes.edgecolor": "#333333",
            "axes.facecolor": "#F7F7F7",
            "axes.linewidth": 0.8,
            "font.family": "DejaVu Sans",
            "grid.color": "#DADADA",
            "grid.linewidth": 0.7,
        },
    )
    row_spacing = 0.66
    positions = np.arange(len(plot_values)) * row_spacing + 1.0
    fig, ax = plt.subplots(figsize=(5.7, 2.35))

    if show_points:
        rng = np.random.default_rng(0)
        for idx, values in enumerate(plot_values):
            values = np.asarray(values)
            if max_points_per_metric and len(values) > max_points_per_metric:
                values = rng.choice(values, size=max_points_per_metric, replace=False)
            jitter = rng.uniform(-0.11, 0.11, size=len(values))
            ax.scatter(
                values,
                np.full(len(values), positions[idx]) + jitter,
                color="#222222",
                alpha=0.22,
                s=4.0,
                linewidths=0,
                zorder=1,
            )

    box = ax.boxplot(
        plot_values,
        vert=False,
        positions=positions,
        widths=0.32,
        patch_artist=True,
        showfliers=show_fliers,
        medianprops={"color": "#111111", "linewidth": 0.0},
        boxprops={"color": "#555555", "linewidth": 0.85, "zorder": 3},
        whiskerprops={"color": "#333333", "linewidth": 0.95, "zorder": 4},
        capprops={"color": "#333333", "linewidth": 1.0, "zorder": 4},
        flierprops={
            "marker": "o",
            "markersize": 2.4,
            "markerfacecolor": "white",
            "markeredgecolor": "#777777",
            "alpha": 0.35,
        },
    )

    for patch, metric in zip(box["boxes"], metric_order):
        patch.set_facecolor(palette_map[metric])
        patch.set_alpha(0.5)
        patch.set_edgecolor("#555555")

    medians = [float(np.median(values)) for values in plot_values]
    ax.scatter(
        medians,
        positions,
        marker="D",
        s=20,
        color="#111111",
        edgecolor="white",
        linewidth=0.45,
        label="Median",
        zorder=6,
    )
    for metric, median, pos in zip(metric_order, medians, positions):
        if metric == "Fsm":
            ax.annotate(
                f"{median:.0f}%",
                xy=(median, pos),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.0,
                color="#222222",
                zorder=7,
            )
        else:
            ax.annotate(
                f"{median:.0f}%",
                xy=(median, pos),
                xytext=(7, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=8.0,
                color="#222222",
                zorder=7,
            )

    if show_median_labels:
        for idx, values in enumerate(plot_values):
            median = float(np.median(values))
            label_x = median - 2.4 if median > 94 else median + 2.4
            label_ha = "right" if median > 94 else "left"
            ax.text(
                max(1.0, min(label_x, 99.0)),
                positions[idx],
                f"{median:.0f}",
                ha=label_ha,
                va="center",
                fontsize=8.5,
                color="#222222",
                bbox=dict(facecolor="white", edgecolor="#D6D6D6", linewidth=0.3, alpha=0.88, pad=1.0),
            )

    ax.set_xlabel("Coverage (%)", fontsize=11.5)
    ax.set_ylabel("")
    ax.set_xlim(0, 102)
    ax.set_ylim(positions[0] - 0.36, positions[-1] + 0.36)
    ax.set_yticks(positions)
    ax.set_yticklabels(metric_order)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.tick_params(axis="x", which="major", labelsize=9.5)
    ax.tick_params(axis="y", which="major", labelsize=10.5, length=0)
    ax.grid(axis="x", visible=True)
    ax.grid(axis="y", visible=False)
    sns.despine(ax=ax, left=True, bottom=False)

    ax.legend(
        loc="upper left",
        frameon=True,
        framealpha=0.92,
        edgecolor="#D0D0D0",
        fontsize=8.5,
        handletextpad=0.35,
        borderpad=0.35,
    )

    if title:
        ax.set_title(title, fontsize=12, pad=8)

    fig.tight_layout()

    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        png_path = out.with_suffix(".png")
        pdf_path = out.with_suffix(".pdf")
    else:
        png_path = path.with_suffix(".box.seaborn.png")
        pdf_path = path.with_suffix(".box.seaborn.pdf")

    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight", facecolor="white")

    print(f"Saved seaborn box plot to {png_path.resolve()}")
    print(f"Saved seaborn box plot to {pdf_path.resolve()}")


def plot_coverage_summary_bar_seaborn(
    input_path: str,
    metrics=None,
    save_path=None,
    title="",
    show_points=True,
    max_points_per_metric=700,
):
    path = Path(input_path)
    if metrics is None:
        metrics = METRICS_DEFAULT

    data = _collect_metrics(path, metrics)
    rows = []
    metric_values = []
    for metric in metrics:
        values = _to_percent_if_needed(data.get(metric, []))
        values = np.asarray([v for v in values if isinstance(v, (int, float)) and np.isfinite(v)])
        if len(values) == 0:
            continue
        metric_values.append(values)
        p5, median, p95 = np.percentile(values, [5, 50, 95])
        rows.append(
            {
                "metric": metric.capitalize(),
                "median": median,
                "p5": p5,
                "p95": p95,
                "mean": float(np.mean(values)),
            }
        )

    if not rows:
        print("No numeric coverage data to plot.")
        return

    df = pd.DataFrame(rows)

    sns.set_theme(
        context="paper",
        style="darkgrid",
        rc={
            "axes.edgecolor": "#333333",
            "axes.facecolor": "#EAEAF2",
            "axes.linewidth": 0.9,
            "font.family": "DejaVu Sans",
            "grid.color": "white",
            "grid.linewidth": 1.0,
        },
    )

    colors = ["#A7C7A5", "#C9B2D8", "#A9BBD8", "#D9AAA8", "#D8C49A", "#B9D6D2"]
    fig, ax = plt.subplots(figsize=(5.6, 2.45))
    y = np.arange(len(df))

    ax.barh(
        y,
        df["median"],
        height=0.58,
        color=colors[: len(df)],
        edgecolor="#555555",
        linewidth=0.9,
        alpha=0.72,
        zorder=2,
    )

    if show_points:
        rng = np.random.default_rng(0)
        for idx, values in enumerate(metric_values):
            if max_points_per_metric and len(values) > max_points_per_metric:
                values = rng.choice(values, size=max_points_per_metric, replace=False)
            jitter = rng.uniform(-0.21, 0.21, size=len(values))
            ax.scatter(
                values,
                np.full(len(values), y[idx]) + jitter,
                s=5,
                color="#222222",
                alpha=0.28,
                linewidths=0,
                zorder=3,
            )

    xerr = np.vstack([df["median"] - df["p5"], df["p95"] - df["median"]])
    ax.errorbar(
        df["median"],
        y,
        xerr=xerr,
        fmt="none",
        ecolor="#111111",
        elinewidth=1.45,
        capsize=3.5,
        capthick=1.45,
        zorder=6,
    )

    for idx, median in enumerate(df["median"]):
        ax.text(
            min(median + 1.8, 103.0),
            y[idx],
            f"{median:.0f}",
            ha="left",
            va="center",
            fontsize=8.0,
            color="#222222",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(df["metric"], fontsize=9.5)
    ax.set_xlabel("Coverage (%)", fontsize=10.5)
    ax.set_xlim(0, 108)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.tick_params(axis="x", labelsize=9.5)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", visible=True)
    ax.grid(axis="y", visible=False)
    sns.despine(ax=ax)

    if title:
        ax.set_title(title, fontsize=11, pad=8)

    fig.tight_layout()

    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        png_path = out.with_suffix(".png")
        pdf_path = out.with_suffix(".pdf")
    else:
        png_path = path.with_suffix(".summary_bar.seaborn.png")
        pdf_path = path.with_suffix(".summary_bar.seaborn.pdf")

    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight", facecolor="white")

    print(f"Saved seaborn summary bar plot to {png_path.resolve()}")
    print(f"Saved seaborn summary bar plot to {pdf_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Plot coverage scatter and overlaid histogram from dataset JSON/JSONL.")
    parser.add_argument("--input", required=True, help="Path to dataset (.json or .jsonl)")
    parser.add_argument("--save", default=None, help="Optional output file path for scatter (.png/.pdf)")
    parser.add_argument("--save-hist", default=None, help="Optional output file path for histogram (.png/.pdf)")
    parser.add_argument("--save-box", default=None, help="Optional output file path for box plot (.png/.pdf)")
    parser.add_argument("--save-ecdf", default=None, help="Optional output file path for survival/CCDF plot (.png/.pdf)")
    parser.add_argument("--save-summary", default=None, help="Optional output file path for summary bar plot (.png/.pdf)")
    parser.add_argument("--bins", type=int, default=50, help="Histogram bins (default 50)")
    parser.add_argument("--metrics", nargs="*", default=METRICS_DEFAULT, help="Metrics to include")
    parser.add_argument("--threshold", type=float, default=95.0, help="Scatter threshold line (default 95%)")
    args = parser.parse_args()


    # plot_scatter(args.input, metrics=args.metrics, save_path=args.save, threshold=args.threshold)
    plot_metrics = ['fsm', 'toggle', 'line', 'cond', 'branch']
    # plot_overlay_histogram_seaborn(args.input, metrics=plot_metrics, save_path=args.save, bins=50, density=True)
    # plot_overlay_histogram_seaborn_log(args.input, metrics=plot_metrics, save_path=args.save, bins=50, density=True)
    # plot_overlay_histogram_seaborn_cbrt(args.input, metrics=plot_metrics, save_path=args.save, bins=50, density=True)
    plot_coverage_boxplot_seaborn(args.input, metrics=plot_metrics, save_path=args.save_box)
    # plot_coverage_summary_bar_seaborn(args.input, metrics=plot_metrics, save_path=args.save_summary)

    # plot_overlay_histogram_better(args.input,  metrics=['fsm', 'toggle', 'line', 'cond', 'branch'], save_path=args.save, bins=30)

if __name__ == "__main__":
    main()
