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


# ===================================================================================================
# Expert vs CorrectBench Grouped Performance Plotter for VerilogEval and RTLLM
# ===================================================================================================

import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# import json
# from statistics import mean

# def compute_metrics(data):
#     verilogeval = []
#     rtllm = []

#     for name, vals in data.items():
        
#         # Extract the required metrics
#         pass10_syntax = vals.get("pass@10_syntax")
#         pass10_func = vals.get("pass@10_func")
        
#         cov10 = vals.get("cov@10", {})
#         # if cov10:
#         #     avg_cov10 = mean(cov10.values())
#         # else:
#         #     avg_cov10 = 0
#         nonzero_cov = [v for v in cov10.values() if v > 0]

#         if len(nonzero_cov) > 0:
#             avg_cov10 = mean(nonzero_cov)
        
#         record = {
#             "name": name,
#             "pass10_syntax": pass10_syntax,
#             "pass10_func": pass10_func,
#             "avg_cov10": avg_cov10
#         }
        
#         # Split into groups
#         if name.startswith("Prob"):
#             verilogeval.append(record)
#         else:
#             rtllm.append(record)

#     # Compute group averages
#     def group_avg(group):
#         if not group:
#             return {"pass10_syntax": 0, "pass10_func": 0, "avg_cov10": 0}
#         return {
#             "pass10_syntax": mean([x["pass10_syntax"] for x in group]),
#             "pass10_func": mean([x["pass10_func"] for x in group]),
#             "avg_cov10": mean([x["avg_cov10"] for x in group]),
#         }

#     return {
#         "verilogeval_avg": group_avg(verilogeval),
#         "rtllm_avg": group_avg(rtllm) #,
#         # "verilogeval_details": verilogeval,
#         # "rtllm_details": rtllm,
#     }


# # -------------------------
# # Example usage:
# # -------------------------

# data = json.load(open("/users/mnouh/data/mnouh/ScaleLab/LLM4TB/output/coverage_metrics/result_eval.json"))
# results = compute_metrics(data)
# print(results)


# -------------------------
# Data
# -------------------------
sources = ["VerilogEval", "RTLLM"]

verilog_eval_expert = [1, 1, 0.966]
verilog_eval_correct = [0.91667, 0.429487, 0.360313]

rtllm_expert = [1, 1, 0.827311]
rtllm_correct = [0.578947, 0.131578, 0.101338]

# verilog_eval_ours = [98.08, 84.62, 96.13]    # Syntax, Functional, Coverage
# rtllm_ours = [82.98, 70.21, 93.40]         # Syntax, Functional, Coverage

verilog_eval_ours = [0.9808, 0.8077, 0.7576]    # Syntax, Functional, Coverage
rtllm_ours = [0.8298, 0.6809, 0.5986]         # Syntax, Functional, Coverage

ours_data = np.array([
    verilog_eval_ours,
    rtllm_ours
])

correct_data = np.array([
    verilog_eval_correct,
    rtllm_correct
])

expert_data = np.array([
    verilog_eval_expert,
    rtllm_expert
])

metrics = ["Syntax", "Functional", "Coverage"]

# Colors to match paper style
metric_colors = {
    "Syntax": "#b19cd9",
    "Functional": "#4D85EA",
    "Coverage": "#8DBA73"
}

# To collect star positions for each source
expert_star_x = [[], []]   # 0 = VerilogEval, 1 = RTLLM
expert_star_y = [[], []]

# If you want lines for your own approach too:
ours_star_x = [[], []]
ours_star_y = [[], []]

# -------------------------
# Visual tuning parameters
# -------------------------
# Make these smaller to squeeze bars together
width = 0.1             # bar width (smaller -> tighter)
offset_factor = 1     # compress intra-group offsets (0.0..1.0)
# Make the two source groups closer by using x positions < 1 apart
x = np.array([0.0, 0.4])

# -------------------------
# Apply seaborn matching style
# -------------------------
sns.set_theme(style="darkgrid")
plt.rcParams.update({
    "axes.facecolor": "#f2f2f2",
    "figure.facecolor": "white",
    "axes.titlesize": 18,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 10,
})

# -------------------------
# Plot
# -------------------------
fig, ax = plt.subplots(figsize=(7, 3.4))  # paper-sized

# Bars + Stars + Arrows
for i in range(3):
    metric = metrics[i]
    color = metric_colors[metric]

    # compute compacted offsets
    pos = x + (i - 1) * width * offset_factor

    # Bars = CorrectBench
    ax.bar(
        pos,
        correct_data[:, i],
        width,
        label=metric if i == 0 else None,
        color=color,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.9,
        zorder=2
    )

    # Stars = Expert
    ax.scatter(
        pos - (width * 0.25),
        expert_data[:, i],
        marker="*",
        s=220,
        color="white",
        edgecolors="black",
        linewidth=1.0,
        zorder=5
    )

    # SECOND STAR = Our Approach
    ax.scatter(
        pos + (width * 0.25),       # slight offset so they don't overlap
        ours_data[:, i],
        marker="*",
        s=220,
        color="black",              # filled black star to distinguish
        edgecolors="black",
        linewidth=1.0,
        zorder=6
    )

    # Arrow from bar → star
    for j in range(len(x)):
        ax.annotate(
            "",
            xy=(pos[j] - (width * 0.25), expert_data[j, i]),
            xytext=(pos[j] - (width * 0.25), correct_data[j, i]),
            # xy=(pos[j], expert_data[j, i]),
            # xytext=(pos[j], correct_data[j, i]),
            arrowprops=dict(
                arrowstyle="<->",
                color="black",
                linestyle="dashed",
                lw=1.0
            ),
            zorder=3
        )
    
    for j in range(len(x)):
        ax.annotate(
            "",
            xy=(pos[j] + (width * 0.25), ours_data[j, i]),
            xytext=(pos[j] + (width * 0.25), correct_data[j, i]),
            # xy=(pos[j] - (width * 0.25), expert_data[j, i]),
            # xytext=(pos[j]- (width * 0.25), correct_data[j, i]),
            arrowprops=dict(
                arrowstyle="<->",
                color="black",
                linestyle="dashed",
                lw=1.0
            ),
            zorder=3
        )

    # Save Expert star positions
    for j in range(len(x)):
        expert_star_x[j].append(pos[j])
        expert_star_y[j].append(expert_data[j, i])
    
    # Save Ours star positions
    for j in range(len(x)):
        ours_star_x[j].append(pos[j])  
        ours_star_y[j].append(ours_data[j, i])


# Lines connecting expert stars
# for src in range(2):  # 0 = VE, 1 = RTLLM
#     ax.plot(
#         expert_star_x[src],
#         expert_star_y[src],
#         linestyle='-', 
#         linewidth=1.2,
#         color='black',
#         zorder=4
#     )

# # Lines connecting Ours stars (optional)
# for src in range(2):
#     ax.plot(
#         ours_star_x[src],
#         ours_star_y[src],
#         linestyle='--',
#         linewidth=1.2,
#         color='black',
#         zorder=4
#     )


# -------------------------
# Formatting
# -------------------------
ax.set_xticks(x)
ax.set_xlim(-0.2, x[1] + 0.45)
ax.set_xticklabels(sources)
ax.set_ylabel("Metric")
# ax.set_ylim(0, 115)
ax.set_ylim(0, 1.1)

# Create compact legend placed to upper-right OUTSIDE axes
handles = [
    plt.Rectangle((0, 0), 1, 1, color=metric_colors[m], alpha=0.85)
    for m in metrics
]

labels = metrics

expert_handle = Line2D(
    [0], [0],
    marker='*',
    color='white',              # marker face
    markeredgecolor='black',    # outline color
    markeredgewidth=1.5,        # outline thickness
    linestyle='None',
    markersize=12,
    label='Expert (star)'
)
ours_handle = Line2D(
    [0], [0],
    marker='*',
    color='black',
    markeredgecolor='black',
    linestyle='None',
    markersize=12,
    label='Ours (star)'
)


handles.append(expert_handle)
labels.append("Expert")

handles.append(ours_handle)
labels.append("CovR (ours)")

legend = ax.legend(
    handles,
    labels,
    title="Metrics",
    title_fontsize=10,
    loc='upper right',   # <-- now inside the plot
    frameon=True,
    borderpad=0.4
    # loc='upper left',
    # bbox_to_anchor=(1.02, 1.0),
    # frameon=False,
    # borderaxespad=0.
)

plt.tight_layout()
plt.show()
fig.savefig("expert_vs_correctbench_grouped_performance_compact.png",
            dpi=300, bbox_inches="tight", facecolor="white")

