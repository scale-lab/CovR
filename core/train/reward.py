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
import re
import contextlib
import io
import math
import tempfile

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np

from core.data_generation.utils import compute_avg_coverage
from core.metrics.vcs import report_coverage_vcs, run_sim, run_urg, run_vcs
from core.train.evaluator import Evaluator, _extract_testbrench
from core.train.utils import cleanup_sim_dir


def reward_format(
    completion
):
    testbench_match = re.search(
        r"<testbench>\s*(.*?)\s*</testbench>",
        completion or "",
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not testbench_match:
        return 0.0

    testbench = testbench_match.group(1)
    has_module = bool(re.search(r"\bmodule\b", testbench))
    has_endmodule = bool(re.search(r"\bendmodule\b", testbench))
    return 1.0 if has_module and has_endmodule else 0.0


def compute_reward(format_ok, syntax_ok, func_ok, cov_norm, w_format, w_syntax, w_func, w_cov):
    if not format_ok:
        return 0.0

    if not syntax_ok:
        return float(w_format)

    if not func_ok:
        return float(w_format + w_syntax)

    return float(w_format + w_syntax + w_func + w_cov * cov_norm)


def _score_one_completion(i, rtl_path, tb_path, top_module_name, sim_output_dir, w_func, w_cov, cov_clip):
    compiled_ok, compile_artifacts = run_vcs(
        verilog_files=[rtl_path],
        testbench=tb_path,
        output_dir=sim_output_dir,
        top_module_name=top_module_name,
        timeout = 120
    )
    
    syntax_ok = 1 if compiled_ok else 0

    if not compiled_ok or w_func == 0.0:
        return i, syntax_ok, 0, 0.0, 0.0

    file_name = os.path.basename(tb_path)    
    simv_path = os.path.join(sim_output_dir, f"{file_name}.simv")

    sim_ok, sim_result = run_sim(
        out_file=simv_path,
        top_module_name=top_module_name,
        output_dir=sim_output_dir,
        timeout = 120
    )

    func_ok = 1 if sim_ok else 0

    if not sim_ok or w_cov == 0.0:
        return i, syntax_ok, func_ok, 0.0, 0.0

    urg_ok, report_file = run_urg(
        testbench=tb_path,
        top_module_name=top_module_name,
        output_dir=sim_output_dir,
        timeout = 120
    )

    if not urg_ok or not report_file or not os.path.exists(report_file):
        cov_raw = 0.0
    else:
        try:
            cov_stats = report_coverage_vcs(report_file=report_file)
            cov_raw = compute_avg_coverage(cov_stats) # expect 0..100
        except Exception as exc:
            print(f"[GRPO] Failed to parse coverage report {report_file}: {exc}")
            cov_raw = 0.0

    cov_norm = max(cov_clip[0], min(cov_raw / 100.0, cov_clip[1]))
    return i, syntax_ok, func_ok, cov_raw, cov_norm


def reward_coverage(
    completions,
    num_samples,
    output_dir,
    *,
    step_num=None,
    max_workers=1,
    w_syntax=0.15,
    w_func=0.20,
    w_cov=0.60,
    w_format=0.05,    
    cov_clip=(0.0, 1.0),
    cleanup_sim_artifacts=True,
    keep_reports=True,
    **kwargs
):

    os.makedirs(output_dir, exist_ok=True)

    # step_tag = f"step_{step_num}" if step_num is not None else "step_unknown"
    tb_save_dir = os.path.join(output_dir, "generated_testbenches")
    completion_save_dir = os.path.join(output_dir, "generated_completions")
    os.makedirs(tb_save_dir, exist_ok=True)
    os.makedirs(completion_save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    rtl_codes = kwargs.get("rtl_code")
    top_modules = kwargs.get("top_module")
    log_metric = kwargs.get("log_metric")

    def _safe_name(value):
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))[:120]
    
    for i, completion in enumerate(completions):
        top_module = top_modules[i // num_samples] if top_modules else f"sample_{i}"
        completion_file = os.path.abspath(
            os.path.join(completion_save_dir, f"{timestamp}_{i}_{top_module}_completion.txt")
        )
        with open(completion_file, "w") as f:
            f.write(completion if completion is not None else "")

    testbenches = _extract_testbrench(completions, num_samples=num_samples)
   
    if not rtl_codes or len(rtl_codes) != len(testbenches):
        raise ValueError("verilog_files must be provided and match num_samples")

    format_scores = []
    for i, tb in enumerate(testbenches):
        completion_idx = i * num_samples
        completion = completions[completion_idx] if completion_idx < len(completions) else tb[0]
        format_score = reward_format(completion)
        format_scores.append(format_score)

    testbench_files = []
    verilog_files = []
    sim_output_dirs = []
    reward_jobs = []
    for i, (testbench, rtl_code, top_module) in enumerate(zip(testbenches, rtl_codes, top_modules)):
        safe_top_module = _safe_name(top_module)
        tb_file = os.path.abspath(os.path.join(tb_save_dir, f"{timestamp}_{i}_{safe_top_module}_pred.v")) 
        rtl_file = os.path.abspath(os.path.join(tb_save_dir, f"{timestamp}_{i}_{safe_top_module}_rtl_file.v")) 
        sim_output_dir = os.path.abspath(os.path.join(tb_save_dir, f"{timestamp}_{i}_{safe_top_module}_sim"))

        with open(tb_file, "w") as f:
            f.write(testbench[0])
        
        with open(rtl_file, "w") as f:
            f.write(rtl_code)

        testbench_files.append(tb_file)
        verilog_files.append(rtl_file)
        sim_output_dirs.append(sim_output_dir)
        if format_scores[i] > 0.0 and testbench[0].strip():
            reward_jobs.append((i, rtl_file, tb_file, top_module, sim_output_dir))

  
    n_jobs = len(testbench_files)
    syntax_ok_list = [0] * n_jobs
    func_ok_list = [0] * n_jobs
    cov_scores_raw = [0.0] * n_jobs   # 0..100
    cov_scores_norm = [0.0] * n_jobs  # 0..1

    requested_max_workers = int(max_workers)
    max_workers = max(1, min(requested_max_workers, len(reward_jobs))) if reward_jobs else 1
    skipped_jobs = n_jobs - len(reward_jobs)
    print(
        f"[GRPO] Running {len(reward_jobs)} VCS reward jobs "
        f"with max_workers={max_workers} requested_max_workers={requested_max_workers}; "
        f"skipped={skipped_jobs}"
    )

    if not reward_jobs:
        results = []
    elif max_workers == 1:
        results = [
            _score_one_completion(
                i, 
                rtl_path, 
                tb_path, 
                top_module_name, 
                sim_output_dir,
                w_func=w_func,
                w_cov=w_cov,
                cov_clip=cov_clip
            )
            for i, rtl_path, tb_path, top_module_name, sim_output_dir in reward_jobs
        ]
    else:
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _score_one_completion,
                    i,
                    rtl_path,
                    tb_path,
                    top_module_name,
                    sim_output_dir,
                    w_func=w_func,
                    w_cov=w_cov,
                    cov_clip=cov_clip
                )
                for i, rtl_path, tb_path, top_module_name, sim_output_dir in reward_jobs
            ]
            for future in as_completed(futures):
                results.append(future.result())

    for i, syntax_ok, func_ok, cov_raw, cov_norm in results:
        syntax_ok_list[i] = syntax_ok
        func_ok_list[i] = func_ok
        cov_scores_raw[i] = cov_raw
        cov_scores_norm[i] = cov_norm

    rewards = []
    details = []
    n = len(syntax_ok_list)  # or len(verilog_files)

    for i in range(n):
        format_ok = format_scores[i] > 0.0
        syntax_ok = syntax_ok_list[i] > 0
        func_ok = func_ok_list[i] > 0
        cov_norm = cov_scores_norm[i] if format_ok and syntax_ok and func_ok else 0.0

        if not format_ok:
            func_ok_list[i] = 0
            syntax_ok_list[i] = 0
            cov_scores_norm[i] = 0.0
            cov_scores_raw[i] = 0.0
        elif not syntax_ok:
            func_ok_list[i] = 0
            cov_scores_norm[i] = 0.0
            cov_scores_raw[i] = 0.0
        elif not func_ok:
            cov_scores_norm[i] = 0.0
            cov_scores_raw[i] = 0.0

        r = compute_reward(
            format_ok=format_ok,
            syntax_ok=syntax_ok,
            func_ok=func_ok,
            cov_norm=cov_norm,
            w_format=w_format,
            w_syntax=w_syntax,
            w_func=w_func,
            w_cov=w_cov,
        )
        rewards.append(float(r))
        details.append({
            "syntax_ok": int(syntax_ok_list[i]),
            "func_ok": int(func_ok_list[i]),
            "cov_avg_raw": float(cov_scores_raw[i]),   # 0..100
            "cov_norm": float(cov_scores_norm[i]),     # 0..1
            "format_score": float(format_scores[i]),   # 0 or 1
            "reward": float(r),
        })

    print("[GRPO] format_score :", format_scores)
    print("[GRPO] syntax_ok:", syntax_ok_list)
    print("[GRPO] func_ok  :", func_ok_list)
    print("[GRPO] cov_raw  :", cov_scores_raw)
    print("[GRPO] cov_norm :", cov_scores_norm)
    print("[GRPO] rewards  :", rewards)

    if cleanup_sim_artifacts:
        for sim_output_dir in sim_output_dirs:
            cleanup_sim_dir(sim_output_dir, keep_reports=keep_reports)

    if log_metric is not None and rewards:
        log_metric("reward_accuracy/format", float(np.mean(format_scores)))
        log_metric("reward_accuracy/syntax", float(np.mean(syntax_ok_list)))
        log_metric("reward_accuracy/functional", float(np.mean(func_ok_list)))
        log_metric("reward_accuracy/coverage_norm", float(np.mean(cov_scores_norm)))
        log_metric("reward_accuracy/reward", float(np.mean(rewards)))

    return rewards



def main():
    dut_rtl = """
module dut (
    input  logic clk,
    input  logic rst_n,
    input  logic en,
    input  logic [3:0] a,
    input  logic [3:0] b,
    output logic [4:0] y
);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            y <= 5'd0;
        end else if (en) begin
            y <= a + b;
        end
    end

endmodule
"""

    completions = ["""
    <think>
    I will instantiate the DUT, reset it, drive a few deterministic additions,
    and check that y updates only when en is high.
    </think>

    Here is a direct self-checking testbench.

    <testbench>
    module tb;

        logic clk;
        logic rst_n;
        logic en;
        logic [3:0] a;
        logic [3:0] b;
        logic [4:0] y;

        dut u_dut (
            .clk(clk),
            .rst_n(rst_n),
            .en(en),
            .a(a),
            .b(b),
            .y(y)
        );

        initial begin
            clk = 1'b0;
            forever #5 clk = ~clk;
        end

        initial begin
            rst_n = 1'b0;
            en = 1'b0;
            a = 4'd0;
            b = 4'd0;

            repeat (2) @(posedge clk);
            rst_n = 1'b1;

            @(posedge clk);
            en = 1'b1;
            a = 4'd3;
            b = 4'd4;

            @(posedge clk);
            if (y !== 5'd7) $fatal(1, "Expected y=7, got %0d", y);

            a = 4'd10;
            b = 4'd5;

            @(posedge clk);
            if (y !== 5'd15) $fatal(1, "Expected y=15, got %0d", y);

            en = 1'b0;
            a = 4'd1;
            b = 4'd1;

            @(posedge clk);
            if (y !== 5'd15) $fatal(1, "Expected y to hold 15, got %0d", y);

            $finish;
        end

    endmodule
    </testbench>
    """,
    """
    <think>
    To improve coverage, I will use several randomized input pairs while en is
    asserted, then deassert en before finishing.
    </think>

    The generated testbench is fenced below.

    <testbench>
    ```verilog
    module tb;

        logic clk;
        logic rst_n;
        logic en;
        logic [3:0] a;
        logic [3:0] b;
        logic [4:0] y;

        dut u_dut (
            .clk(clk),
            .rst_n(rst_n),
            .en(en),
            .a(a),
            .b(b),
            .y(y)
        );

        initial begin
            clk = 1'b0;
            forever #5 clk = ~clk;
        end

        initial begin
            rst_n = 1'b0;
            en = 1'b0;
            a = 4'd0;
            b = 4'd0;

            repeat (3) @(posedge clk);
            rst_n = 1'b1;

            repeat (8) begin
                @(posedge clk);
                en = 1'b1;
                a = $urandom_range(0, 15);
                b = $urandom_range(0, 15);

                @(posedge clk);
                if (y !== a + b) $fatal(1, "Mismatch: a=%0d b=%0d y=%0d", a, b, y);
            end

            @(posedge clk);
            en = 1'b0;
            a = 4'd15;
            b = 4'd15;

            @(posedge clk);
            $finish;
        end

    endmodule
    ```
    </testbench>
    """,
    """
    <think>
    I will cover reset behavior and corner-case additions with a small reusable
    task so the stimulus is clearer.
    </think>

    Final answer:

    <testbench>
    ```systemverilog
    module tb;

        logic clk;
        logic rst_n;
        logic en;
        logic [3:0] a;
        logic [3:0] b;
        logic [4:0] y;
        logic [4:0] expected;

        dut u_dut (
            .clk(clk),
            .rst_n(rst_n),
            .en(en),
            .a(a),
            .b(b),
            .y(y)
        );

        initial begin
            clk = 1'b0;
            forever #5 clk = ~clk;
        end

        task automatic drive_and_check(input logic [3:0] ta, input logic [3:0] tb);
            begin
                @(posedge clk);
                en = 1'b1;
                a = ta;
                b = tb;
                expected = ta + tb;

                @(posedge clk);
                if (y !== expected) begin
                    $fatal(1, "Expected %0d + %0d = %0d, got %0d", ta, tb, expected, y);
                end
            end
        endtask

        initial begin
            rst_n = 1'b0;
            en = 1'b0;
            a = 4'd0;
            b = 4'd0;
            expected = 5'd0;

            repeat (2) @(posedge clk);
            rst_n = 1'b1;

            drive_and_check(4'd0, 4'd0);
            drive_and_check(4'd15, 4'd0);
            drive_and_check(4'd0, 4'd15);
            drive_and_check(4'd15, 4'd15);
            drive_and_check(4'd8, 4'd7);

            @(posedge clk);
            en = 1'b0;
            rst_n = 1'b0;

            @(posedge clk);
            if (y !== 5'd0) $fatal(1, "Reset did not clear y, got %0d", y);

            $finish;
        end

    endmodule
    ```
    </testbench>
    """,
"""
  <think>

the moudle will have the following code... 
    module tb;

        logic clk;
        logic rst_n;
        logic en;
        logic [3:0] a;
        logic [3:0] b;
        logic [4:0] y;
        logic [4:0] expected;

        dut u_dut (
            .clk(clk),
            .rst_n(rst_n),
            .en(en),
            .a(a),
            .b(b),
            .y(y)
        );

        initial begin
            clk = 1'b0;
            forever #5 clk = ~clk;
        end

        task automatic drive_and_check(input logic [3:0] ta, input logic [3:0] tb);
            begin
                @(posedge clk);
                en = 1'b1;
                a = ta;
                b = tb;
                expected = ta + tb;

                @(posedge clk);
                if (y !== expected) begin
                    $fatal(1, "Expected %0d + %0d = %0d, got %0d", ta, tb, expected, y);
                end
            end
        endtask

        initial begin
            rst_n = 1'b0;
            en = 1'b0;
            a = 4'd0;
            b = 4'd0;
            expected = 5'd0;

            repeat (2) @(posedge clk);
            rst_n = 1'b1;

            drive_and_check(4'd0, 4'd0);
            drive_and_check(4'd15, 4'd0);
            drive_and_check(4'd0, 4'd15);
            drive_and_check(4'd15, 4'd15);
            drive_and_check(4'd8, 4'd7);

            @(posedge clk);
            en = 1'b0;
            rst_n = 1'b0;

            @(posedge clk);
            if (y !== 5'd0) $fatal(1, "Reset did not clear y, got %0d", y);

            $finish;
        end

    endmodule
"""]

    output_dir = tempfile.mkdtemp(prefix="reward_debug_")
    # print("debug output_dir:", output_dir)


    rewards = reward_coverage(
        completions,
        num_samples=1,
        output_dir="./output/reward_debug",
        rtl_code=[dut_rtl] * len(completions),
        top_module=["dut"] * len(completions),
        max_workers=3,
        w_syntax=0.05,
        w_func=0.08,
        w_cov=0.85,
        w_format=0.02,
        cleanup_sim_artifacts=True,
        keep_reports=False
    )




if __name__ == "__main__":
    main()
