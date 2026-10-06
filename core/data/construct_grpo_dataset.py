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

"""Construct a GRPO prompt dataset from SFT/data-generation JSONL files.

GRPO does not need gold completions in the same way SFT does.  It needs prompts
whose generated completions can be scored reliably.  This script builds a
train-only prompt pool by:

1. reading one or more candidate JSONL files,
2. filtering out broken/trivial/impossible-looking samples using existing
   coverage metadata,
3. deduplicating by RTL, and
4. writing a GRPO-ready JSONL plus construction stats.

The output keeps the original fields by default because the current
InstructDataset loader expects at least natural_spec, rtl_code, top_module, and
testbench, and extra fields are harmless.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import statistics
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from core.data_generation.utils import compute_avg_coverage, read_jsonl
from core.metrics.vcs import report_coverage_vcs, run_sim, run_urg, run_vcs
from core.train.evaluator import _extract_testbrench
from core.train.prompts import SYS_PROMPT_WO_REASONING, USER_PROMPT_SECTIONED
from core.train.utils import cleanup_sim_dir


DEFAULT_INPUTS = [
    "output/data_merged_all/all_merged_deduplicated.jsonl",
]

DEFAULT_EVALS: list[str] = []

REQUIRED_FIELDS = ("natural_spec", "rtl_code", "top_module", "testbench")

VERILOG_KEYWORDS = {
    "module", "endmodule", "input", "output", "inout", "wire", "reg", "logic",
    "signed", "unsigned", "parameter", "localparam", "assign", "always",
    "always_comb", "always_ff", "always_latch", "initial", "begin", "end",
    "if", "else", "case", "endcase", "casex", "casez", "for", "while",
    "repeat", "forever", "generate", "endgenerate", "genvar", "posedge",
    "negedge", "or", "and", "not", "xor", "xnor", "nand", "nor",
    "function", "endfunction", "task", "endtask", "integer", "int",
    "bit", "byte", "shortint", "longint", "real", "time", "typedef",
    "struct", "enum", "union", "packed", "default", "disable", "fork",
    "join", "wait", "return",
}

BAD_RELIABILITY_PATTERNS = [
    r"VFS_ZLIB_ERROR",
    r"VFS_SDB_ERROR",
    r"dsaloader\.c",
    r"segmentation fault",
    r"cannot find .*_archive",
    r"cannot open .*\.db",
    r"database file .*corrupt",
    r"timeout while synchronizing",
    r"permission denied",
]


def strip_verilog_comments(code: str) -> str:
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    code = re.sub(r"//.*", "", code)
    return code


def normalize_rtl(code: str) -> str:
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    code = strip_verilog_comments(code)
    code = re.sub(r"^\s*`timescale.*$", "", code, flags=re.MULTILINE)
    lines = [line.strip() for line in code.split("\n")]
    lines = [line for line in lines if line]
    code = "\n".join(lines)
    return re.sub(r"\s+", " ", code).strip()


def rtl_fingerprint(code: str) -> str:
    norm = normalize_rtl(code)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def structural_fingerprint(code: str) -> str:
    token_pattern = re.compile(
        r"""
        \b\d+'[bodhBODH][0-9a-fA-F_xXzZ?]+\b |
        \b\d+\b |
        \b[A-Za-z_][A-Za-z0-9_$]*\b |
        ==|!=|<=|>=|&&|\|\||<<|>>|<<<|>>>|
        [~!%^&*+=|?:<>/\-] |
        [()\[\]{};,.@#]
        """,
        re.VERBOSE,
    )
    raw_tokens = token_pattern.findall(normalize_rtl(code))
    tokens = []
    for token in raw_tokens:
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$]*", token):
            tokens.append(token if token in VERILOG_KEYWORDS else "ID")
        else:
            tokens.append(token)
    return hashlib.sha256(" ".join(tokens).encode("utf-8")).hexdigest()


def normalize_sample_id(value: Any) -> str:
    return str(value).strip()


def nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def safe_avg_coverage(entry: dict[str, Any]) -> float | None:
    try:
        value = compute_avg_coverage(entry.get("coverage_stats"))
    except Exception:
        return None
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value):
        return None
    return value


def count_present_coverage_dims(entry: dict[str, Any]) -> int:
    stats = entry.get("coverage_stats")
    if not isinstance(stats, dict):
        return 0
    count = 0
    for key in ("line", "cond", "toggle", "fsm", "branch"):
        value = stats.get(key)
        if value is None:
            continue
        try:
            if not math.isnan(float(value)):
                count += 1
        except (TypeError, ValueError):
            continue
    return count


def text_contains_bad_reliability_marker(entry: dict[str, Any]) -> bool:
    haystack_parts = [
        str(entry.get("coverage_report", "")),
        str(entry.get("error", "")),
        str(entry.get("errors", "")),
        str(entry.get("errors_encountered", "")),
    ]
    haystack = "\n".join(haystack_parts)
    return any(re.search(pattern, haystack, flags=re.IGNORECASE) for pattern in BAD_RELIABILITY_PATTERNS)


def approx_rtl_loc(entry: dict[str, Any]) -> int:
    rtl = entry.get("rtl_code", "")
    if not isinstance(rtl, str):
        return 0
    lines = []
    for line in rtl.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("//"):
            lines.append(stripped)
    return len(lines)


def is_eval_sample(
    entry: dict[str, Any],
    eval_sample_ids: set[str],
    eval_rtl_fingerprints: set[str],
    eval_structural_fingerprints: set[str],
) -> bool:
    sample_id = normalize_sample_id(entry.get("sample_id", ""))
    if sample_id and sample_id in eval_sample_ids:
        return True

    rtl_code = entry.get("rtl_code", "")
    if not nonempty(rtl_code):
        return False

    return (
        rtl_fingerprint(rtl_code) in eval_rtl_fingerprints
        or structural_fingerprint(rtl_code) in eval_structural_fingerprints
    )


def room_to_move_score(avg_cov: float, target_cov: float) -> float:
    """Higher when coverage is close to target_cov, lower near extremes."""
    return max(0.0, 1.0 - abs(avg_cov - target_cov) / 100.0)


def validate_candidate(
    entry: dict[str, Any],
    args: argparse.Namespace,
    eval_sample_ids: set[str],
    eval_rtl_fingerprints: set[str],
    eval_structural_fingerprints: set[str],
) -> tuple[bool, str, float | None]:
    for field in REQUIRED_FIELDS:
        if not nonempty(entry.get(field)):
            return False, f"missing_{field}", None

    if is_eval_sample(entry, eval_sample_ids, eval_rtl_fingerprints, eval_structural_fingerprints):
        return False, "eval_overlap", None

    loc = approx_rtl_loc(entry)
    if loc < args.min_rtl_loc:
        return False, "rtl_too_small", None
    if args.max_rtl_loc is not None and loc > args.max_rtl_loc:
        return False, "rtl_too_large", None

    avg_cov = safe_avg_coverage(entry)
    if avg_cov is None:
        return False, "missing_or_bad_coverage", None

    if count_present_coverage_dims(entry) < args.min_coverage_dims:
        return False, "too_few_coverage_dims", avg_cov

    if avg_cov < args.min_existing_coverage:
        return False, "coverage_too_low", avg_cov

    if avg_cov > args.max_existing_coverage:
        return False, "coverage_too_high", avg_cov

    if args.drop_bad_reliability_markers and text_contains_bad_reliability_marker(entry):
        return False, "bad_reliability_marker", avg_cov

    return True, "kept", avg_cov


def load_eval_exclusions(paths: list[str]) -> tuple[set[str], set[str], set[str]]:
    sample_ids: set[str] = set()
    rtl_fingerprints: set[str] = set()
    structural_fingerprints: set[str] = set()

    for path in paths:
        if not path or not os.path.exists(path):
            continue
        for entry in read_jsonl(path):
            sample_id = normalize_sample_id(entry.get("sample_id", ""))
            if sample_id:
                sample_ids.add(sample_id)
            rtl_code = entry.get("rtl_code", "")
            if nonempty(rtl_code):
                rtl_fingerprints.add(rtl_fingerprint(rtl_code))
                structural_fingerprints.add(structural_fingerprint(rtl_code))

    return sample_ids, rtl_fingerprints, structural_fingerprints


def deduplicate_by_rtl(entries: list[dict[str, Any]], target_cov: float) -> list[dict[str, Any]]:
    best_by_fingerprint: dict[str, tuple[float, dict[str, Any]]] = {}

    for entry in entries:
        rtl_code = entry["rtl_code"]
        fingerprint = rtl_fingerprint(rtl_code)
        avg_cov = float(entry["_grpo_avg_existing_coverage"])
        score = room_to_move_score(avg_cov, target_cov)

        previous = best_by_fingerprint.get(fingerprint)
        if previous is None or score > previous[0]:
            best_by_fingerprint[fingerprint] = (score, entry)

    return [entry for _, entry in best_by_fingerprint.values()]


def probe_enabled(args: argparse.Namespace) -> bool:
    return bool(args.probe_model and args.probe_api_base)


def generate_probe_completions(entry: dict[str, Any], args: argparse.Namespace) -> list[str]:
    from openai import OpenAI

    client = OpenAI(api_key=args.probe_api_key, base_url=args.probe_api_base)
    user_prompt = USER_PROMPT_SECTIONED(
        natural_spec=entry["natural_spec"],
        rtl_code=entry["rtl_code"],
        context="",
    )

    completions: list[str] = []
    for _ in range(args.probe_num_samples):
        response = client.chat.completions.create(
            model=args.probe_model,
            messages=[
                {"role": "system", "content": SYS_PROMPT_WO_REASONING},
                {"role": "user", "content": user_prompt},
            ],
            temperature=args.probe_temperature,
            max_tokens=args.probe_max_tokens,
        )
        content = response.choices[0].message.content
        completions.append(content or "")
    return completions


def _safe_artifact_name(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))[:120] or "sample"


def compute_probe_reward(syntax_ok: bool, func_ok: bool, cov_norm: float, args: argparse.Namespace) -> float:
    """Reward used only while constructing the GRPO dataset.

    Data construction accepts the same formats as evaluation extraction. It does
    not penalize completions for omitting training-only wrapper tags such as
    <testbench> when the generated Verilog can be extracted and scored.
    """
    if not syntax_ok:
        return 0.0
    if not func_ok:
        return float(args.probe_w_syntax)
    return float(args.probe_w_syntax + args.probe_w_func + args.probe_w_cov * cov_norm)


def score_probe_completion(
    i: int,
    completion: str,
    rtl_code: str,
    top_module: str,
    output_dir: str,
    timestamp: str,
    args: argparse.Namespace,
) -> tuple[int, float, dict[str, Any]]:
    safe_top = _safe_artifact_name(top_module)
    tb_dir = os.path.join(output_dir, "generated_testbenches")
    completion_dir = os.path.join(output_dir, "generated_completions")
    os.makedirs(tb_dir, exist_ok=True)
    os.makedirs(completion_dir, exist_ok=True)

    completion_file = os.path.abspath(
        os.path.join(completion_dir, f"{timestamp}_{i}_{safe_top}_completion.txt")
    )
    with open(completion_file, "w", encoding="utf-8") as f:
        f.write(completion or "")

    extracted = _extract_testbrench([completion or ""], num_samples=1)
    testbench = extracted[0][0] if extracted and extracted[0] else ""

    tb_file = os.path.abspath(os.path.join(tb_dir, f"{timestamp}_{i}_{safe_top}_pred.v"))
    rtl_file = os.path.abspath(os.path.join(tb_dir, f"{timestamp}_{i}_{safe_top}_rtl_file.v"))
    sim_output_dir = os.path.abspath(os.path.join(tb_dir, f"{timestamp}_{i}_{safe_top}_sim"))

    with open(tb_file, "w", encoding="utf-8") as f:
        f.write(testbench)
    with open(rtl_file, "w", encoding="utf-8") as f:
        f.write(rtl_code)

    compiled_ok, compile_result = run_vcs(
        verilog_files=[rtl_file],
        testbench=tb_file,
        output_dir=sim_output_dir,
        top_module_name=top_module,
        timeout=120,
    )
    syntax_ok = bool(compiled_ok)
    func_ok = False
    cov_raw = 0.0
    cov_norm = 0.0

    if compiled_ok:
        file_name = os.path.basename(tb_file)
        simv_path = os.path.join(sim_output_dir, f"{file_name}.simv")
        sim_ok, sim_result = run_sim(
            out_file=simv_path,
            top_module_name=top_module,
            output_dir=sim_output_dir,
            timeout=120,
        )
        func_ok = bool(sim_ok)
        print(sim_result)

        if sim_ok:
            urg_ok, report_file = run_urg(
                testbench=tb_file,
                top_module_name=top_module,
                output_dir=sim_output_dir,
                timeout=120,
            )
            print(report_file)
            if urg_ok and report_file and os.path.exists(report_file):
                try:
                    cov_stats = report_coverage_vcs(report_file=report_file)
                    cov_raw = compute_avg_coverage(cov_stats)
                except Exception as exc:
                    print(f"[GRPO-PROBE] Failed to parse coverage report {report_file}: {exc}")

    cov_norm = max(0.0, min(cov_raw / 100.0, 1.0))
    reward = compute_probe_reward(syntax_ok, func_ok, cov_norm, args)

    if args.probe_keep_sim_artifacts:
        pass
    else:
        cleanup_sim_dir(sim_output_dir, keep_reports=args.probe_keep_reports)

    return i, reward, {
        "syntax_ok": int(syntax_ok),
        "func_ok": int(func_ok),
        "cov_avg_raw": float(cov_raw),
        "cov_norm": float(cov_norm),
        "reward": float(reward),
        "testbench_file": tb_file,
        "completion_file": completion_file,
        "compile_result": "" if compiled_ok else str(compile_result)[:2000],
    }


def reward_probe_coverage(
    completions: list[str],
    entry: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[float], list[dict[str, Any]]]:
    os.makedirs(args.probe_output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    max_workers = max(1, min(int(args.probe_reward_workers), len(completions))) if completions else 1

    print(f"[GRPO-PROBE] Running {len(completions)} VCS reward jobs with max_workers={max_workers}")

    if max_workers == 1:
        results = [
            score_probe_completion(
                i,
                completion,
                entry["rtl_code"],
                entry["top_module"],
                args.probe_output_dir,
                timestamp,
                args,
            )
            for i, completion in enumerate(completions)
        ]
    else:
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    score_probe_completion,
                    i,
                    completion,
                    entry["rtl_code"],
                    entry["top_module"],
                    args.probe_output_dir,
                    timestamp,
                    args,
                )
                for i, completion in enumerate(completions)
            ]
            for future in as_completed(futures):
                results.append(future.result())

    results.sort(key=lambda item: item[0])
    rewards = [reward for _, reward, _ in results]
    details = [detail for _, _, detail in results]

    print("[GRPO-PROBE] syntax_ok:", [detail["syntax_ok"] for detail in details])
    print("[GRPO-PROBE] func_ok  :", [detail["func_ok"] for detail in details])
    print("[GRPO-PROBE] cov_raw  :", [detail["cov_avg_raw"] for detail in details])
    print("[GRPO-PROBE] cov_norm :", [detail["cov_norm"] for detail in details])
    print("[GRPO-PROBE] rewards  :", rewards)

    return rewards, details


def reward_probe_entry(entry: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    completions = generate_probe_completions(entry, args)
    rewards, reward_details = reward_probe_coverage(completions, entry, args)

    mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
    reward_std = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
    nonzero_rate = sum(1 for reward in rewards if reward > args.probe_nonzero_epsilon) / len(rewards) if rewards else 0.0

    entry = dict(entry)
    entry["_grpo_probe_rewards"] = rewards
    entry["_grpo_probe_reward_details"] = reward_details
    entry["_grpo_probe_mean_reward"] = mean_reward
    entry["_grpo_probe_reward_std"] = reward_std
    entry["_grpo_probe_nonzero_reward_rate"] = nonzero_rate

    if mean_reward < args.probe_min_mean_reward:
        return entry, "probe_mean_too_low"
    if mean_reward > args.probe_max_mean_reward:
        return entry, "probe_mean_too_high"
    if reward_std < args.probe_min_reward_std:
        return entry, "probe_std_too_low"
    if nonzero_rate < args.probe_min_nonzero_rate:
        return entry, "probe_nonzero_rate_too_low"
    if nonzero_rate > args.probe_max_nonzero_rate:
        return entry, "probe_nonzero_rate_too_high"

    return entry, "kept"


def apply_reward_probe(
    entries: list[dict[str, Any]],
    args: argparse.Namespace,
    rejection_counts: Counter[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not probe_enabled(args):
        return entries, {"enabled": False}

    candidate_limit = args.probe_candidate_limit
    if candidate_limit is None:
        candidate_limit = args.max_samples * 2 if args.max_samples else len(entries)
    candidate_limit = min(candidate_limit, len(entries))

    print(
        f"Running reward probe on {candidate_limit} candidates "
        f"({args.probe_num_samples} samples each) using {args.probe_model} at {args.probe_api_base}"
    )

    probed_entries: list[dict[str, Any]] = []
    probe_failures: list[dict[str, str]] = []

    for idx, entry in enumerate(entries[:candidate_limit], start=1):
        sample_id = normalize_sample_id(entry.get("sample_id", idx))
        print(f"[probe {idx}/{candidate_limit}] sample_id={sample_id}")
        try:
            probed_entry, reason = reward_probe_entry(entry, args)
        except Exception as exc:
            rejection_counts["probe_exception"] += 1
            probe_failures.append({"sample_id": sample_id, "error": repr(exc)})
            print(f"[probe {idx}/{candidate_limit}] failed: {exc!r}")
            continue

        if reason == "kept":
            probed_entries.append(probed_entry)
        else:
            rejection_counts[reason] += 1

        print(
            f"[probe {idx}/{candidate_limit}] {reason}: "
            f"mean={probed_entry['_grpo_probe_mean_reward']:.4f}, "
            f"std={probed_entry['_grpo_probe_reward_std']:.4f}, "
            f"nonzero={probed_entry['_grpo_probe_nonzero_reward_rate']:.2f}"
        )

    return probed_entries, {
        "enabled": True,
        "model": args.probe_model,
        "api_base": args.probe_api_base,
        "candidate_limit": candidate_limit,
        "num_samples": args.probe_num_samples,
        "kept": len(probed_entries),
        "failures": probe_failures[:20],
    }


def maybe_minimal_entry(entry: dict[str, Any], minimal: bool) -> dict[str, Any]:
    if not minimal:
        return entry
    keys = [
        "sample_id",
        "dataset",
        "natural_spec",
        "rtl_code",
        "top_module",
        "testbench",
        "coverage_stats",
        "_grpo_avg_existing_coverage",
    ]
    return {key: entry.get(key) for key in keys if key in entry}


def write_jsonl(path: str, entries: list[dict[str, Any]], minimal: bool) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(maybe_minimal_entry(entry, minimal), ensure_ascii=False) + "\n")


def write_stats(path: str, stats: dict[str, Any]) -> None:
    stats_path = Path(path)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, sort_keys=True)
        f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Construct a GRPO-ready JSONL dataset.")
    parser.add_argument("--input", nargs="+", default=DEFAULT_INPUTS, help="Candidate JSONL input path(s).")
    parser.add_argument("--eval", nargs="*", default=DEFAULT_EVALS, help="Optional eval/test JSONL path(s) to exclude.")
    parser.add_argument("--output", default="output/grpo/grpo_train.jsonl", help="Output GRPO JSONL path.")
    parser.add_argument("--stats-output", default=None, help="Optional stats JSON path.")
    parser.add_argument("--max-samples", type=int, default=1000, help="Maximum samples to write after filtering.")
    parser.add_argument("--seed", type=int, default=0, help="Shuffle seed.")

    parser.add_argument("--min-existing-coverage", type=float, default=30.0)
    parser.add_argument("--max-existing-coverage", type=float, default=95.0)
    parser.add_argument("--target-existing-coverage", type=float, default=70.0)
    parser.add_argument("--min-coverage-dims", type=int, default=2)
    parser.add_argument("--min-rtl-loc", type=int, default=5)
    parser.add_argument("--max-rtl-loc", type=int, default=None)

    parser.add_argument("--minimal", action="store_true", help="Write only fields needed by current GRPO loader.")
    parser.add_argument(
        "--keep-bad-reliability-markers",
        dest="drop_bad_reliability_markers",
        action="store_false",
        help="Do not drop records whose logs mention known VCS/database instability markers.",
    )
    parser.set_defaults(drop_bad_reliability_markers=True)

    parser.add_argument("--probe-model", default=None, help="Optional OpenAI-compatible model name for reward probing.")
    parser.add_argument("--probe-api-base", default=None, help="Optional OpenAI-compatible API base, e.g. http://127.0.0.1:8000/v1.")
    parser.add_argument("--probe-api-key", default="EMPTY")
    parser.add_argument("--probe-num-samples", type=int, default=4)
    parser.add_argument("--probe-candidate-limit", type=int, default=None)
    parser.add_argument("--probe-temperature", type=float, default=1.0)
    parser.add_argument("--probe-max-tokens", type=int, default=10000)
    parser.add_argument("--probe-output-dir", default="output/grpo/reward_probe")
    parser.add_argument("--probe-reward-workers", type=int, default=1)
    parser.add_argument("--probe-keep-sim-artifacts", action="store_true")
    parser.add_argument("--probe-keep-reports", action="store_true")
    parser.add_argument("--probe-min-mean-reward", type=float, default=0.10)
    parser.add_argument("--probe-max-mean-reward", type=float, default=0.85)
    parser.add_argument("--probe-min-reward-std", type=float, default=0.03)
    parser.add_argument("--probe-min-nonzero-rate", type=float, default=0.20)
    parser.add_argument("--probe-max-nonzero-rate", type=float, default=0.90)
    parser.add_argument("--probe-nonzero-epsilon", type=float, default=1e-8)
    parser.add_argument("--probe-w-format", type=float, default=0.05)
    parser.add_argument("--probe-w-syntax", type=float, default=0.15)
    parser.add_argument("--probe-w-func", type=float, default=0.20)
    parser.add_argument("--probe-w-cov", type=float, default=0.60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    rejection_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    kept: list[dict[str, Any]] = []

    eval_sample_ids, eval_rtl_fingerprints, eval_structural_fingerprints = load_eval_exclusions(args.eval)

    for input_path in args.input:
        if not os.path.exists(input_path):
            rejection_counts["missing_input_file"] += 1
            continue
        source_entries = read_jsonl(input_path)
        source_counts[input_path] += len(source_entries)

        for entry in source_entries:
            ok, reason, avg_cov = validate_candidate(
                entry,
                args,
                eval_sample_ids,
                eval_rtl_fingerprints,
                eval_structural_fingerprints,
            )
            if not ok:
                rejection_counts[reason] += 1
                continue
            entry = dict(entry)
            entry["_grpo_avg_existing_coverage"] = avg_cov
            kept.append(entry)

    before_dedup = len(kept)
    kept = deduplicate_by_rtl(kept, target_cov=args.target_existing_coverage)
    after_dedup = len(kept)

    kept.sort(
        key=lambda entry: room_to_move_score(
            float(entry["_grpo_avg_existing_coverage"]),
            args.target_existing_coverage,
        ),
        reverse=True,
    )

    kept, probe_stats = apply_reward_probe(kept, args, rejection_counts)

    if args.max_samples and len(kept) > args.max_samples:
        top_window = kept[: max(args.max_samples * 2, args.max_samples)]
        rng.shuffle(top_window)
        kept = top_window[: args.max_samples]
        kept.sort(key=lambda entry: normalize_sample_id(entry.get("sample_id", "")))

    write_jsonl(args.output, kept, minimal=args.minimal)

    coverage_values = [float(entry["_grpo_avg_existing_coverage"]) for entry in kept]
    stats = {
        "inputs": dict(source_counts),
        "eval_exclusions": {
            "sample_ids": len(eval_sample_ids),
            "rtl_fingerprints": len(eval_rtl_fingerprints),
            "structural_fingerprints": len(eval_structural_fingerprints),
        },
        "rejections": dict(rejection_counts),
        "kept_before_dedup": before_dedup,
        "kept_after_dedup": after_dedup,
        "written": len(kept),
        "coverage": {
            "min": min(coverage_values) if coverage_values else None,
            "max": max(coverage_values) if coverage_values else None,
            "mean": sum(coverage_values) / len(coverage_values) if coverage_values else None,
        },
        "reward_probe": probe_stats,
        "args": vars(args),
    }

    stats_output = args.stats_output or str(Path(args.output).with_suffix(".stats.json"))
    write_stats(stats_output, stats)

    print(f"Wrote {len(kept)} GRPO samples to {args.output}")
    print(f"Wrote construction stats to {stats_output}")
    print(f"Rejected: {dict(rejection_counts)}")


if __name__ == "__main__":
    main()
