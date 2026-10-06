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

"""
Deduplicate a JSONL dataset by RTL code, keeping the sample with the
highest average coverage among duplicates.

Pass 1: normalized RTL dedup
Pass 2: choose best sample by coverage within each normalized group
Pass 3: safer structural dedup (identifier-normalized only; numbers preserved)

Additionally:
- Dump pass3 duplicate pairs to a text file for debugging:
    fingerprint
    kept RTL
    removed RTL
"""

import os
import re
import json
import hashlib
import argparse
from core.utils import get_logger
from core.data_generation.utils import compute_avg_coverage


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
    "join", "wait", "return"
}


def strip_verilog_comments(code: str) -> str:
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    code = re.sub(r"//.*", "", code)
    return code


def normalize_rtl(code: str) -> str:
    if not code:
        return ""

    code = code.replace("\r\n", "\n").replace("\r", "\n")
    code = strip_verilog_comments(code)

    # Remove `timescale
    code = re.sub(r"^\s*`timescale.*$", "", code, flags=re.MULTILINE)

    lines = [line.strip() for line in code.split("\n")]
    lines = [line for line in lines if line]

    code = "\n".join(lines)
    code = re.sub(r"\s+", " ", code).strip()
    return code


def rtl_fingerprint(code: str) -> str:
    norm = normalize_rtl(code)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def tokenize_verilog_structure(code: str):
    """
    Safer structural tokenizer for deduplication.

    Goal:
      - ignore comments/formatting
      - normalize user-defined identifiers to ID
      - preserve numeric literals
      - preserve Verilog keywords and operators/punctuation

    This is intentionally conservative to reduce false merges.
    """
    code = normalize_rtl(code)

    token_pattern = re.compile(
        r"""
        \b\d+'[bodhBODH][0-9a-fA-F_xXzZ?]+\b |   # sized literals: 8'hFF
        \b\d+\b |                               # plain integers
        \b[A-Za-z_][A-Za-z0-9_$]*\b |           # identifiers/keywords
        ==|!=|<=|>=|&&|\|\||<<|>>|<<<|>>>|      # multi-char operators
        [~!%^&*+=|?:<>/\-] |                    # single-char operators
        [()\[\]{};,.@#]                         # punctuation
        """,
        re.VERBOSE,
    )

    raw_tokens = token_pattern.findall(code)
    tokens = []

    for tok in raw_tokens:
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$]*", tok):
            if tok in VERILOG_KEYWORDS:
                tokens.append(tok)
            else:
                tokens.append("ID")
        else:
            # Keep numbers, punctuation, operators as-is
            tokens.append(tok)

    return tokens


def structural_fingerprint(code: str) -> str:
    tokens = tokenize_verilog_structure(code)
    token_stream = " ".join(tokens)
    return hashlib.sha256(token_stream.encode("utf-8")).hexdigest()


def safe_compute_avg_coverage(entry, logger=None):
    """
    Wrapper around compute_avg_coverage so bad/missing coverage does not crash dedup.
    Returns -1.0 if coverage cannot be computed.
    """
    try:
        cov = compute_avg_coverage(entry["coverage_stats"])
        if cov is None:
            return -1.0
        return float(cov)
    except Exception as e:
        if logger is not None:
            sample_id = entry.get("sample_id", "unknown")
            logger.warning(
                f"YELLOW: Failed to compute coverage for sample_id={sample_id}: {e}"
            )
        return -1.0


def choose_better_entry(existing_entry, new_entry, logger=None):
    existing_cov = safe_compute_avg_coverage(existing_entry, logger)
    new_cov = safe_compute_avg_coverage(new_entry, logger)

    if new_cov > existing_cov:
        return new_entry, "replaced_higher_coverage"
    if new_cov < existing_cov:
        return existing_entry, "kept_higher_coverage"

    existing_fields = sum(v is not None and v != "" for v in existing_entry.values())
    new_fields = sum(v is not None and v != "" for v in new_entry.values())

    if new_fields > existing_fields:
        return new_entry, "replaced_more_complete"
    if new_fields < existing_fields:
        return existing_entry, "kept_more_complete"

    existing_tb = (existing_entry.get("testbench", "") or "").strip()
    new_tb = (new_entry.get("testbench", "") or "").strip()

    existing_has_tb = bool(existing_tb)
    new_has_tb = bool(new_tb)

    if new_has_tb and not existing_has_tb:
        return new_entry, "replaced_has_testbench"
    if existing_has_tb and not new_has_tb:
        return existing_entry, "kept_has_testbench"

    existing_tb_len = len(existing_tb)
    new_tb_len = len(new_tb)

    if new_tb_len < existing_tb_len:
        return new_entry, "replaced_shorter_testbench"
    if new_tb_len > existing_tb_len:
        return existing_entry, "kept_shorter_testbench"

    return existing_entry, "kept_existing_tie"


def ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def dedup_entries_by_fingerprint(
    entries,
    fingerprint_fn,
    logger,
    stage_name="dedup",
    dump_duplicate_pairs_file=None
):
    """
    Generic dedup pass:
      - group by fingerprint
      - keep best entry in each group using choose_better_entry

    If dump_duplicate_pairs_file is provided, every duplicate encountered in this pass
    is written as:
      - fingerprint
      - kept rtl_code
      - removed rtl_code
    """
    best_by_fp = {}
    duplicate_count = 0
    replacement_count = 0
    duplicate_pairs = []

    for entry in entries:
        rtl_code = entry.get("rtl_code", "")
        if not isinstance(rtl_code, str) or not rtl_code.strip():
            continue

        fp = fingerprint_fn(rtl_code)

        if fp not in best_by_fp:
            best_by_fp[fp] = entry
            continue

        duplicate_count += 1
        previous_best = best_by_fp[fp]
        chosen_entry, reason = choose_better_entry(previous_best, entry, logger)

        if chosen_entry is not previous_best:
            replacement_count += 1
            old_cov = safe_compute_avg_coverage(previous_best, logger)
            new_cov = safe_compute_avg_coverage(entry, logger)
            logger.info(
                f"YELLOW: [{stage_name}] Replaced duplicate for fp={fp[:12]}... "
                f"old_cov={old_cov:.4f}, new_cov={new_cov:.4f}, reason={reason}"
            )

        # Determine kept vs removed for debug dumping
        if chosen_entry is previous_best:
            kept_entry = previous_best
            removed_entry = entry
        else:
            kept_entry = entry
            removed_entry = previous_best

        duplicate_pairs.append({
            "fingerprint": fp,
            "reason": reason,
            "kept_entry": kept_entry,
            "removed_entry": removed_entry,
        })

        best_by_fp[fp] = chosen_entry

    if dump_duplicate_pairs_file:
        ensure_parent_dir(dump_duplicate_pairs_file)
        with open(dump_duplicate_pairs_file, "w", encoding="utf-8") as f:
            for i, pair in enumerate(duplicate_pairs, start=1):
                kept_cov = safe_compute_avg_coverage(pair["kept_entry"], logger)
                removed_cov = safe_compute_avg_coverage(pair["removed_entry"], logger)

                f.write("=" * 100 + "\n")
                f.write(f"DUPLICATE_PAIR {i}\n")
                f.write(f"FINGERPRINT: {pair['fingerprint']}\n")
                f.write(f"REASON: {pair['reason']}\n")
                f.write(f"KEPT_COVERAGE: {kept_cov:.4f}\n")
                f.write(f"REMOVED_COVERAGE: {removed_cov:.4f}\n")
                f.write("-" * 100 + "\n")
                f.write("KEPT RTL:\n")
                f.write(pair["kept_entry"].get("rtl_code", ""))
                f.write("\n")
                f.write("-" * 100 + "\n")
                f.write("REMOVED RTL:\n")
                f.write(pair["removed_entry"].get("rtl_code", ""))
                f.write("\n")

        logger.info(
            f"GREEN: [{stage_name}] dumped {len(duplicate_pairs)} duplicate pairs to "
            f"{dump_duplicate_pairs_file}"
        )

    logger.info(
        f"GREEN: [{stage_name}] kept {len(best_by_fp)} unique groups, "
        f"removed {duplicate_count} duplicates, replacements={replacement_count}"
    )

    return list(best_by_fp.values()), duplicate_count, replacement_count


def deduplicate_keep_best(data_file, output_file, logger, pass3_duplicates_out=None):
    """
    Full pipeline:
      Pass 1: normalized RTL dedup
      Pass 2: best-of-group selection is handled during pass 1 itself
      Pass 3: safer structural token-based dedup

    Optionally dumps all duplicate pairs encountered in pass3.
    """
    if not os.path.exists(data_file):
        logger.error(f"RED: Input file {data_file} does not exist.")
        return

    entries = []
    total_count = 0
    missing_rtl_count = 0
    parse_error_count = 0

    with open(data_file, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            total_count += 1

            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                logger.error(f"RED: Failed to parse JSON on line {line_no}: {e}")
                parse_error_count += 1
                continue

            rtl_code = entry.get("rtl_code", "")
            if not isinstance(rtl_code, str) or not rtl_code.strip():
                logger.warning(
                    f"YELLOW: Missing or invalid 'rtl_code' at line {line_no} — skipping."
                )
                missing_rtl_count += 1
                continue

            entries.append(entry)

    logger.info(f"GREEN: Loaded {len(entries)} valid entries from {data_file}")

    # Pass 1 + Pass 2
    pass12_entries, pass12_dups, pass12_replacements = dedup_entries_by_fingerprint(
        entries,
        rtl_fingerprint,
        logger,
        stage_name="pass1_normalized_rtl",
        dump_duplicate_pairs_file=pass3_duplicates_out,
    )

    # Pass 3
    # pass3_entries, pass3_dups, pass3_replacements = dedup_entries_by_fingerprint(
    #     pass12_entries,
    #     structural_fingerprint,
    #     logger,
    #     stage_name="pass3_structural_tokens",
    #     dump_duplicate_pairs_file=pass3_duplicates_out,
    # )

    pass3_entries = pass12_entries
    ensure_parent_dir(output_file)
    with open(output_file, "w", encoding="utf-8") as out:
        for entry in pass3_entries:
            json.dump(entry, out, ensure_ascii=False)
            out.write("\n")

    logger.info(f"GREEN: Processed {total_count} total entries")
    logger.info(f"YELLOW: Missing rtl_code skipped: {missing_rtl_count}")
    logger.info(f"RED: JSON parse errors: {parse_error_count}")

    logger.info(
        f"GREEN: Pass 1/2 result: {len(pass12_entries)} entries "
        f"(duplicates={pass12_dups}, replacements={pass12_replacements})"
    )
    # logger.info(
    #     f"GREEN: Pass 3 result: {len(pass3_entries)} entries "
    #     f"(duplicates={pass3_dups}, replacements={pass3_replacements})"
    # )

    if pass3_duplicates_out:
        logger.info(f"GREEN: Pass 3 duplicate pairs dump written to {pass3_duplicates_out}")

    logger.info(f"GREEN: Final deduplicated dataset written to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Deduplicate JSONL by RTL and keep the sample with highest average coverage"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="output/data_merged/all_merged.jsonl",
        help="Path to input JSONL file",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="output/data_merged/all_merged_deduplicated.jsonl",
        help="Path to output deduplicated JSONL file",
    )
    parser.add_argument(
        "--pass3-duplicates-out",
        type=str,
        default="output/data_merged/pass3_duplicate_pairs.txt",
        help="Path to text file where pass3 duplicate pairs will be written",
    )

    args = parser.parse_args()
    logger = get_logger("deduplicate_json_rtl.log")

    deduplicate_keep_best(
        data_file=args.data,
        output_file=args.out,
        logger=logger,
        pass3_duplicates_out=args.pass3_duplicates_out,
    )


if __name__ == "__main__":
    main()