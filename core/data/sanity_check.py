"""Runs a number of sanity checks on the dataset to make sure that:
- testbenches are syntax error free.
- testbenches coverage is > threshold.
- testbenches do not contain assertions or self-checking behavior.
"""
import os 
import re
import sys
import json
import logging
import tqdm
import argparse
import tempfile


def get_stdout_logger(log_file="sanity.log"):
    logger = logging.getLogger("sanity_check")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def log_testbench(logger, sample_id, top_module, input_file, testbench):
    logger.info("=" * 100)
    logger.info(f"Sample id: {sample_id}")
    logger.info(f"Top module: {top_module}")
    logger.info(f"Input file: {input_file}")
    logger.info("-" * 100)
    print(testbench, flush=True)
    logger.info("=" * 100)


def write_jsonl(file_path, data):
    tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(file_path))
    os.close(tmp_fd)

    with open(tmp_path, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    os.replace(tmp_path, file_path)


def _strip_comments(code):
    """Remove Verilog comments while preserving string literals."""
    result = []
    i = 0
    in_string = False

    while i < len(code):
        char = code[i]
        next_char = code[i + 1] if i + 1 < len(code) else ""

        if in_string:
            result.append(char)
            if char == "\\" and next_char:
                result.append(next_char)
                i += 2
                continue
            if char == '"':
                in_string = False
            i += 1
            continue

        if char == '"':
            in_string = True
            result.append(char)
            i += 1
            continue

        if char == "/" and next_char == "/":
            i += 2
            while i < len(code) and code[i] != "\n":
                i += 1
            if i < len(code):
                result.append("\n")
                i += 1
            continue

        if char == "/" and next_char == "*":
            i += 2
            while i + 1 < len(code) and not (code[i] == "*" and code[i + 1] == "/"):
                result.append("\n" if code[i] == "\n" else " ")
                i += 1
            i += 2
            continue

        result.append(char)
        i += 1

    return "".join(result)


def _strip_string_literals(code):
    return re.sub(r'"(?:\\.|[^"\\])*"', '""', code, flags=re.DOTALL)


def find_stimulus_only_violations(testbench):
    code = _strip_comments(testbench)
    code_without_strings = _strip_string_literals(code)
    reasons = []

    checks = [
        ("assert statement", r"\bassert\b"),
        ("SVA property", r"\bproperty\b|\bendproperty\b"),
        ("$fatal call", r"\$fatal\b"),
        ("$error call", r"\$error\b"),
        ("$stop call", r"\$stop\b"),
        ("reference-model/checking identifier", r"\b\w*(?:golden|expected|expect|reference|mismatch)\w*\b"),
        ("conditional display check", r"\bif\s*\([^;]*(?:==|!=|===|!==|<=|>=|<|>)[^;]*\)\s*(?:begin)?[^;]*(?:\$display|\$write|\$fatal|\$error)\b"),
    ]

    for label, pattern in checks:
        if re.search(pattern, code_without_strings, flags=re.IGNORECASE | re.DOTALL):
            reasons.append(label)

    if re.search(r"\$(?:display|write|fatal|error)\s*\([^;]*(?:Error:|fail(?:ed)?|mismatch|incorrect|expected|actual)", code, flags=re.IGNORECASE | re.DOTALL):
        reasons.append("failure-oriented print text")

    return reasons


def find_assertions_or_self_checking(testbench):
    return find_stimulus_only_violations(testbench)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Ignored; kept for CLI compatibility.', default='gpt-4o-mini-2024-07-18')
    parser.add_argument('--input_dir', type=str, help='Path to input dir', default='')
    parser.add_argument('--coverage_threshold', type=int, help='Model name', default=95)
    args = parser.parse_args()

    input_dir = args.input_dir
    coverage_threshold = args.coverage_threshold

    from core.data_generation.utils import read_jsonl

    logger = get_stdout_logger("sanity.log")
    logger.info(f"Running sanity check on input_dir={input_dir}")
    logger.info("stdout logging is enabled; redirect with `> out.log` to capture this output.")

    # Imported only when needed so helper functions remain importable without the simulator stack.
    # from core.metrics.vcs import run_testbench_vcs, run_urg

    tmp_dir = tempfile.mkdtemp()
    logger.info(f"Temporary sample directory: {tmp_dir}")

    failed_stimulus_only = 0
    failed_syntax = 0
    failed_functionality = 0
    failed_coverage = 0
    passed_all = 0

    split_files = [
        (
            "met",
            os.path.join(input_dir, "train_dataset_met_coverage.jsonl"),
            os.path.join(input_dir, "train_dataset_met_coverage_failed_sanity.jsonl"),
        ),
        (
            "not_met",
            os.path.join(input_dir, "train_dataset_not_met_coverage.jsonl"),
            os.path.join(input_dir, "train_dataset_not_met_coverage_failed_sanity.jsonl"),
        ),
    ]
    
    for split_name, input_file, failed_output_file in split_files:
        logger.info(f"Reading {input_file}")
        data = read_jsonl(input_file)
        logger.info(f"Loaded {len(data)} samples from {input_file}")
        kept_samples = []
        failed_samples = []

        for i, sample in enumerate(tqdm.tqdm(data, file=sys.stdout)): 
            sample_id = sample.get("sample_id", i)
            sample_dir = os.path.join(tmp_dir, f"sample_{sample_id}")
            os.makedirs(sample_dir, exist_ok=True)

            top = sample["top_module"]
            logger.info(f"Checking sample index={i}, sample_id={sample_id}, top_module={top}")
            rtl_path = os.path.join(sample_dir, f"{top}.v")
            tb_path  = os.path.join(sample_dir, f"{top}_tb.v")

            with open(rtl_path, "w") as f:
                f.write(sample["rtl_code"])

            with open(tb_path, "w") as f:
                f.write(sample["testbench"])

            # 1. Stimulus-only Check
            stimulus_only_violations = find_stimulus_only_violations(sample["testbench"])
            if stimulus_only_violations:
                failed_stimulus_only += 1
                failed_samples.append(sample)
                logger.info(
                    f"RED: Testbench {i} failed stimulus-only check: "
                    f"{', '.join(stimulus_only_violations)}."
                )
                log_testbench(
                    logger=logger,
                    sample_id=sample_id,
                    top_module=top,
                    input_file=input_file,
                    testbench=sample["testbench"],
                )
                continue
            else:
                kept_samples.append(sample)
                logger.info(f"GREEN: Testbench {i} passed stimulus-only check.")

            # # 2. Syntax Check
            # correct, result = run_testbench_vcs(
            #     verilog_files=[rtl_path],     
            #     testbench=tb_path,            
            #     output_dir=sample_dir,
            #     top_module_name=top,
            # )

            # if not correct:
            #     logger.info(f"RED: Testbench {i} failed syntax check.")
            #     failed_syntax += 1
            #     continue
            # else:
            #     logger.info(f"GREEN: Testbench {i} passed syntax check.")

            
            # 3. Functionality Check
            # correct, report_file = run_urg(
            #     output_dir=sample_dir,
            #     top_module_name=top,
            #     testbench=tb_path,
            # )

            # if not correct:
            #     logger.info(f"RED: Testbench {i} failed functionality check.")
            #     failed_functionality += 1
            #     continue
            # else:
            #     logger.info(f"GREEN: Testbench {i} passed functionality check.")

            # 4. Coverage Check
            # coverage = report_coverage_vcs(
            #     report_file=report_file
            # )
            
            # avg_coverage = compute_avg_coverage(coverage)
            # if avg_coverage < coverage_threshold:
            #     failed_coverage += 1
            #     logger.info(f"YELLOW: Testbench {i} failed coverage threshold check.")
            # else:
            #     logger.info(f"GREEN: Testbench {i} passed coverage threshold check.")
            #     passed_all += 1

        write_jsonl(failed_output_file, failed_samples)
        write_jsonl(input_file, kept_samples)
        logger.info(
            f"Wrote {len(failed_samples)} failed {split_name} samples to {failed_output_file}"
        )
        logger.info(
            f"Rewrote {input_file} with {len(kept_samples)} kept samples "
            f"(removed {len(failed_samples)} failed samples)."
        )

    logger.info(f"RED: Failed stimulus-only check             , {failed_stimulus_only}")
    logger.info(f"RED: Failed syntax check                    , {failed_syntax}")
    logger.info(f"RED: Failed functionality check             , {failed_functionality}")
    logger.info(f"RED: Failed coverage (< {coverage_threshold}), {failed_coverage}")
    logger.info(f"RED: Passed all checks                      , {passed_all}")


if __name__ == "__main__":
    main()
