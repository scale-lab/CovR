"""
Deduplicate the met-coverage and not-met-coverage JSONL splits together.

The script removes exact duplicate RTL entries across both files. When two
entries share the same RTL fingerprint, it keeps the one with:
  1. higher average coverage
  2. a non-empty testbench
  3. shorter testbench text
  4. met-coverage source, if still tied

Outputs are written back as separate met/not-met JSONL files according to the
source split of the kept entry.
"""

import argparse
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass

from core.data_generation.utils import compute_avg_coverage


MET_SOURCE = "met"
NOT_MET_SOURCE = "not_met"
MET_FILENAME = "train_dataset_met_coverage.jsonl"
NOT_MET_FILENAME = "train_dataset_not_met_coverage.jsonl"


@dataclass
class SourcedEntry:
    entry: dict
    source: str
    line_no: int
    ordinal: int
    datasource: str = ""
    input_path: str = ""


def get_script_logger(log_path: str):
    ensure_parent_dir(log_path)
    logger = logging.getLogger("deduplicate_coverage_splits")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def strip_verilog_comments(code: str) -> str:
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    code = re.sub(r"//.*", "", code)
    return code


def normalize_rtl(code: str) -> str:
    if not code:
        return ""

    code = code.replace("\r\n", "\n").replace("\r", "\n")
    code = strip_verilog_comments(code)
    code = re.sub(r"^\s*`timescale.*$", "", code, flags=re.MULTILINE)

    lines = [line.strip() for line in code.split("\n")]
    lines = [line for line in lines if line]

    code = "\n".join(lines)
    code = re.sub(r"\s+", " ", code).strip()
    return code


def safe_compute_avg_coverage(entry, logger=None):
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


def exact_rtl_fingerprint(entry: dict) -> str:
    rtl_code = entry.get("rtl_code", "")
    normalized = normalize_rtl(rtl_code)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def testbench_len(entry: dict) -> int:
    testbench = (entry.get("testbench", "") or "").strip()
    return len(testbench)


def choose_better_sourced(existing: SourcedEntry, new: SourcedEntry, logger):
    existing_cov = safe_compute_avg_coverage(existing.entry, logger)
    new_cov = safe_compute_avg_coverage(new.entry, logger)

    if new_cov > existing_cov:
        return new, "replaced_higher_coverage"
    if new_cov < existing_cov:
        return existing, "kept_higher_coverage"

    existing_tb_len = testbench_len(existing.entry)
    new_tb_len = testbench_len(new.entry)
    existing_has_tb = existing_tb_len > 0
    new_has_tb = new_tb_len > 0

    if new_has_tb and not existing_has_tb:
        return new, "replaced_has_testbench"
    if existing_has_tb and not new_has_tb:
        return existing, "kept_has_testbench"

    if new_tb_len < existing_tb_len:
        return new, "replaced_shorter_testbench"
    if new_tb_len > existing_tb_len:
        return existing, "kept_shorter_testbench"

    if new.source == MET_SOURCE and existing.source != MET_SOURCE:
        return new, "replaced_met_source_tie"
    if existing.source == MET_SOURCE and new.source != MET_SOURCE:
        return existing, "kept_met_source_tie"

    existing_fields = sum(v is not None and v != "" for v in existing.entry.values())
    new_fields = sum(v is not None and v != "" for v in new.entry.values())

    if new_fields > existing_fields:
        return new, "replaced_more_complete"
    if new_fields < existing_fields:
        return existing, "kept_more_complete"

    return existing, "kept_existing_tie"


def read_jsonl(
    path: str,
    source: str,
    start_ordinal: int,
    logger,
    datasource: str = "",
):
    entries = []
    total_count = 0
    missing_rtl_count = 0
    parse_error_count = 0

    if not os.path.exists(path):
        logger.error(f"RED: Input file does not exist: {path}")
        return entries, total_count, missing_rtl_count, parse_error_count

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            total_count += 1

            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                logger.error(f"RED: Failed to parse {path}:{line_no}: {e}")
                parse_error_count += 1
                continue

            rtl_code = entry.get("rtl_code", "")
            if not isinstance(rtl_code, str) or not rtl_code.strip():
                logger.warning(
                    f"YELLOW: Missing or invalid rtl_code at {path}:{line_no}; skipping."
                )
                missing_rtl_count += 1
                continue

            ordinal = start_ordinal + len(entries)
            entries.append(
                SourcedEntry(
                    entry=entry,
                    source=source,
                    line_no=line_no,
                    ordinal=ordinal,
                    datasource=datasource,
                    input_path=path,
                )
            )

    return entries, total_count, missing_rtl_count, parse_error_count


def write_jsonl(path: str, entries):
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        for sourced in entries:
            json.dump(sourced.entry, f, ensure_ascii=False)
            f.write("\n")


def write_duplicate_report(path: str, duplicate_records):
    if not path:
        return

    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        for record in duplicate_records:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")


def write_duplicate_pairs_text(path: str, duplicate_records):
    if not path:
        return

    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        for i, record in enumerate(duplicate_records, start=1):
            f.write("=" * 100 + "\n")
            f.write(f"DUPLICATE_PAIR {i}\n")
            f.write(f"FINGERPRINT: {record['fingerprint']}\n")
            f.write(f"REASON: {record['reason']}\n")
            f.write("-" * 100 + "\n")
            f.write("KEPT\n")
            if record.get("kept_datasource"):
                f.write(f"  datasource: {record['kept_datasource']}\n")
                f.write(f"  input_path: {record['kept_input_path']}\n")
            f.write(f"  source: {record['kept_source']}\n")
            f.write(f"  line_no: {record['kept_line_no']}\n")
            f.write(f"  sample_id: {record['kept_sample_id']}\n")
            f.write(f"  coverage: {record['kept_coverage']:.4f}\n")
            f.write(f"  testbench_len: {record['kept_testbench_len']}\n")
            f.write("\nKEPT RTL:\n")
            f.write(record.get("kept_rtl_code", "") or "")
            f.write("\n")
            f.write("-" * 100 + "\n")
            f.write("REMOVED\n")
            if record.get("removed_datasource"):
                f.write(f"  datasource: {record['removed_datasource']}\n")
                f.write(f"  input_path: {record['removed_input_path']}\n")
            f.write(f"  source: {record['removed_source']}\n")
            f.write(f"  line_no: {record['removed_line_no']}\n")
            f.write(f"  sample_id: {record['removed_sample_id']}\n")
            f.write(f"  coverage: {record['removed_coverage']:.4f}\n")
            f.write(f"  testbench_len: {record['removed_testbench_len']}\n")
            f.write("\nREMOVED RTL:\n")
            f.write(record.get("removed_rtl_code", "") or "")
            f.write("\n")
            f.write("\n")


def deduplicate_sourced_entries(entries, logger):
    best_by_fp = {}
    duplicate_records = []
    duplicate_count = 0
    replacement_count = 0

    for sourced in entries:
        fp = exact_rtl_fingerprint(sourced.entry)

        if fp not in best_by_fp:
            best_by_fp[fp] = sourced
            continue

        duplicate_count += 1
        previous = best_by_fp[fp]
        chosen, reason = choose_better_sourced(previous, sourced, logger)

        if chosen is not previous:
            replacement_count += 1
            best_by_fp[fp] = chosen

        removed = sourced if chosen is previous else previous
        duplicate_records.append(
            {
                "fingerprint": fp,
                "reason": reason,
                "kept_datasource": chosen.datasource,
                "kept_input_path": chosen.input_path,
                "kept_source": chosen.source,
                "kept_line_no": chosen.line_no,
                "kept_sample_id": chosen.entry.get("sample_id"),
                "kept_coverage": safe_compute_avg_coverage(chosen.entry, logger),
                "kept_testbench_len": testbench_len(chosen.entry),
                "kept_rtl_code": chosen.entry.get("rtl_code", ""),
                "removed_datasource": removed.datasource,
                "removed_input_path": removed.input_path,
                "removed_source": removed.source,
                "removed_line_no": removed.line_no,
                "removed_sample_id": removed.entry.get("sample_id"),
                "removed_coverage": safe_compute_avg_coverage(removed.entry, logger),
                "removed_testbench_len": testbench_len(removed.entry),
                "removed_rtl_code": removed.entry.get("rtl_code", ""),
            }
        )
        logger.info(
            f"YELLOW: Duplicate fp={fp[:12]}... reason={reason}; "
            f"kept datasource={chosen.datasource or 'single'} source={chosen.source} "
            f"sample_id={chosen.entry.get('sample_id')} "
            f"coverage={safe_compute_avg_coverage(chosen.entry, logger):.4f} "
            f"testbench_len={testbench_len(chosen.entry)}; "
            f"removed datasource={removed.datasource or 'single'} source={removed.source} "
            f"sample_id={removed.entry.get('sample_id')} "
            f"coverage={safe_compute_avg_coverage(removed.entry, logger):.4f} "
            f"testbench_len={testbench_len(removed.entry)}"
        )

    kept_entries = sorted(best_by_fp.values(), key=lambda item: item.ordinal)
    return kept_entries, duplicate_records, duplicate_count, replacement_count


def deduplicate_coverage_splits(
    met_file: str,
    not_met_file: str,
    met_out: str,
    not_met_out: str,
    logger,
    combined_out: str = "",
    duplicates_out: str = "",
    duplicate_pairs_txt: str = "",
):
    met_entries, met_total, met_missing, met_parse_errors = read_jsonl(
        met_file, MET_SOURCE, 0, logger
    )
    not_met_entries, not_met_total, not_met_missing, not_met_parse_errors = read_jsonl(
        not_met_file, NOT_MET_SOURCE, len(met_entries), logger
    )
    entries = met_entries + not_met_entries

    logger.info(
        f"GREEN: Loaded {len(met_entries)} valid met entries and "
        f"{len(not_met_entries)} valid not-met entries"
    )

    kept_entries, duplicate_records, duplicate_count, replacement_count = (
        deduplicate_sourced_entries(entries, logger)
    )
    met_kept = [item for item in kept_entries if item.source == MET_SOURCE]
    not_met_kept = [item for item in kept_entries if item.source == NOT_MET_SOURCE]

    write_jsonl(met_out, met_kept)
    write_jsonl(not_met_out, not_met_kept)
    if combined_out:
        write_jsonl(combined_out, kept_entries)
    write_duplicate_report(duplicates_out, duplicate_records)
    write_duplicate_pairs_text(duplicate_pairs_txt, duplicate_records)

    logger.info(f"GREEN: Processed {met_total + not_met_total} total input entries")
    logger.info(f"YELLOW: Missing rtl_code skipped: {met_missing + not_met_missing}")
    logger.info(f"RED: JSON parse errors: {met_parse_errors + not_met_parse_errors}")
    logger.info(
        f"GREEN: Removed {duplicate_count} duplicates across both splits; "
        f"replacements={replacement_count}"
    )
    logger.info(f"GREEN: Wrote {len(met_kept)} met entries to {met_out}")
    logger.info(f"GREEN: Wrote {len(not_met_kept)} not-met entries to {not_met_out}")
    if combined_out:
        logger.info(f"GREEN: Wrote {len(kept_entries)} combined entries to {combined_out}")
    if duplicates_out:
        logger.info(
            f"GREEN: Wrote {len(duplicate_records)} duplicate records to {duplicates_out}"
        )
    if duplicate_pairs_txt:
        logger.info(
            f"GREEN: Wrote {len(duplicate_records)} duplicate pairs to "
            f"{duplicate_pairs_txt}"
        )


def datasource_name(datasource_dir: str) -> str:
    return os.path.basename(os.path.normpath(datasource_dir))


def discover_datasource_dirs(root: str):
    if not os.path.isdir(root):
        return []

    datasource_dirs = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        met_path = os.path.join(path, MET_FILENAME)
        not_met_path = os.path.join(path, NOT_MET_FILENAME)
        if os.path.exists(met_path) and os.path.exists(not_met_path):
            datasource_dirs.append(path)
    return datasource_dirs


def output_path_for_split(datasource_dir: str, source: str, suffix: str):
    if source == MET_SOURCE:
        filename = f"train_dataset_met_coverage{suffix}.jsonl"
    elif source == NOT_MET_SOURCE:
        filename = f"train_dataset_not_met_coverage{suffix}.jsonl"
    else:
        raise ValueError(f"Unknown source split: {source}")
    return os.path.join(datasource_dir, filename)


def deduplicate_across_datasources(
    datasource_dirs,
    logger,
    output_suffix="_global_deduplicated",
    combined_out="",
    duplicates_out="output/deduplication/across_datasources_duplicate_records.jsonl",
    duplicate_pairs_txt="output/deduplication/across_datasources_duplicate_pairs.txt",
):
    entries = []
    total_count = 0
    missing_rtl_count = 0
    parse_error_count = 0
    loaded_by_datasource = {}

    for datasource_dir in datasource_dirs:
        name = datasource_name(datasource_dir)
        met_path = os.path.join(datasource_dir, MET_FILENAME)
        not_met_path = os.path.join(datasource_dir, NOT_MET_FILENAME)

        met_entries, met_total, met_missing, met_parse_errors = read_jsonl(
            met_path, MET_SOURCE, len(entries), logger, datasource=name
        )
        entries.extend(met_entries)

        not_met_entries, not_met_total, not_met_missing, not_met_parse_errors = read_jsonl(
            not_met_path, NOT_MET_SOURCE, len(entries), logger, datasource=name
        )
        entries.extend(not_met_entries)

        total_count += met_total + not_met_total
        missing_rtl_count += met_missing + not_met_missing
        parse_error_count += met_parse_errors + not_met_parse_errors
        loaded_by_datasource[name] = {
            MET_SOURCE: len(met_entries),
            NOT_MET_SOURCE: len(not_met_entries),
        }

        logger.info(
            f"GREEN: Loaded datasource={name}: "
            f"{len(met_entries)} met, {len(not_met_entries)} not-met"
        )

    kept_entries, duplicate_records, duplicate_count, replacement_count = (
        deduplicate_sourced_entries(entries, logger)
    )

    kept_by_datasource_split = {}
    for item in kept_entries:
        kept_by_datasource_split.setdefault((item.datasource, item.source), []).append(item)

    for datasource_dir in datasource_dirs:
        name = datasource_name(datasource_dir)
        for source in (MET_SOURCE, NOT_MET_SOURCE):
            out_path = output_path_for_split(datasource_dir, source, output_suffix)
            source_entries = kept_by_datasource_split.get((name, source), [])
            write_jsonl(out_path, source_entries)
            logger.info(
                f"GREEN: Wrote datasource={name} source={source} "
                f"entries={len(source_entries)} to {out_path}"
            )

    if combined_out:
        write_jsonl(combined_out, kept_entries)
    write_duplicate_report(duplicates_out, duplicate_records)
    write_duplicate_pairs_text(duplicate_pairs_txt, duplicate_records)

    logger.info(f"GREEN: Processed {total_count} total input entries")
    logger.info(f"YELLOW: Missing rtl_code skipped: {missing_rtl_count}")
    logger.info(f"RED: JSON parse errors: {parse_error_count}")
    logger.info(
        f"GREEN: Removed {duplicate_count} duplicates across datasources; "
        f"replacements={replacement_count}"
    )
    logger.info(f"GREEN: Kept {len(kept_entries)} globally unique entries")
    logger.info(f"GREEN: Loaded by datasource: {loaded_by_datasource}")
    if combined_out:
        logger.info(f"GREEN: Wrote combined global deduplicated data to {combined_out}")
    if duplicates_out:
        logger.info(
            f"GREEN: Wrote {len(duplicate_records)} duplicate records to {duplicates_out}"
        )
    if duplicate_pairs_txt:
        logger.info(
            f"GREEN: Wrote {len(duplicate_records)} duplicate pairs to "
            f"{duplicate_pairs_txt}"
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Deduplicate met and not-met coverage JSONL splits together by exact RTL, "
            "keeping higher coverage and then shorter testbenches."
        )
    )
    parser.add_argument(
        "--met",
        type=str,
        default="output/data_generation_veritriplets/train_dataset_met_coverage.jsonl",
        help="Path to the met-coverage JSONL file",
    )
    parser.add_argument(
        "--not-met",
        type=str,
        default="output/data_generation_veritriplets/train_dataset_not_met_coverage.jsonl",
        help="Path to the not-met-coverage JSONL file",
    )
    parser.add_argument(
        "--met-out",
        type=str,
        default=(
            "output/data_generation_veritriplets/"
            "train_dataset_met_coverage_deduplicated.jsonl"
        ),
        help="Path to write deduplicated met-coverage entries",
    )
    parser.add_argument(
        "--not-met-out",
        type=str,
        default=(
            "output/data_generation_veritriplets/"
            "train_dataset_not_met_coverage_deduplicated.jsonl"
        ),
        help="Path to write deduplicated not-met-coverage entries",
    )
    parser.add_argument(
        "--combined-out",
        type=str,
        default="",
        help="Optional path to write all kept entries in one JSONL file",
    )
    parser.add_argument(
        "--duplicates-out",
        type=str,
        default="output/deduplication/met_not_met_duplicate_records.jsonl",
        help="Optional JSONL report describing each removed duplicate",
    )
    parser.add_argument(
        "--duplicate-pairs-txt",
        type=str,
        default="output/deduplication/met_not_met_duplicate_pairs.txt",
        help="Optional text report describing each kept/removed duplicate pair",
    )
    parser.add_argument(
        "--datasource-dirs",
        nargs="+",
        default=[],
        help=(
            "Deduplicate all met/not-met files from these datasource directories "
            "together, then write per-datasource outputs."
        ),
    )
    parser.add_argument(
        "--discover-datasources-root",
        type=str,
        default="",
        help=(
            "Directory whose child folders should be scanned for "
            "train_dataset_met_coverage.jsonl and train_dataset_not_met_coverage.jsonl."
        ),
    )
    parser.add_argument(
        "--global-output-suffix",
        type=str,
        default="_global_deduplicated",
        help=(
            "Suffix used for per-datasource outputs in multi-datasource mode. "
            "For example, _global_deduplicated writes "
            "train_dataset_met_coverage_global_deduplicated.jsonl."
        ),
    )

    args = parser.parse_args()
    logger = get_script_logger("output/deduplication/deduplicate_coverage_splits.log")

    datasource_dirs = list(args.datasource_dirs)
    if args.discover_datasources_root:
        datasource_dirs.extend(discover_datasource_dirs(args.discover_datasources_root))

    if datasource_dirs:
        seen_dirs = set()
        unique_datasource_dirs = []
        for datasource_dir in datasource_dirs:
            normalized_dir = os.path.normpath(datasource_dir)
            if normalized_dir in seen_dirs:
                continue
            seen_dirs.add(normalized_dir)
            unique_datasource_dirs.append(normalized_dir)

        deduplicate_across_datasources(
            datasource_dirs=unique_datasource_dirs,
            output_suffix=args.global_output_suffix,
            combined_out=args.combined_out,
            duplicates_out=args.duplicates_out,
            duplicate_pairs_txt=args.duplicate_pairs_txt,
            logger=logger,
        )
        return

    deduplicate_coverage_splits(
        met_file=args.met,
        not_met_file=args.not_met,
        met_out=args.met_out,
        not_met_out=args.not_met_out,
        combined_out=args.combined_out,
        duplicates_out=args.duplicates_out,
        duplicate_pairs_txt=args.duplicate_pairs_txt,
        logger=logger,
    )


if __name__ == "__main__":
    main()
