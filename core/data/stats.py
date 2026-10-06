"""Count LOC, coverage, and structural RTL statistics."""

import os
import re
import json
import argparse
import tempfile
import subprocess
from statistics import median

from core.data_generation.utils import read_jsonl


# -----------------------------
# Original helpers
# -----------------------------

def _to_float_or_none(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().rstrip("%")
        if cleaned == "":
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def collect_coverage_values(entries, dims):
    coverage_by_dim = {dim: [] for dim in dims}
    for entry in entries:
        coverage_stats = entry.get("coverage_stats")
        if not isinstance(coverage_stats, dict):
            continue

        for dim in dims:
            value = _to_float_or_none(coverage_stats.get(dim))
            if value is not None:
                coverage_by_dim[dim].append(value)
    return coverage_by_dim


def print_coverage_medians(entries, dims):
    coverage_by_dim = collect_coverage_values(entries, dims)
    parts = []
    for dim in dims:
        values = coverage_by_dim[dim]
        med_value = median(values) if values else None
        med_text = f"{med_value:.2f}" if med_value is not None else "N/A"
        parts.append(f"{dim}={med_text}")
    print("  Median Coverage: {" + ", ".join(parts) + "}")


def count_samples_with_dim(entries, dim):
    count = 0
    for entry in entries:
        coverage_stats = entry.get("coverage_stats")
        if not isinstance(coverage_stats, dict):
            continue
        value = _to_float_or_none(coverage_stats.get(dim))
        if value is not None:
            count += 1
    return count


def has_coverage_dim(entry, dim):
    coverage_stats = entry.get("coverage_stats")
    if not isinstance(coverage_stats, dict):
        return False
    return _to_float_or_none(coverage_stats.get(dim)) is not None


def _sum_report_total_rows(report: str, label: str):
    if not report:
        return None

    pattern = re.compile(
        rf"(?m)^\s*{re.escape(label)}\s+"
        r"(?P<total>\d+)\s+"
        r"(?P<covered>\d+)"
        r"(?:\s+[\d.]+)?"
        r".*$"
    )
    totals = [int(m.group("total")) for m in pattern.finditer(report)]
    return sum(totals) if totals else None


def extract_coverage_report_counts(entry):
    report = entry.get("coverage_report")
    if not isinstance(report, str):
        return {}

    return {
        "branches": _sum_report_total_rows(report, "Branches"),
        "fsm_states": _sum_report_total_rows(report, "States"),
        "fsm_transitions": _sum_report_total_rows(report, "Transitions"),
    }


def count_loc(code: str) -> int:
    if not code:
        return 0

    lines = code.splitlines()
    count = 0
    for line in lines:
        stripped = line.strip()
        if stripped == "":
            continue
        if stripped.startswith("//"):
            continue
        if stripped.startswith("/*") and stripped.endswith("*/"):
            continue
        count += 1

    return count


# -----------------------------
# Structural stat helpers
# -----------------------------

def strip_comments(code: str) -> str:
    if not code:
        return ""
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    code = re.sub(r"//.*", "", code)
    return code


def safe_int_expr(expr: str):
    try:
        expr = expr.strip().replace("_", "")
        if re.fullmatch(r"[0-9+\-*/() ]+", expr):
            return int(eval(expr, {"__builtins__": {}}, {}))
    except Exception:
        pass
    return None


def width_from_range(range_text: str) -> int:
    if not range_text:
        return 1

    m = re.search(r"\[\s*([^:\]]+)\s*:\s*([^:\]]+)\s*\]", range_text)
    if not m:
        return 1

    hi = safe_int_expr(m.group(1))
    lo = safe_int_expr(m.group(2))

    if hi is None or lo is None:
        return 1

    return abs(hi - lo) + 1


def split_signal_names(names: str):
    out = []
    for x in names.split(","):
        x = x.strip()
        x = re.sub(r"=.*", "", x)
        x = re.sub(r"\[[^\]]+\]", "", x)
        x = x.strip()
        if re.match(r"^[A-Za-z_][A-Za-z0-9_$]*$", x):
            out.append(x)
    return out


def static_count_port_bits(code: str, direction: str) -> int:
    code = strip_comments(code)
    total = 0
    seen = set()

    pattern = re.compile(
        rf"\b{re.escape(direction)}\b\s+"
        r"(?:(?:wire|reg|logic|signed|unsigned)\s+)*"
        r"(?P<range>\[[^\]]+\])?\s*"
        r"(?P<names>[^;,)]+(?:,[^;,)]+)*)",
        re.MULTILINE,
    )

    for m in pattern.finditer(code):
        width = width_from_range(m.group("range"))
        for name in split_signal_names(m.group("names")):
            if name not in seen:
                seen.add(name)
                total += width

    return total


def static_count_input_bits(code: str) -> int:
    return static_count_port_bits(code, "input")


def static_count_output_bits(code: str) -> int:
    return static_count_port_bits(code, "output")


def static_count_reg_bits(code: str) -> int:
    code = strip_comments(code)
    total = 0

    pattern = re.compile(
        r"\b(?:reg|logic)\b\s+"
        r"(?:(?:signed|unsigned)\s+)*"
        r"(?P<range>\[[^\]]+\])?\s*"
        r"(?P<names>[^;]+);",
        re.MULTILINE,
    )

    for m in pattern.finditer(code):
        width = width_from_range(m.group("range"))
        names = split_signal_names(m.group("names"))
        total += width * len(names)

    return total


def static_count_branches(code: str) -> int:
    code = strip_comments(code)
    return (
        len(re.findall(r"\bif\s*\(", code))
        + len(re.findall(r"\bcase[zx]?\s*\(", code))
        + len(re.findall(r"\?", code))
    )


def static_estimate_fsm_states(code: str) -> int:
    code = strip_comments(code)
    state_names = set()

    for m in re.finditer(r"\benum\b[^{]*\{(?P<body>[^}]+)\}", code, flags=re.DOTALL):
        for item in m.group("body").split(","):
            name = re.sub(r"=.*", "", item).strip()
            if re.match(r"^[A-Za-z_][A-Za-z0-9_$]*$", name):
                state_names.add(name)

    for m in re.finditer(r"\b(?:localparam|parameter)\b(?P<body>[^;]+);", code):
        candidates = re.findall(r"\b([A-Za-z_][A-Za-z0-9_$]*)\s*=", m.group("body"))
        for c in candidates:
            if re.search(r"(state|idle|wait|run|done|start|stop|reset)", c, re.IGNORECASE):
                state_names.add(c)

    for m in re.finditer(
        r"\bcase\s*\(\s*(?:state|current_state|next_state)\s*\)(?P<body>.*?)\bendcase",
        code,
        flags=re.DOTALL | re.IGNORECASE,
    ):
        labels = re.findall(r"^\s*([A-Za-z_][A-Za-z0-9_$]*)\s*:", m.group("body"), flags=re.MULTILINE)
        for label in labels:
            if label.lower() != "default":
                state_names.add(label)

    return len(state_names)


# -----------------------------
# Yosys extraction
# -----------------------------

def run_yosys_json(rtl_code: str, timeout: int = 20):
    with tempfile.TemporaryDirectory() as td:
        rtl_path = os.path.join(td, "design.sv")
        json_path = os.path.join(td, "design.json")
        script_path = os.path.join(td, "script.ys")

        with open(rtl_path, "w", encoding="utf-8") as f:
            f.write(rtl_code)

        with open(script_path, "w", encoding="utf-8") as f:
            f.write(f"""
read_verilog -sv {rtl_path}
hierarchy -auto-top
proc
opt
memory
opt
write_json {json_path}
""")

        try:
            result = subprocess.run(
                ["yosys", "-q", script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
            )
        except FileNotFoundError:
            return None
        except subprocess.TimeoutExpired:
            return None

        if result.returncode != 0 or not os.path.exists(json_path):
            return None

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None


def run_yosys_fsm_state_count(rtl_code: str, timeout: int = 20):
    with tempfile.TemporaryDirectory() as td:
        rtl_path = os.path.join(td, "design.sv")
        script_path = os.path.join(td, "script.ys")

        with open(rtl_path, "w", encoding="utf-8") as f:
            f.write(rtl_code)

        with open(script_path, "w", encoding="utf-8") as f:
            f.write(f"""
read_verilog -sv {rtl_path}
hierarchy -auto-top
proc
opt -nosdff -nodffe
fsm_detect -ignore-self-reset
fsm_extract
fsm_export -origenc
""")

        try:
            result = subprocess.run(
                ["yosys", "-q", script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
                cwd=td,
            )
        except FileNotFoundError:
            return None
        except subprocess.TimeoutExpired:
            return None

        if result.returncode != 0:
            return None

        total_states = 0
        exported_fsms = 0
        for filename in os.listdir(td):
            if not filename.endswith(".kiss2"):
                continue

            exported_fsms += 1
            path = os.path.join(td, filename)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
            except OSError:
                return None

            m = re.search(r"^\s*\.s\s+(\d+)\s*$", text, flags=re.MULTILINE)
            if m is None:
                return None
            total_states += int(m.group(1))

        if exported_fsms == 0:
            return None

        return total_states


def yosys_input_bits(yosys_json) -> int:
    total = 0
    for module in yosys_json.get("modules", {}).values():
        for port in module.get("ports", {}).values():
            if port.get("direction") == "input":
                total += len(port.get("bits", []))
    return total


def yosys_output_bits(yosys_json) -> int:
    total = 0
    for module in yosys_json.get("modules", {}).values():
        for port in module.get("ports", {}).values():
            if port.get("direction") == "output":
                total += len(port.get("bits", []))
    return total


def yosys_reg_bits(yosys_json) -> int:
    total = 0

    reg_prefixes = (
        "$dff",
        "$adff",
        "$sdff",
        "$dffe",
        "$adffe",
        "$sdffe",
        "$aldff",
        "$ff",
    )

    for module in yosys_json.get("modules", {}).values():
        for cell in module.get("cells", {}).values():
            cell_type = cell.get("type", "")
            if cell_type.startswith(reg_prefixes):
                total += len(cell.get("connections", {}).get("Q", []))

    return total


def yosys_cell_count(yosys_json) -> int:
    total = 0
    for module in yosys_json.get("modules", {}).values():
        total += len(module.get("cells", {}))
    return total


def extract_structural_stats(rtl_code: str, use_yosys=True, yosys_timeout=20):
    branches = static_count_branches(rtl_code)
    fsm_states = None
    input_bits = static_count_input_bits(rtl_code)
    output_bits = static_count_output_bits(rtl_code)
    reg_bits = None
    cell_count = None

    used_yosys = False

    if use_yosys:
        yjson = run_yosys_json(rtl_code, timeout=yosys_timeout)
        if yjson is None:
            raise RuntimeError("Yosys JSON extraction failed")
        used_yosys = True
        input_bits = yosys_input_bits(yjson)
        output_bits = yosys_output_bits(yjson)
        reg_bits = yosys_reg_bits(yjson)
        cell_count = yosys_cell_count(yjson)

        fsm_states = run_yosys_fsm_state_count(rtl_code, timeout=yosys_timeout)

    return input_bits, output_bits, branches, fsm_states, reg_bits, cell_count, used_yosys


def triplet(values):
    return min(values), median(values), max(values)


def print_triplet(label, values):
    if not values:
        print(f"  {label:<14}: {{Min=N/A, Median=N/A, Max=N/A}}")
        return

    vmin, vmed, vmax = triplet(values)
    print(f"  {label:<14}: {{Min={vmin}, Median={vmed}, Max={vmax}}}")


def format_power_of_two(exp):
    if isinstance(exp, float) and exp.is_integer():
        exp = int(exp)
    return f"2^{exp}"


def print_stimulus_space(values):
    if not values:
        print("  Stimulus Space: {Min=N/A, Median=N/A, Max=N/A}")
        return

    vmin, vmed, vmax = triplet(values)
    print(
        "  Stimulus Space: "
        f"{{Min={format_power_of_two(vmin)}, "
        f"Median={format_power_of_two(vmed)}, "
        f"Max={format_power_of_two(vmax)}}}"
    )


def log_progress(message: str):
    print(f"[stats] {message}", flush=True)


def process_and_print(
    name,
    data,
    coverage_dims,
    use_yosys=True,
    yosys_timeout=20,
    skip_short_tb=False,
    log_every=100,
):
    rtl_counts = []
    testbench_counts = []

    input_bits = []
    output_bits = []
    io_bits = []
    branch_counts = []
    fsm_state_counts = []
    fsm_transition_counts = []
    reg_bits = []
    cell_counts = []

    yosys_success = 0
    yosys_fallback = 0
    skipped_missing_fields = 0
    skipped_short = 0
    skipped_yosys = 0

    total_entries = len(data)
    log_progress(f"{name}: starting {total_entries} entries (use_yosys={use_yosys})")

    for idx, entry in enumerate(data, start=1):
        if log_every and (idx == 1 or idx % log_every == 0 or idx == total_entries):
            log_progress(
                f"{name}: processed {idx}/{total_entries} entries "
                f"(kept={len(rtl_counts)}, yosys_success={yosys_success}, "
                f"fallback={yosys_fallback}, skipped_yosys={skipped_yosys})"
            )

        if not entry.get("testbench") or not entry.get("rtl_code"):
            skipped_missing_fields += 1
            continue

        rtl_count = count_loc(entry["rtl_code"])
        tb_count = count_loc(entry["testbench"])

        if skip_short_tb and tb_count <= 10:
            skipped_short += 1
            continue

        try:
            ib, ob, br, fs, rb, cells, used_yosys = extract_structural_stats(
                entry["rtl_code"],
                use_yosys=use_yosys,
                yosys_timeout=yosys_timeout,
            )
        except RuntimeError as exc:
            sample_id = entry.get("sample_id", "<unknown>")
            skipped_yosys += 1
            log_progress(
                f"{name}: skipping sample_id={sample_id} at entry {idx}/{total_entries}: {exc}"
            )
            continue

        rtl_counts.append(rtl_count)
        testbench_counts.append(tb_count)
        report_counts = extract_coverage_report_counts(entry)

        if ib is not None:
            input_bits.append(ib)
        if ob is not None:
            output_bits.append(ob)
        if ib is not None and ob is not None:
            io_bits.append(ib + ob)
        if has_coverage_dim(entry, "branch"):
            report_branches = report_counts.get("branches")
            branch_counts.append(report_branches if report_branches is not None else br)
        if has_coverage_dim(entry, "fsm"):
            report_fsm_states = report_counts.get("fsm_states")
            report_fsm_transitions = report_counts.get("fsm_transitions")
            fsm_states_value = report_fsm_states if report_fsm_states is not None else fs
            if fsm_states_value is not None:
                fsm_state_counts.append(fsm_states_value)
            if report_fsm_transitions is not None:
                fsm_transition_counts.append(report_fsm_transitions)
        if rb is not None:
            reg_bits.append(rb)
        if cells is not None:
            cell_counts.append(cells)

        if used_yosys:
            yosys_success += 1
        else:
            yosys_fallback += 1

    log_progress(
        f"{name}: finished scan. kept={len(rtl_counts)}, "
        f"skipped_missing_fields={skipped_missing_fields}, skipped_short={skipped_short}, "
        f"skipped_yosys={skipped_yosys}, "
        f"yosys_success={yosys_success}, fallback={yosys_fallback}"
    )

    if len(rtl_counts) == 0:
        print(f"No entries found in {name}, skipping.")
        return

    print("\nDirectory:", name)

    # Original reported information
    print_triplet("RTL LOC", rtl_counts)
    print_triplet("Testbench LOC", testbench_counts)
    print_coverage_medians(data, coverage_dims)
    fsm_count = count_samples_with_dim(data, "fsm")
    print(f"  FSM Samples   : {fsm_count}/{len(data)}")
    print(f"  Designs Count : {len(data)}")

    # New table-oriented information
    print_triplet("Input Bits", input_bits)
    print_triplet("Output Bits", output_bits)
    print_triplet("I/O Bits", io_bits)
    print_stimulus_space(input_bits)
    print_triplet("Branches", branch_counts)
    print_triplet("FSM States", fsm_state_counts)
    print_triplet("FSM Transitions", fsm_transition_counts)
    print_triplet("Reg Bits", reg_bits)
    print_triplet("Yosys Cells", cell_counts)
    print(
        f"  Yosys Extract : {yosys_success} success / {yosys_fallback} fallback "
        f"/ {skipped_yosys} skipped"
    )

    print("-" * 60)


def load_generation_dir(dir_path):
    
    if "llm4cov" in dir_path: 
        met_coverage = os.path.join(dir_path, "train_dataset_met_coverage_reasoner_v0_2_output_sdg.jsonl")
        not_met_coverage = os.path.join(dir_path, "train_dataset_not_met_coverage_reasoner_v0_2_output_sdg.jsonl")
    else:
        met_coverage = os.path.join(dir_path, "train_dataset_met_coverage.jsonl")
        not_met_coverage = os.path.join(dir_path, "train_dataset_not_met_coverage.jsonl")

    met_data = read_jsonl(met_coverage) if os.path.exists(met_coverage) else []
    not_met_data = read_jsonl(not_met_coverage) if os.path.exists(not_met_coverage) else []

    data = met_data + not_met_data
    log_progress(
        f"{dir_path}: loaded {len(met_data)} met + {len(not_met_data)} not-met = {len(data)} entries"
    )
    return data


def merge_verithought_dirs(input_dirs):
    verithought_dirs = []

    for dir_path in input_dirs:
        dirname = os.path.basename(os.path.normpath(dir_path))
        if dirname in {"data_generation_verithoughts", "data_generation_verithoughts_second_half"}:
            verithought_dirs.append(dir_path)

    if len(verithought_dirs) <= 1:
        return [(dir_path, [dir_path]) for dir_path in input_dirs]

    ordered_verithought_dirs = [
        dir_path for dir_path in input_dirs
        if os.path.basename(os.path.normpath(dir_path))
        in {"data_generation_verithoughts", "data_generation_verithoughts_second_half"}
    ]
    first_verithought_idx = min(input_dirs.index(dir_path) for dir_path in ordered_verithought_dirs)

    groups = []
    inserted_merged_group = False
    for idx, dir_path in enumerate(input_dirs):
        dirname = os.path.basename(os.path.normpath(dir_path))
        is_verithought = dirname in {
            "data_generation_verithoughts",
            "data_generation_verithoughts_second_half",
        }
        if is_verithought:
            if idx == first_verithought_idx and not inserted_merged_group:
                groups.append(("VeriThoughts", ordered_verithought_dirs))
                inserted_merged_group = True
            continue
        groups.append((dir_path, [dir_path]))

    return groups


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_dirs",
        nargs="+",
        required=False,
        default=[
            "./output/data_generation_verithoughts_second_half",
            "./output/data_generation_verithoughts",
            "./output/data_generation_pyra",
            "./output/data_generation_llm4cov",
            "./output/data_generation_verireason",
            "./output/data_generation_veritriplets",
        ],
    )

    parser.add_argument(
        "--merged_json",
        default="./output/data_merged_all/all_merged.jsonl",
    )

    parser.add_argument(
        "--eval_json",
        default="./data/eval_dataset.jsonl",
    )

    parser.add_argument(
        "--no_yosys",
        action="store_true",
    )

    parser.add_argument(
        "--yosys_timeout",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--log_every",
        type=int,
        default=100,
        help="Print progress every N entries. Set to 0 to disable per-entry progress logs.",
    )

    args = parser.parse_args()

    coverage_dims = ["toggle", "line", "cond", "branch", "fsm"]
    use_yosys = not args.no_yosys

    for group_name, dir_paths in merge_verithought_dirs(args.input_dirs):
        log_progress(f"Loading {group_name}")
        data = []
        for dir_path in dir_paths:
            data.extend(load_generation_dir(dir_path))

        process_and_print(
            name=group_name,
            data=data,
            coverage_dims=coverage_dims,
            use_yosys=use_yosys,
            yosys_timeout=args.yosys_timeout,
            skip_short_tb=False,
            log_every=args.log_every,
        )

    if os.path.exists(args.merged_json):
        log_progress(f"Loading merged JSON {args.merged_json}")
        data = read_jsonl(args.merged_json)
        log_progress(f"Loaded {len(data)} merged entries")

        process_and_print(
            name="CovR Train",
            data=data,
            coverage_dims=coverage_dims,
            use_yosys=use_yosys,
            yosys_timeout=args.yosys_timeout,
            skip_short_tb=True,
            log_every=args.log_every,
        )

    if os.path.exists(args.eval_json):
        log_progress(f"Loading eval JSON {args.eval_json}")
        data = read_jsonl(args.eval_json)
        log_progress(f"Loaded {len(data)} eval entries")

        process_and_print(
            name="CovR Test",
            data=data,
            coverage_dims=coverage_dims,
            use_yosys=use_yosys,
            yosys_timeout=args.yosys_timeout,
            skip_short_tb=False,
            log_every=args.log_every,
        )


if __name__ == "__main__":
    main()
