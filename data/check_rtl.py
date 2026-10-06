import json
import re
import shutil
import subprocess
import tempfile
import sys
from pathlib import Path


def load_jsonl(json_path: str):
    data = []
    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def _defines_module(rtl_code: str, module_name: str):
    return bool(re.search(rf"\bmodule\s+{re.escape(module_name)}\b", rtl_code))


def check_rtl_record(data):
    issues = []
    warnings = []

    rtl_code = data.get("rtl_code", "")
    rtl_files = data.get("rtl_files", [])
    dut_module_name = data.get("dut_module_name") or data.get("top_module")

    if not rtl_code:
        issues.append("Missing or empty rtl_code")

    if not data.get("top_module") and dut_module_name:
        warnings.append(f"top_module is empty; using dut_module_name={dut_module_name!r}")

    if isinstance(rtl_files, list) and rtl_files:
        missing_from_rtl_code = []
        for rtl_file in rtl_files:
            name = rtl_file.get("name", "<unnamed>")
            content = (rtl_file.get("content") or "").strip()
            if content and content not in rtl_code:
                missing_from_rtl_code.append(name)

        if len(rtl_files) > 1 and missing_from_rtl_code:
            issues.append(
                "rtl_code does not contain all rtl_files content. "
                f"Missing from rtl_code: {missing_from_rtl_code}"
            )

        if dut_module_name and not _defines_module(rtl_code, dut_module_name):
            issues.append(
                f"rtl_code does not define dut_module_name/top module {dut_module_name!r}"
            )

    return issues, warnings


def write_rtl_sources(data, tmpdir: Path, sample_idx: int):
    rtl_file = tmpdir / f"dut_{sample_idx}.sv"
    rtl_file.write_text(data.get("rtl_code", ""))
    return [rtl_file]


def run_command(cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def run_vlogan_on_jsonl(json_path: str):
    records = load_jsonl(json_path)
    found_issue = False
    vlogan = shutil.which("vlogan")
    vcs = shutil.which("vcs")

    for i, data in enumerate(records):
        sample_id = data.get("sample_id", i)
        print(f"\n=== Sample {i}: {sample_id} ===")

        issues, warnings = check_rtl_record(data)
        for warning in warnings:
            print(f"⚠️  {warning}")
        if issues:
            found_issue = True
            for issue in issues:
                print(f"❌ {issue}")

        if "rtl_code" not in data:
            print(f"❌ Missing rtl_code. Available keys: {list(data.keys())}")
            found_issue = True
            continue

        if not vlogan and not vcs:
            continue

        top_module = data.get("top_module") or data.get("dut_module_name")
        if not top_module:
            print("❌ Missing both top_module and dut_module_name; cannot elaborate")
            found_issue = True
            continue

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            rtl_paths = write_rtl_sources(data, tmp_path, i)

            if vlogan:
                cmd = ["vlogan", "-full64", "-sverilog", *map(str, rtl_paths)]
                result = run_command(cmd, tmp_path)

                if result.returncode == 0:
                    print("✅ vlogan PASS")
                else:
                    found_issue = True
                    print("❌ vlogan FAIL")
                    print(result.stderr)
                    print(result.stdout)
                    continue

            if vcs:
                simv_path = tmp_path / "simv"
                cmd = ["vcs", "-full64", "-sverilog", "-top", top_module, "-o", str(simv_path)]
                result = run_command(cmd, tmp_path)

                if result.returncode == 0:
                    print("✅ elaboration PASS")
                else:
                    found_issue = True
                    print("❌ elaboration FAIL")
                    print(result.stderr)
                    print(result.stdout)

    if not vlogan:
        print("\nSkipped vlogan compilation: vlogan was not found on PATH.")
    if not vcs:
        print("\nSkipped VCS elaboration: vcs was not found on PATH.")

    return 1 if found_issue else 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_vlogan_from_json.py <input.json>")
        sys.exit(1)

    json_path = sys.argv[1]

    try:
        rc = run_vlogan_on_jsonl(json_path)
        if rc == 0:
            print("\nCompilation succeeded.")
        else:
            print(f"\nCompilation failed with return code {rc}.")
        sys.exit(rc)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
