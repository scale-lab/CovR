#!/usr/bin/env python3
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

import argparse, csv, json, re, sys
from pathlib import Path
from typing import Any, Dict, Tuple
from datasets import load_dataset

CFG_MAP = {
    ("nonagentic", "code_generation", "no_commercial"): "cvdp_nonagentic_code_generation_no_commercial",
    ("nonagentic", "code_generation", "commercial")   : "cvdp_nonagentic_code_generation_commercial",
    ("agentic",    "code_generation", "no_commercial"): "cvdp_agentic_code_generation_no_commercial",
    ("agentic",    "code_generation", "commercial")   : "cvdp_agentic_code_generation_commercial",
    ("nonagentic", "code_comprehension", None)        : "cvdp_nonagentic_code_comprehension",
}

SAFE = re.compile(r"[^A-Za-z0-9._/-]+")

def safe_relpath(p: str) -> Path:
    """Sanitize user-provided subpaths and keep them relative."""
    p = (p or "").strip().lstrip("/")
    p = p.replace("..", "_")
    p = SAFE.sub("_", p)
    while "//" in p: p = p.replace("//", "/")
    return Path(p or "file")

def pick_id(ex: Dict[str, Any], idx: int) -> str:
    for k in ["id","sample_id","name","slug","problem_id","task_id"]:
        v = ex.get(k)
        if isinstance(v, str) and v.strip():
            return re.sub(r"[^A-Za-z0-9._-]", "_", v.strip())
    return f"item_{idx:05d}"

def coerce_map(v: Any) -> Tuple[Dict[str, Any], str]:
    """Accept dict or JSON string -> dict; otherwise {}."""
    if isinstance(v, dict): return v, "dict"
    if isinstance(v, str):
        try:
            parsed = json.loads(v)
            if isinstance(parsed, dict): return parsed, "json-string->dict"
        except Exception:
            return {}, "string(non-json)"
    return {}, type(v).__name__

def write_text(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)

def write_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def export_group(ddir: Path, subdir: str, mapping: Dict[str, Any], manifest_rows, cfg, item_id) -> int:
    """Write non-null entries in mapping under ddir/subdir/<path>"""
    target_root = ddir / subdir
    target_root.mkdir(parents=True, exist_ok=True)
    wrote = 0
    for raw_path, content in (mapping or {}).items():
        if content is None:
            continue
        outp = target_root / safe_relpath(str(raw_path))
        if isinstance(content, str):
            write_text(outp, content)          # write even empty strings
        else:
            # Non-strings go as JSON (rare)
            if outp.suffix == "":
                outp = outp.with_suffix(".json")
            write_json(outp, content)
        wrote += 1
        if manifest_rows is not None:
            rel = str(outp.relative_to(ddir))
            manifest_rows.append({"config": cfg, "item_id": item_id, "file": rel, "group": subdir})
    return wrote

def main():
    ap = argparse.ArgumentParser(description="Export CVDP items into input/, output/, harness/ with non-null files.")
    ap.add_argument("--variant", choices=["agentic","nonagentic"], required=True)
    ap.add_argument("--task", choices=["code_generation","code_comprehension"], required=True)
    ap.add_argument("--license", choices=["commercial","no_commercial"], default=None)
    ap.add_argument("-o", "--outdir", type=Path, default=Path("cvdp_fs"))
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--only-id", type=str, default=None, help="Export only the item with this exact id")
    ap.add_argument("--manifest", action="store_true")
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--include-prompt-response", action="store_true",
                    help="Also write input/prompt.txt and output/response.txt if present")
    args = ap.parse_args()

    key = (args.variant, args.task, args.license) if args.task != "code_comprehension" else (args.variant, args.task, None)
    if key not in CFG_MAP:
        sys.exit(f"Unsupported combo {key}")
    cfg = CFG_MAP[key]

    ds = load_dataset("nvidia/cvdp-benchmark-dataset", name=cfg, split="eval")

    # Determine which rows to process
    indices = range(len(ds))
    if args.only_id is not None:
        idx = next((i for i in range(len(ds)) if ds[i].get("id") == args.only_id), None)
        if idx is None:
            sys.exit(f"ID not found: {args.only_id}")
        indices = [idx]

    out_root = args.outdir / cfg
    out_root.mkdir(parents=True, exist_ok=True)

    manifest_rows = [] if args.manifest else None
    processed = 0
    total_written = 0

    for i in indices:
        row = ds[i]
        item_id = pick_id(row, i)
        ddir = out_root / item_id

        # Always create the three group folders first (even if they end up empty)
        (ddir / "input").mkdir(parents=True, exist_ok=True)
        (ddir / "output").mkdir(parents=True, exist_ok=True)
        (ddir / "harness").mkdir(parents=True, exist_ok=True)

        if args.raw:
            write_json(ddir / "_raw.json", row)

        # Optionally write prompt/response sidecars
        if args.include_prompt_response:
            inp = row.get("input") or {}
            out = row.get("output") or {}
            if isinstance(inp, dict) and isinstance(inp.get("prompt"), str):
                write_text(ddir / "input" / "prompt.txt", inp["prompt"])
                if manifest_rows is not None:
                    manifest_rows.append({"config": cfg, "item_id": item_id, "file": "input/prompt.txt", "group": "input"})
            if isinstance(out, dict) and isinstance(out.get("response"), str):
                write_text(ddir / "output" / "response.txt", out["response"])
                if manifest_rows is not None:
                    manifest_rows.append({"config": cfg, "item_id": item_id, "file": "output/response.txt", "group": "output"})

        # Write non-null maps under the three folders
        wrote = 0
        inp = row.get("input") or {}
        out = row.get("output") or {}
        harness = row.get("harness") or {}

        # input.context -> input/
        if isinstance(inp, dict) and isinstance(inp.get("context"), dict):
            wrote += export_group(ddir, "input", inp["context"], manifest_rows, cfg, item_id)

        # output.context -> output/
        if isinstance(out, dict) and isinstance(out.get("context"), dict):
            wrote += export_group(ddir, "output", out["context"], manifest_rows, cfg, item_id)

        # harness.files -> harness/
        if isinstance(harness, dict):
            files_h, _ = coerce_map(harness.get("files", {}))
            wrote += export_group(ddir, "harness", files_h, manifest_rows, cfg, item_id)

        total_written += wrote
        processed += 1
        print(f"[info] row {i} ({item_id}): wrote {wrote} files into input/, output/, harness/ -> {ddir}")

        if args.max_items is not None and processed >= args.max_items:
            break

    if manifest_rows is not None:
        mpath = out_root / "manifest.csv"
        with open(mpath, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["config","item_id","file","group"])
            w.writeheader(); w.writerows(manifest_rows)
        print(f"[CVDP] Wrote manifest: {mpath}")

    print(f"[done] processed {processed} item(s); total files written: {total_written}; root: {out_root}")

if __name__ == "__main__":
    main()
