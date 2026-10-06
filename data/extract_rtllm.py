import os
import json
from glob import glob

def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip()

def extract_rtllm(dataset_root: str, output_json: str = "rtllm_tasks.json"):
    """
    Scan RTLLM-style dataset and extract tasks into the structure:
        {
            "task_id": problem_name,
            "task_number": task_counter,
            "description": prompt,
            "ref_module": verilog_code,
            "top_module": top_module_code,
            "testbench": testbench
        }

    Assumptions:
    - Each leaf problem directory contains at least:
        design_description.txt
        testbench.v
      and typically a Verilog design file like verified_*.v
    - Leaf directories are 3 levels under dataset_root (e.g., Arithmetic/Adder/adder_16bit)
      but the code is robust and will find them anywhere under dataset_root.
    """
    tasks = []
    task_counter = 1

    for root, dirs, files in os.walk(dataset_root):
        files_set = set(files)
        # Detect a leaf problem dir by required files:
        if "design_description.txt" in files_set and "testbench.v" in files_set:
            problem_dir = root
            # Build a human-readable ID relative to dataset_root
            rel_id = os.path.relpath(problem_dir, dataset_root)
            problem_name = rel_id.replace(os.sep, "_")

            prompt_path = os.path.join(problem_dir, "design_description.txt")
            tb_path = os.path.join(problem_dir, "testbench.v")

            prompt = _read_text(prompt_path)
            testbench = _read_text(tb_path)

            # Gather candidate .v files excluding the testbench
            v_candidates = [p for p in glob(os.path.join(problem_dir, "*.v")) if os.path.basename(p) != "testbench.v"]

            # Choose ref_module: prefer verified_*.v, else first available
            ref_path = ""
            verified = [p for p in v_candidates if os.path.basename(p).lower().startswith("verified_")]
            if verified:
                ref_path = verified[0]
            elif v_candidates:
                ref_path = v_candidates[0]

            ref_module_code = _read_text(ref_path) if ref_path else ""
            ref_module_code = ref_module_code.replace("verified_", "")

            task = {
                "task_id": problem_name,            # e.g., "Arithmetic/Adder/adder_16bit"
                "task_number": task_counter,
                "description": prompt,              # from design_description.txt
                "ref_module": ref_module_code,      # from verified_*.v or first .v
                "top_module": "",      # from top*.v or *_top.v if present
                "testbench": testbench,             # from testbench.v
            }

            # Basic sanity check: require prompt and testbench at minimum
            if not prompt or not testbench:
                print(f"[WARNING] Skipping {problem_name}: empty prompt or testbench.")
                continue

            # If there is truly no design .v (unlikely), we still keep the task with empty ref_module
            tasks.append(task)
            task_counter += 1

    with open(output_json, "w", encoding="utf-8") as out_f:
        for item in tasks:
            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")


    print(f"[INFO] Extracted {len(tasks)} tasks to {output_json}")


def main():
    rtllm_path = "/oscar/data/sreda/mabdelat/RTLLM"
    extract_rtllm(dataset_root=rtllm_path, output_json="./data/rtllm_tasks.json")


if __name__ == "__main__":
    main()
