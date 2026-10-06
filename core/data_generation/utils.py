import os 
import re 
import json 
import tempfile

def read_jsonl(
   file_path     
): 
    data = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:        # skip empty lines
                continue
            try:
                obj = json.loads(line)
                data.append(obj)
            except json.JSONDecodeError as e:
                print(f"[WARN] Failed to parse line: {line[:50]}... | {e}")
    return data 



def _max_sample_id(path: str, dataset: str) -> int:
    if not os.path.exists(path):
        return -1
    max_id = -1
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if obj["dataset"] != dataset: 
                    continue

                sid = obj.get("sample_id", None)
                if isinstance(sid, int) and sid > max_id:
                    max_id = sid
            except Exception:
                continue
    return max_id


def compute_avg_coverage(coverage_stats):
    avg_coverage = 0.0 
    count = 0
    for key, value in coverage_stats.items():
        if value is not None: 
            avg_coverage += value 
            count += 1

    if count != 0: 
        avg_coverage = avg_coverage / count
    
    return avg_coverage


def remove_entry_from_jsonl(file_path: str, sample_id: str) -> None:
    tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(file_path))
    os.close(tmp_fd)

    removed = False
    with open(file_path, 'r', encoding='utf-8') as src, \
         open(tmp_path,  'w', encoding='utf-8') as dst:
        for line in src:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                dst.write(line)
                continue
            if record.get("sample_id") == sample_id and not removed:
                removed = True
                continue        
            dst.write(json.dumps(record, ensure_ascii=False) + "\n")

    os.replace(tmp_path, file_path)



def _strip_testbench(response):
    if not response:
        return ""

    reasoning = str(response)

    reasoning = re.sub(
        r"<testbench\b[^>]*>.*?</testbench>",
        "",
        reasoning,
        flags=re.DOTALL | re.IGNORECASE,
    )

    def strip_fenced_testbench(match):
        language = (match.group(1) or "").strip().lower()
        code = match.group(2) or ""

        if language in {"testbench", "systemverilog", "verilog", "sv", "v"}:
            return ""

        if re.search(r"\bmodule\b", code) and re.search(r"\bendmodule\b", code):
            return ""

        return match.group(0)

    reasoning = re.sub(
        r"```\s*([A-Za-z0-9_+-]*)[^\n]*\n(.*?)```",
        strip_fenced_testbench,
        reasoning,
        flags=re.DOTALL,
    )

    reasoning = re.sub(
        r"\bmodule\b[\s\S]*?\bendmodule\b",
        "",
        reasoning,
        flags=re.IGNORECASE,
    )

    return reasoning.strip()


def extract_testbench_code(response):
    if not response:
        return ""

    content = str(response)

    fenced_pattern = re.compile(
        r"```(?:\s*(?:testbench|systemverilog|verilog|sv|v))?\s*(.*?)```",
        re.DOTALL | re.IGNORECASE,
    )
    matches = fenced_pattern.findall(content)
    if matches:
        return matches[-1].strip()

    module_pattern = re.compile(
        r"\bmodule\b[\s\S]*?\bendmodule\b",
        re.IGNORECASE,
    )
    module_match = module_pattern.search(content)
    if module_match:
        return module_match.group(0).strip()

    last_module_match = None
    for match in re.finditer(r"\bmodule\b", content, re.IGNORECASE):
        last_module_match = match

    if last_module_match:
        partial_testbench = content[last_module_match.start():].strip()
        partial_testbench = re.sub(r"```+\s*$", "", partial_testbench).strip()
        return partial_testbench

    return ""