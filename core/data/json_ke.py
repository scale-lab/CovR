import json, os

inp = "/oscar/data/sreda/mabdelat/LLM4TB/output/data_generation_veritriplets/train_dataset_not_met_coverage_rtl_hygiene_improved.jsonl"
out = "/oscar/data/sreda/mabdelat/LLM4TB/output/data_generation_veritriplets/improved.jsonl"

fields = ["timestamp","dataset","sample_id","rtl_code","testbench","natural_spec","coverage_report","coverage_stats", "top_module"]

with open(inp) as f, open(out, "w") as w:
    for line in f:
        line=line.strip()
        if not line: continue
        obj=json.loads(line)
        new={k: obj.get(k) for k in fields}
        w.write(json.dumps(new)+"\n")

