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

