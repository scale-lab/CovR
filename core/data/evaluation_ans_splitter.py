import json
import os

# splits answers based on problem source indicated in the problem text (RTLLM vs VerilogEval)

def split_jsonl_by_source(input_path):
    base = os.path.splitext(input_path)[0]
    out_verilogeval = base + "_verilogeval_2.jsonl"
    out_rtllm = base + "_rtllm_2.jsonl"

    with open(input_path, "r") as f_in, \
         open(out_verilogeval, "w") as f_veri, \
         open(out_rtllm, "w") as f_rtllm:

        for line in f_in:
            record = json.loads(line)

            problems = record["problems"]
            rtl_code = record["rtl_code"]
            top_module = record["top_module"]
            answers_flat = record["answers"]
            num_samples = record["num_samples"]

            # ---------------------------
            # 1) UNFLATTEN ANSWERS
            # ---------------------------
            answers = []
            idx = 0
            for _ in problems:
                answers.append(answers_flat[idx: idx + num_samples])
                idx += num_samples

            # Prepare output containers
            verilog_eval_entries = {"problems": [], "rtl_code": [], "top_module": [], "answers": []}
            rtllm_entries = {"problems": [], "rtl_code": [], "top_module": [], "answers": []}

            # ---------------------------
            # 2) SPLIT EXACTLY LIKE ORIGINAL
            # ---------------------------
            for i, prob in enumerate(problems):
                if prob.startswith("Prob"):
                    target = verilog_eval_entries
                else:
                    target = rtllm_entries

                target["problems"].append(prob)
                target["rtl_code"].append(rtl_code[i])
                target["top_module"].append(top_module[i])
                target["answers"].append(answers[i])   # now answers[i] is a list of length num_samples

            # ---------------------------
            # 3) WRITE OUT NON-EMPTY
            # ---------------------------
            if verilog_eval_entries["problems"]:
                v_answers_flat = [ans for group in verilog_eval_entries["answers"] for ans in group]
                verilog_eval_entries["answers"] = v_answers_flat
                verilog_eval_entries["num_samples"] = num_samples
                f_veri.write(json.dumps(verilog_eval_entries) + "\n")

            if rtllm_entries["problems"]:
                r_answers_flat = [ans for group in rtllm_entries["answers"] for ans in group]
                rtllm_entries["answers"] = r_answers_flat
                rtllm_entries["num_samples"] = num_samples
                f_rtllm.write(json.dumps(rtllm_entries) + "\n")



# Example usage:
# split_jsonl_by_problem("path/to/data.jsonl")



# ----------------------------
# Example usage:
# ----------------------------
input_file = "output/sft/qwen_coder_reasoning_20251117_210504/answers.jsonl"
split_jsonl_by_source(input_file)
