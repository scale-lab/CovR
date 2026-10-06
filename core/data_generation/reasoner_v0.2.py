"""Generates reasoning traces for Testbench Generation."""
import argparse
import json
import os
import re
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import BadRequestError, OpenAI
from langchain_core.prompts import ChatPromptTemplate

from core.data_generation.config.config import FlowConfigs, LMConfigs
from core.data_generation.utils import read_jsonl

load_dotenv()


# DEFAULT_INPUT = "/oscar/data/sreda/mnouh/ScaleLab/LLM4TB/output/data_generation_verithoughts_second_half/train_dataset_met_coverage.jsonl"
# DEFAULT_OUTPUT = "/oscar/data/sreda/mnouh/ScaleLab/LLM4TB/output/data_generation_verithoughts_second_half/train_dataset_met_coverage_reasoner_v0_2_output.jsonl"
DEFAULT_MODEL = "gpt-4o-mini-2024-07-18"
DEFAULT_NUM_TRACES = 3
DEFAULT_NUM_ITERS = 1
DEFAULT_PROMPT_VARIANT = "structured"
DEFAULT_REASONER_BACKEND = "deepseek"
DEFAULT_VERIFIER_BACKEND = "gpt"
DEFAULT_REASONER_THINKING_MODE = "disabled"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY is None:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if DEEPSEEK_API_KEY is None:
    raise ValueError("DEEPSEEK_API_KEY not found in environment variables.")

_REASONER_GENERAL_SYS_PROMPT = """
Your task is to INFER the internal reasoning process of a verification engineer.

Given the RTL design, the final testbench, reconstruct the step-by-step verification reasoning that would naturally lead to writing this testbench.

IMPORTANT:
- The reasoning must be written as if it precedes the testbench.
- Do NOT describe the testbench at a surface level.
- Instead, explain the intent behind each verification decision:
  why specific inputs are chosen, why certain behaviors are tested,
  how they relate to the RTL logic, and how they help exercise important coverage-relevant behaviors.
- Every stimulus described must be explicitly linked to a concrete RTL condition (e.g., specific if/else branch, case condition, or state transition).
- Do NOT introduce behaviors, checks, branches, states, or coverage claims that are not supported by the RTL, testbench, or coverage evidence.
- When discussing a test scenario, explicitly connect it to the RTL behavior it is intended to exercise, such as a branch condition, case item, reset path, state transition, corner case, or output condition.
- Emphasize why the selected stimuli and checks are useful not only for functional validation, but also for covering distinct behaviors in the design.
- If important RTL behaviors remain uncovered, mention them briefly as limitations of this first-pass reasoning, but do not invent tests that are not present.
- Every stimulus described must be explicitly linked to a concrete RTL condition (e.g., specific if/else branch, case condition, or state transition).
- Do NOT narrate the testbench (e.g., "the testbench applies..."). Instead, express reasoning as verification decisions.
- Coverage discussion must refer to distinct RTL behaviors, not generic coverage terms.
- Each statement should reflect a verification decision, not a description of actions.

This is NOT an explanation task.
This is a reconstruction of the reasoning process that leads to the testbench.

The reasoning should reflect a coverage-aware verification mindset:
identify how a verification engineer would choose inputs and checks to exercise meaningful and distinct RTL behaviors, not just to observe outputs.

Use concise, technical, verification-oriented language.
Prefer short numbered steps or compact paragraphs.

For each major test stimulus, explain both:
(1) the functional intent, and
(2) the distinct RTL behavior or coverage target it is meant to exercise.
"""

_REASONER_STRUCTURED_FORMAT_PROMPT = """
Use a concise structured format.

Focus on the most important reasoning only. Do not add routine DUT-instantiation discussion unless it is unusual or materially relevant to the verification intent.

Structure each reasoning trace using only the relevant sections below:

1) Verification Goal
   What functionality or behavior needs to be validated.

2) Stimulus Design
   Why the important inputs, sequences, or scenarios are chosen and which RTL behaviors they are intended to exercise.
   Highlight when a stimulus is targeting something specific such as a branch, case split, reset path, FSM transition, or boundary condition.

3) Timing and Execution
   Include only timing, reset, or sampling details that materially affect behavior.

4) Coverage Intent and Corner Cases
   Identify the distinct RTL behaviors and edge cases being targeted, and briefly note important uncovered behaviors if present.

Keep each section concise and grounded in the RTL and testbench.
Omit any section that is not relevant instead of adding generic content.
"""

_REASONER_USER_PROMPT = """
Natural language specification:
{natural_spec}

RTL code:
```verilog
{verilog}
```

Testbench code:
```verilog
{testbench}
```
"""

_VERIFIER_SYS_PROMPT = """
You are an expert verification reviewer acting as an LLM judge.

You will score each candidate reasoning traces for how well they explain the provided testbench relative to the natural language spec, RTL, and coverage report.

Return exactly a JSON object with this schema:
{{
  "ranking": [
    {{
      "id": "trace_1",
      "score": 9,
      "reason": "why this trace received this rank and score, explicitly grounded in the testbench and coverage report"
    }}
  ]
}}

Judging criteria:
- correctness about the RTL and testbench
- specificity of connection to actual testbench behavior
- use of coverage report evidence
- usefulness for testbench-generation reasoning data
- clarity and non-redundancy
- penalize hallucinations or claims not supported by the RTL, testbench, or coverage report

Do not return markdown fences.
"""

_VERIFIER_USER_PROMPT = """
Natural language specification:
{natural_spec}

RTL code:
```verilog
{verilog}
```

Testbench code:
```verilog
{testbench}
```

Coverage report:
{coverage_report}

Candidate reasoning traces:
{reasoning_traces}
"""


def _response_text(response: Any) -> str:
    return response.content if hasattr(response, "content") else str(response)


def _log(message: str) -> None:
    print(message, flush=True)


def _extract_json_block(text: str) -> str:
    text = text.strip()
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced[-1].strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end >= start:
        return text[start : end + 1]
    return text


def _safe_json_loads(text: str, default: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return json.loads(_extract_json_block(text))
    except Exception:
        return default


def _parse_reasoning_trace_text(text: str) -> Dict[str, str]:
    text = (text or "").strip()
    if not text:
        return {"id": "", "reasoning": ""}

    if text.startswith("{") and text.endswith("}"):
        try:
            parsed = json.loads(_extract_json_block(text))
            if isinstance(parsed, dict):
                reasoning = parsed.get("reasoning") or parsed.get("reason") or parsed.get("trace") or ""
                trace_id = parsed.get("id") or ""
                return {"id": str(trace_id), "reasoning": str(reasoning)}
        except Exception:
            pass

    return {"id": "", "reasoning": text}


def _build_client_and_model(backend: str, role: str, thinking_mode: str = DEFAULT_REASONER_THINKING_MODE):
    if backend == "gpt":
        return OpenAI(api_key=OPENAI_API_KEY), "gpt-4o-mini"
    if backend == "deepseek":
        model = "deepseek-reasoner" if role == "reasoner" else "deepseek-chat"
        return OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com"), model
    raise ValueError(f"Unsupported backend for {role}: {backend}")


def _normalize_reasoning_traces(payload: Dict[str, Any], fallback_text: str) -> List[Dict[str, str]]:
    traces = payload.get("reasoning_traces", [])
    if isinstance(traces, list) and traces:
        normalized = []
        for idx, trace in enumerate(traces, start=1):
            if not isinstance(trace, dict):
                trace = {"reasoning": str(trace)}
            normalized.append(
                {
                    "id": str(trace.get("id") or f"trace_{idx}"),
                    "reasoning": str(trace.get("reasoning") or ""),
                }
            )
        return normalized

    return [
        {
            "id": "trace_1",
            "reasoning": fallback_text.strip(),
        }
    ]


def _normalize_verifier_feedback(payload: Dict[str, Any], traces: List[Dict[str, str]], fallback_text: str) -> Dict[str, Any]:
    ranking = payload.get("ranking", [])
    if not isinstance(ranking, list):
        ranking = []

    if not ranking and traces:
        ranking = [
            {
                "id": traces[0]["id"],
                "score": 0,
                "reason": "",
            }
        ]

    return {
        "ranking": ranking,
        "raw_feedback": fallback_text.strip(),
    }


class Reasoner:
    def __init__(
        self,
        config=None,
        num_traces: int = DEFAULT_NUM_TRACES,
        prompt_variant: str = DEFAULT_PROMPT_VARIANT,
        backend: str = DEFAULT_REASONER_BACKEND,
        thinking_mode: str = DEFAULT_REASONER_THINKING_MODE,
    ):
        self.config = config
        self.num_traces = num_traces
        self.prompt_variant = prompt_variant
        self.backend = backend
        self.thinking_mode = thinking_mode
        self.client, self.model = _build_client_and_model(backend, "reasoner", thinking_mode)
        format_prompt = self._get_format_prompt(prompt_variant)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _REASONER_GENERAL_SYS_PROMPT + "\n\n" + format_prompt),
                ("user", _REASONER_USER_PROMPT),
            ]
        )

    def _get_format_prompt(self, prompt_variant: str) -> str:
        if prompt_variant == "unstructured":
            return ""
        return _REASONER_STRUCTURED_FORMAT_PROMPT

    def forward(self, natural_spec, verilog, testbench, coverage_report):
        _log("[Reasoner] Formatting prompt")
        messages = self.prompt.format_messages(
            natural_spec=natural_spec or "",
            verilog=verilog or "",
            testbench=testbench or "",
        )

        _log(
            f"[Reasoner] Requesting {self.num_traces} traces from {self.model} "
            f"with prompt_variant={self.prompt_variant} backend={self.backend} "
            f"thinking_mode={self.thinking_mode}"
        )
        all_reasoning_traces = []

        request_messages = [
            {"role": "system", "content": messages[0].content},
            {"role": "user", "content": messages[1].content},
        ]

        if self.backend == "gpt":
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=request_messages,
                n=self.num_traces,
            )

            for i, choice in enumerate(completion.choices):
                text = choice.message.content
                parsed = _parse_reasoning_trace_text(text)
                trace = {
                    "id": parsed["id"] or f"trace_{i+1}",
                    "reasoning": parsed["reasoning"],
                }
                all_reasoning_traces.append(trace)
        else:
            for i in range(self.num_traces):
                _log(f"[Reasoner] DeepSeek call {i + 1}/{self.num_traces}")
                request_kwargs = {
                    "model": self.model,
                    "messages": request_messages,
                }
                if self.thinking_mode == "enabled":
                    request_kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
                completion = self.client.chat.completions.create(**request_kwargs)
                if completion and completion.choices and len(completion.choices) > 0:
                    text = completion.choices[0].message.content 
                parsed = _parse_reasoning_trace_text(text)
                trace = {
                    "id": parsed["id"] or f"trace_{i+1}",
                    "reasoning": parsed["reasoning"],
                }
                all_reasoning_traces.append(trace)

        _log(f"[Reasoner] Parsed {len(all_reasoning_traces)} reasoning traces")        
        empty_reasoning_count = sum(1 for trace in all_reasoning_traces if not trace.get("reasoning", "").strip())
        if not all_reasoning_traces:
            _log("[Reasoner] WARNING: no reasoning traces were generated")
        elif empty_reasoning_count == len(all_reasoning_traces):
            _log("[Reasoner] WARNING: all generated reasoning traces are empty")
        elif empty_reasoning_count > 0:
            _log(f"[Reasoner] WARNING: {empty_reasoning_count}/{len(all_reasoning_traces)} generated reasoning traces have empty reasoning")
        for trace in all_reasoning_traces:
            _log(f"[Reasoner] {trace.get('id')}:")
            _log(trace.get("reasoning", ""))

        return all_reasoning_traces


class Verifier:
    def __init__(self, config=None, backend: str = DEFAULT_VERIFIER_BACKEND):
        self.config = config
        self.backend = backend
        self.client, self.model = _build_client_and_model(backend, "verifier")
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _VERIFIER_SYS_PROMPT),
                ("user", _VERIFIER_USER_PROMPT),
            ]
        )

    def forward(self, natural_spec, verilog, testbench, coverage_report, reasoning_traces):
        traces_json = json.dumps(reasoning_traces, indent=2)
        _log(f"[Verifier] Evaluating {len(reasoning_traces)} reasoning traces with {self.model}")
        messages = self.prompt.format_messages(
            natural_spec=natural_spec or "",
            verilog=verilog or "",
            testbench=testbench or "",
            coverage_report=coverage_report or "",
            reasoning_traces=traces_json,
        )
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": messages[0].content},
                    {"role": "user", "content": messages[1].content},
                ],
                stream=False,
                # extra_body={"thinking": {"type": "enabled"}}
            )
        except BadRequestError as exc:
            error_text = str(exc)
            if "context_length_exceeded" not in error_text:
                raise
            _log(f"[Verifier] Skipping verifier: context too long for {self.model}: {error_text}")
            return _normalize_verifier_feedback(
                {},
                reasoning_traces,
                f"Verifier skipped because prompt exceeded context length: {error_text}",
            )
        text = completion.choices[0].message.content or ""
        # _log("[Verifier] Raw model output:")
        # _log(text)
        payload = _safe_json_loads(text, default={})
        verification_feedback = _normalize_verifier_feedback(payload, reasoning_traces, text)
        ranking = verification_feedback.get("ranking", [])
        _log(f"[Verifier] Parsed {len(ranking)} scores")
        for item in ranking:
            _log(
                f"[Verifier] id={item.get('id')} score={item.get('score')} reason={item.get('reason', '')}"
            )
        return verification_feedback


class Runner:
    def __init__(
        self,
        config=None,
        num_traces: int = DEFAULT_NUM_TRACES,
        num_iters: int = DEFAULT_NUM_ITERS,
        prompt_variant: str = DEFAULT_PROMPT_VARIANT,
        reasoner_backend: str = DEFAULT_REASONER_BACKEND,
        verifier_backend: str = DEFAULT_VERIFIER_BACKEND,
        reasoner_thinking_mode: str = DEFAULT_REASONER_THINKING_MODE,
    ):
        self.config = config
        self.num_iters = num_iters
        self.reasoner = Reasoner(
            config,
            num_traces=num_traces,
            prompt_variant=prompt_variant,
            backend=reasoner_backend,
            thinking_mode=reasoner_thinking_mode,
        )
        self.verifier = Verifier(config, backend=verifier_backend)

    def forward(self, natural_spec, verilog, testbench, coverage_report):
        reasoning_traces = []
        verification_feedback = {}

        for iteration in range(self.num_iters):
            # _log(f"[Runner] Starting iteration {iteration + 1}/{self.num_iters}")
            reasoning_traces = self.reasoner.forward(natural_spec, verilog, testbench, coverage_report)
            verification_feedback = self.verifier.forward(
                natural_spec,
                verilog,
                testbench,
                coverage_report,
                reasoning_traces,
            )
            # _log(f"[Runner] Finished iteration {iteration + 1}/{self.num_iters}")

        return reasoning_traces, verification_feedback


def _coerce_rtl_code(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    if "```" not in value and "<answer>" not in value:
        return value

    patterns = [
        r"```verilog\s*([\s\S]*?)\s*```",
        r"```\s*([\s\S]*?)\s*```",
        r"<answer>\s*```verilog\s*([\s\S]*?)\s*```\s*</answer>",
        r"<answer>\s*```\s*([\s\S]*?)\s*```\s*</answer>",
    ]
    for pattern in patterns:
        match = re.search(pattern, value, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return value


def _build_output_entry(entry: Dict[str, Any], reasoning_traces: List[Dict[str, str]], verification_feedback: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(entry)
    best_reasoning_trace = None
    rankings = verification_feedback.get("ranking", [])
    best_ranked = None
    if rankings:
        best_ranked = max(rankings, key=lambda item: item.get("score", float("-inf")))
        if best_ranked and best_ranked.get("id"):
            best_reasoning_trace = next(
                (trace for trace in reasoning_traces if trace.get("id") == best_ranked.get("id")),
                None,
            )
            if best_reasoning_trace is None:
                _log(f"[Output] WARNING: best ranked trace id={best_ranked.get('id')} not found in reasoning traces; defaulting to first trace")
        else:
            _log("[Output] WARNING: best ranked trace has no id; defaulting to first reasoning trace")
    else:
        _log("[Output] WARNING: verification ranking missing; defaulting to first reasoning trace as best_reasoning_trace")

    if best_reasoning_trace is None:
        if reasoning_traces:
            best_reasoning_trace = reasoning_traces[0]
            _log(f"[Output] INFO: selected first reasoning trace (id={best_reasoning_trace.get('id', 'unknown')}) as fallback best_reasoning_trace")
        else:
            _log("[Output] WARNING: no reasoning traces available to select as best_reasoning_trace")

    if isinstance(best_reasoning_trace, dict):
        best_reasoning_text = best_reasoning_trace.get("reasoning", "")
    else:
        best_reasoning_text = ""

    result["reasoning_traces"] = reasoning_traces
    result["verification_feedback"] = verification_feedback
    result["best_reasoning_trace"] = best_reasoning_text
    result["tb_reasoning_gen_output_v0.2"] = reasoning_traces
    if isinstance(best_reasoning_trace, dict) and best_reasoning_trace.get("id"):
        _log(f"[Output] Best reasoning trace selected: {best_reasoning_trace.get('id')}")
    return result


def _cleanup_reasoning_text(value: Any) -> str:
    if isinstance(value, dict):
        reasoning = value.get("reasoning") or value.get("reason") or ""
        return str(reasoning)
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned.startswith("{") and cleaned.endswith("}"):
            try:
                parsed = json.loads(_extract_json_block(cleaned))
                if isinstance(parsed, dict):
                    return str(parsed.get("reasoning") or parsed.get("reason") or "")
            except Exception:
                pass
        return cleaned
    return ""


def _clean_output_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = dict(entry)
    if "best_reasoning_trace" in cleaned:
        cleaned["best_reasoning_trace"] = _cleanup_reasoning_text(cleaned["best_reasoning_trace"])
    if "tb_reasoning_gen_output_v0.2" in cleaned:
        traces = cleaned["tb_reasoning_gen_output_v0.2"]
        if isinstance(traces, list):
            cleaned["tb_reasoning_gen_output_v0.2"] = [
                _cleanup_reasoning_text(trace) if not isinstance(trace, str) else _cleanup_reasoning_text(trace)
                for trace in traces
            ]
    return cleaned


def _clean_output_file(path: str) -> None:
    entries = read_jsonl(path)
    cleaned_entries = [_clean_output_entry(entry) for entry in entries]
    _save_jsonl(cleaned_entries, path)


def _save_jsonl(entries: List[Dict[str, Any]], output_path: str) -> None:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    _log(f"[Output] Wrote {len(entries)} entries to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to the input JSONL dataset.")
    parser.add_argument("--output", type=str, required=True, help="Path to the output JSONL file.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model name used for reasoner and verifier.")
    parser.add_argument("--num-traces", type=int, default=DEFAULT_NUM_TRACES, help="Number of candidate reasoning traces to generate.")
    parser.add_argument("--num-iters", type=int, default=DEFAULT_NUM_ITERS, help="Runner loop count. Defaults to 1.")
    parser.add_argument("--prompt-variant", type=str, choices=["structured", "unstructured"], default=DEFAULT_PROMPT_VARIANT, help="Reasoner prompt variant to use.")
    parser.add_argument("--reasoner-backend", type=str, choices=["gpt", "deepseek"], default=DEFAULT_REASONER_BACKEND, help="Backend to use for the reasoner.")
    parser.add_argument("--reasoner-thinking-mode", type=str, choices=["enabled", "disabled"], default=DEFAULT_REASONER_THINKING_MODE, help="Controls whether to pass DeepSeek's thinking flag in extra_body during trace generation. The DeepSeek reasoner backend still uses deepseek-reasoner.")
    parser.add_argument("--verifier-backend", type=str, choices=["gpt", "deepseek"], default=DEFAULT_VERIFIER_BACKEND, help="Backend to use for the verifier.")
    parser.add_argument("--save-every", type=int, default=0, help="Save interim output every N entries when processing a dataset.")
    parser.add_argument("--clean-jsonl", action="store_true", help="Clean reasoning wrappers in an existing JSONL output file and exit.")
    args = parser.parse_args()

    if args.clean_jsonl:
        if not os.path.exists(args.output):
            _log(f"[Main] Output file {args.output} does not exist for cleaning.")
            return
        _log(f"[Main] Cleaning existing JSONL output file {args.output}")
        _clean_output_file(args.output)
        return

    _log(f"[Main] Loading dataset from {args.input}")
    data = read_jsonl(args.input)
    if not data:
        _log(f"[Main] No data found in {args.input}")
        return
    _log(f"[Main] Loaded {len(data)} entries")

    # Load existing output to resume
    start_idx = 0
    outputs = []
    if os.path.exists(args.output):
        _log(f"[Main] Loading existing output from {args.output} to resume")
        existing_outputs = read_jsonl(args.output)
        processed_ids = {entry.get("sample_id", entry.get("id")) for entry in existing_outputs}
        outputs = existing_outputs
        for idx, entry in enumerate(data):
            entry_id = entry.get("sample_id", entry.get("id", idx + 1))
            if entry_id not in processed_ids:
                start_idx = idx
                break
        else:
            start_idx = len(data)  # All processed
        _log(f"[Main] Resuming from entry {start_idx + 1}/{len(data)}")
    else:
        _log(f"[Main] No existing output found, starting from beginning")

    runner = Runner(
        config=None,
        num_traces=args.num_traces,
        num_iters=args.num_iters,
        prompt_variant=args.prompt_variant,
        reasoner_backend=args.reasoner_backend,
        verifier_backend=args.verifier_backend,
        reasoner_thinking_mode=args.reasoner_thinking_mode,
    )

    for idx in range(start_idx, len(data)):
        entry = data[idx]
        entry_id = entry.get("sample_id", entry.get("id", idx + 1))
        _log(f"[Main] Processing entry {idx + 1}/{len(data)} with id={entry_id}")
        natural_spec = entry.get("natural_spec", "")
        verilog = _coerce_rtl_code(entry.get("rtl_code", ""))
        testbench = entry.get("testbench", "")
        coverage_report = entry.get("coverage_report", "")

        reasoning_traces, verification_feedback = runner.forward(
            natural_spec=natural_spec,
            verilog=verilog,
            testbench=testbench,
            coverage_report=coverage_report,
        )
        outputs.append(_build_output_entry(entry, reasoning_traces, verification_feedback))

        if args.save_every > 0 and (idx + 1) % args.save_every == 0:
            _save_jsonl(outputs, args.output)
            _log(f"[Main] Saved interim output after {idx + 1} entries")

        _log(f"[Main] Finished entry {idx + 1}/{len(data)} with id={entry_id}")

    _save_jsonl(outputs, args.output)
    _log(f"[Main] Processed {len(outputs)} entries")
    _log(f"[Main] Saved results to {args.output}")


if __name__ == "__main__":
    main()
