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

"""Revisor checks 
"""
import os
import re 
import sys 
import json
import argparse 
import tempfile
from enum import Enum
from copy import deepcopy

from langchain_core.prompts import ChatPromptTemplate

from core.metrics.coverage import run_covered , report_coverage
from core.metrics.vcs import run_urg, report_coverage_vcs
from core.data_generation.state import DataGenState
from core.data_generation.config.config import FlowConfigs, LMConfigs
from core.data_generation.compiler import Compiler
from core.data_generation.utils import compute_avg_coverage

from dotenv import load_dotenv
load_dotenv()

_CURATOR_FIX_MODE_SYS_PROMPT_DATA_GEN_MODE =  """
You are an expert iteration-memory curator for an agentic RTL verification pipeline.

Your only job is to maintain the evolving context for the current mode of operation.
You do not generate RTL.
You do not generate testbenches.
You do not rewrite the reflection.
You do not give free-form advice.

You receive:
- the existing evolving context for the current mode
- the latest generator output summary
- the latest tool feedback
- the latest reflection summary

You must return the fully updated evolving context as valid JSON only.

## Purpose
Your goal is to preserve cross-iteration memory in a compact and useful form, similar to an iterative debugging notebook:
- what was tried
- what failed or improved
- what the reflection concluded
- whether the iteration was accepted
- whether the loop appears stalled

## Rules
- Preserve all previous iteration records from the provided evolving context.
- Append exactly one new iteration record for the current iteration.
- Do not delete prior iterations unless the input context is empty.
- Do not copy raw logs verbatim when a concise summary is enough.
- Keep each field concise and factual.
- Do not invent failures, improvements, or changes not supported by the inputs.
- If evidence is uncertain, summarize conservatively.
- If the current iteration appears to repeat a previous failure pattern without meaningful progress, mark `stalled` as true.
- Set `restart_recommended` to true only if the recent trajectory clearly appears stuck.
- Set `exit_recommended` to true only if this trajecotry can't be fixed by further iterations and should be abandoned.

## Output Schema
Return valid JSON with exactly this structure:

Output schema:
{{
  "iterations": [
    {{
      "iteration": <current_iteration_number>,
      "generator_output": "short summary",
      "errors_encountered": "short summary",
      "llm_reflection": "short summary",
      "accepted": false
    }}
  ],
  "current_status": {{
    "latest_issue": "short summary",
    "stalled": false,
    "restart_recommended": false,
    "exit_recommended": false
  }}
}}

## Additional Guidance
- `generator_output` should describe what changed this iteration, not dump the full testbench.
- `errors_encountered` should summarize the current error log.
- `llm_reflection` should summarize the latest reflection into 1-3 sentences worth of content.
- `accepted` should reflect whether this iteration succeeded or improved enough to keep.
- `latest_issue` should describe the most important unresolved problem after this iteration.
- `stalled` should be true if the last two iterations have not been accepted and show no meaningful progress.

## Fix Mode Focus
- This context is for fixing compile/simulation failures.
- `errors_encountered` should summarize the current compile/simulation blocker from tool feedback.
- Focus on concrete issues such as syntax errors, module/port mismatches, invalid constructs, simulation timeouts, unbounded waits, handshake deadlocks, loop-counter termination issues, or batch-simulation termination issues like `$stop` instead of `$finish`.
- Use the current testbench only as factual evidence of the testbench structure.
- `generator_output` should describe what the fixer attempted this iteration.
- `llm_reflection` should summarize the reflection's actionable conclusion, not independently re-diagnose the failure.
- `accepted` must exactly match the provided Acceptance Signal.
"""

_CURATOR_ENHANCE_MODE_SYS_PROMPT_DATA_GEN_MODE = """
You are an expert iteration-memory curator for an agentic RTL verification pipeline.

Your only job is to maintain the evolving context for the current mode of operation.
You do not generate RTL.
You do not generate testbenches.
You do not rewrite the reflection.
You do not give free-form advice.

You receive:
- the existing evolving context for the current mode
- the latest generator output summary
- the latest tool feedback
- the latest reflection summary

You must return the fully updated evolving context as valid JSON only.

## Purpose
Your goal is to preserve cross-iteration memory in a compact and useful form, similar to an iterative debugging notebook:
- what was tried
- what failed or improved
- what the reflection concluded
- whether the iteration was accepted
- whether the loop appears stalled

## Rules
- Preserve all previous iteration records from the provided evolving context.
- Append exactly one new iteration record for the current iteration.
- Do not delete prior iterations unless the input context is empty.
- Do not copy raw logs verbatim when a concise summary is enough.
- Keep each field concise and factual.
- Do not invent failures, improvements, or changes not supported by the inputs.
- If evidence is uncertain, summarize conservatively.
- If the current iteration appears to repeat a previous failure pattern without meaningful progress, mark `stalled` as true.
- Set `restart_recommended` to true only if the recent trajectory clearly appears stuck.
- Set `exit_recommended` to true only if this trajecotry can't be fixed by further iterations and should be abandoned.

## Output Schema
Return valid JSON with exactly this structure:

Output schema:
{{
  "iterations": [
    {{
      "iteration": <current_iteration_number>,
      "generator_output": "short summary",
      "coverage_stats": "current coverage stats",
      "coverage_gaps": "short summary",
      "llm_reflection": "short summary",
      "accepted": false
    }}
  ],
  "current_status": {{
    "latest_issue": "short summary",
    "stalled": false,
    "restart_recommended": false,
    "exit_recommended": false
  }}
}}

## Additional Guidance
- This context is for improving coverage after the testbench compiles and simulates.
- Use the current testbench only as factual evidence of what stimulus was already tried. Do not describe a scenario as missing if it appears in the current testbench.
- `generator_output` should describe what the coverage enhancer attempted this iteration.
- `coverage_stats`: should have the current stats for coverage (line, toggle, cond, branch, FSM). "--" or null means that this coverage is not applicable for the design. 
- `coverage_gaps` should summarize the remaining coverage issue using concrete coverage stats or uncovered items. Ground this in the coverage report and testbench provided. 
    For coverage report, treat null, "--", missing, or absent metrics as not applicable.
- `llm_reflection` should summarize the reflection's actionable coverage guidance, not independently re-diagnose coverage.
- `accepted` should reflect whether this iteration succeeded or improved enough to keep.
- `latest_issue` should describe the most important unresolved problem after this iteration.
- `stalled` should be true if the last two iterations have not been accepted and show no meaningful progress.
"""

_CURATOR_FIX_MODE_SYS_PROMPT ="""
You are an expert iteration-memory curator for an agentic RTL verification pipeline.

Your only job is to maintain the evolving context for the current mode of operation.
You do not generate RTL.
You do not generate testbenches.
You do not rewrite the reflection.
You do not give free-form advice.

You receive:
- the existing evolving context for the current mode
- the latest generator output summary
- the latest tool feedback
- the latest reflection summary

You must return the fully updated evolving context as valid JSON only.

## Purpose
Your goal is to preserve cross-iteration memory in a compact and useful form, similar to an iterative debugging notebook:
- what was tried
- what failed or improved
- what the reflection concluded
- whether the iteration was accepted
- whether the loop appears stalled

## Rules
- Preserve all previous iteration records from the provided evolving context.
- Append exactly one new iteration record for the current iteration.
- Do not delete prior iterations unless the input context is empty.
- Do not copy raw logs verbatim when a concise summary is enough.
- Keep each field concise and factual.
- Do not invent failures, improvements, or changes not supported by the inputs.
- If evidence is uncertain, summarize conservatively.
- If the current iteration appears to repeat a previous failure pattern without meaningful progress, mark `stalled` as true.
- Set `restart_recommended` to true only if the recent trajectory clearly appears stuck.
- Set `exit_recommended` to true only if this trajecotry can't be fixed by further iterations and should be abandoned.

## Output Schema
Return valid JSON with exactly this structure:

Output schema:
{{
  "iterations": [
    {{
      "iteration": <current_iteration_number>,
      "generator_output": "short summary",
      "errors_encountered": "short summary",
      "llm_reflection": "short summary",
      "accepted": false
    }}
  ],
  "current_status": {{
    "latest_issue": "short summary",
    "stalled": false,
    "restart_recommended": false,
    "exit_recommended": false
  }}
}}

## Additional Guidance
- `generator_output` should describe what changed this iteration, not dump the full testbench.
- `errors_encountered` should summarize the current error log.
- `llm_reflection` should summarize the latest reflection into 1-3 sentences worth of content.
- `accepted` should reflect whether this iteration succeeded or improved enough to keep.
- `latest_issue` should describe the most important unresolved problem after this iteration.
- `stalled` should be true if the last two iterations have not been accepted and show no meaningful progress.

## Fix Mode Focus
- This context is for fixing compile/simulation failures.
- `errors_encountered` should summarize the current compile/simulation blocker from tool feedback.
- Focus on concrete issues such as syntax errors, module/port mismatches, invalid constructs, simulation timeouts, unbounded waits, handshake deadlocks, loop-counter termination issues, or batch-simulation termination issues like `$stop` instead of `$finish`.
- Use the current testbench only as factual evidence of the testbench structure.
- `generator_output` should describe what the fixer attempted this iteration.
- `llm_reflection` should summarize the reflection's actionable conclusion, not independently re-diagnose the failure.
- `accepted` must exactly match the provided Acceptance Signal.
"""

_CURATOR_ENHANCE_MODE_SYS_PROMPT = """
You are an expert iteration-memory curator for an agentic RTL verification pipeline.

Your only job is to maintain the evolving context for the current mode of operation.
You do not generate RTL.
You do not generate testbenches.
You do not rewrite the reflection.
You do not give free-form advice.

You receive:
- the existing evolving context for the current mode
- the latest generator output summary
- the latest tool feedback
- the latest reflection summary

You must return the fully updated evolving context as valid JSON only.

## Purpose
Your goal is to preserve cross-iteration memory in a compact and useful form, similar to an iterative debugging notebook:
- what was tried
- what failed or improved
- what the reflection concluded
- whether the iteration was accepted
- whether the loop appears stalled

## Rules
- Preserve all previous iteration records from the provided evolving context.
- Append exactly one new iteration record for the current iteration.
- Do not delete prior iterations unless the input context is empty.
- Do not copy raw logs verbatim when a concise summary is enough.
- Keep each field concise and factual.
- Do not invent failures, improvements, or changes not supported by the inputs.
- If evidence is uncertain, summarize conservatively.
- If the current iteration appears to repeat a previous failure pattern without meaningful progress, mark `stalled` as true.
- Set `restart_recommended` to true only if the recent trajectory clearly appears stuck.
- Set `exit_recommended` to true only if this trajecotry can't be fixed by further iterations and should be abandoned.

## Output Schema
Return valid JSON with exactly this structure:

Output schema:
{{
  "iterations": [
    {{
      "iteration": <current_iteration_number>,
      "generator_output": "short summary",
      "coverage_stats": "current coverage stats",
      "coverage_gaps": "short summary",
      "llm_reflection": "short summary",
      "accepted": false
    }}
  ]
}}

## Additional Guidance
- This context is for improving coverage after the testbench compiles and simulates.
- Use the current testbench only as factual evidence of what stimulus was already tried. Do not describe a scenario as missing if it appears in the current testbench.
- `generator_output` should describe what the coverage enhancer attempted this iteration.
- `coverage_stats`: should have the current stats for coverage (line, toggle, cond, branch, FSM). "--" or null means that this coverage is not applicable for the design. 
- `coverage_gaps` should summarize the remaining coverage issue using concrete coverage stats or uncovered items. Ground this in the coverage report and testbench provided. 
    For coverage report, treat null, "--", missing, or absent metrics as not applicable.
- `llm_reflection` should summarize the reflection's actionable coverage guidance, not independently re-diagnose coverage.
- `accepted` should reflect whether this iteration succeeded or improved enough to keep.
"""

_CURATOR_USER_PROMPT = """
Update the evolving context for the current iteration.
## Current Mode
{reflection_mode}

## Existing Evolving Context
{evolving_context}

## Current Iteration Number
{iteration}

## Current Generator Output
{generator_output}

## Current Generator Testbench 
{testbench}

## Current Tool Feedback
{tool_feedback}

## Current Reflection Summary
{reflection_summary}

## Acceptance Signal
{accepted}

## Instructions
1. Read the existing evolving context.
2. Preserve all prior iterations already present.
3. Append exactly one new iteration record for the current iteration.
4. Summarize the current generator output, current issue, and reflection compactly.
5. Return valid JSON only. Do not wrap it in markdown fences.

Remember:
- Keep summaries concise.
- Do not include the full raw testbench or full raw logs.
- Do not omit previous iterations.
- Do not output explanations outside the JSON.
"""

class Curator:
    def __init__(self, config):
        self.config = config

        if self.config.data_gen_mode: 
            fix_sys_prompt = _CURATOR_FIX_MODE_SYS_PROMPT_DATA_GEN_MODE
            enhance_sys_prompt = _CURATOR_ENHANCE_MODE_SYS_PROMPT_DATA_GEN_MODE
        else:
            fix_sys_prompt = _CURATOR_FIX_MODE_SYS_PROMPT
            enhance_sys_prompt = _CURATOR_ENHANCE_MODE_SYS_PROMPT
        
        self.fix_prompt = ChatPromptTemplate.from_messages([
            ("system", fix_sys_prompt),
            ("user", _CURATOR_USER_PROMPT),
        ])

        self.enhance_prompt = ChatPromptTemplate.from_messages([
            ("system", enhance_sys_prompt),
            ("user", _CURATOR_USER_PROMPT),
        ])
        
        self.llm = config.lm_config.curator_llm_model
        self.fix_curator = self.fix_prompt | self.llm
        self.enhance_curator = self.enhance_prompt | self.llm

    def forward(self, state: DataGenState):
        mode = state.get("reflection_mode", "fix")

        if mode == "enhance":
            context_key = "enhancement_context"
            iteration = state.get("enhance_iters", 0)
            curator = self.enhance_curator
        else:
            context_key = "fixing_context"
            iteration = state.get("fix_iters", 0)
            curator = self.fix_curator

        default_context = {
            "iterations": [],
            "current_status": {
                "latest_issue": "",
                "stalled": False,
                "restart_recommended": False,
                "exit_recommended": False,
            },
        }
        evolving_context = state.get(context_key, default_context)
        evolving_context_for_prompt = (
            evolving_context
            if isinstance(evolving_context, str)
            else json.dumps(evolving_context, indent=2)
        )
        
        if mode == "fix":
            if iteration == 0:
                generator_output = state.get("tb_generator_reasoning_trace", "")
            else:
                generator_output = state.get("fixer_reasoning_trace", "")
        else: 
            if iteration == 0:
                generator_output = state.get("tb_generator_reasoning_trace", "")
            else: 
                generator_output = state.get("enhancer_reasoning_trace", "")

        if self.config.wo_curator:
            updated_context = self._build_fallback_context(
                state=state,
                mode=mode,
                evolving_context=evolving_context,
                iteration=iteration,
                generator_output=generator_output,
            )
        else: 
            response = curator.invoke({
                "reflection_mode": mode,
                "evolving_context": evolving_context_for_prompt,
                "iteration": iteration,
                "generator_output": generator_output,
                "testbench": state.get("testbench"),
                "tool_feedback": self._tool_feedback_summary(state, mode),
                "reflection_summary": state.get("reflection_summary", ""),
                "accepted": json.dumps(self._accepted(state, mode)),
            })

            content = response.content if hasattr(response, "content") else str(response)
            print(f"Raw curator response:\n{content}\n")

            parsed_json = self._parse_json_from_response(content)

            try: 
                updated_context = json.loads(parsed_json)
                updated_context = self._normalize_evolving_context(updated_context)
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                print(f"Parsed JSON string:\n{parsed_json}\n")
                context = deepcopy(evolving_context) if isinstance(evolving_context, dict) else {}
                context.setdefault("iterations", [])
                context.setdefault("current_status", {})
                if mode == "fix":
                    context["iterations"].append({
                        "iteration": iteration,
                        "generator_output": generator_output,
                        "errors_encountered": self._tool_feedback_summary(state, mode),
                        "accepted": self._accepted(state, mode)
                    })
                else:
                    context["iterations"].append({
                        "iteration": iteration,
                        "coverage_stats": state.get("coverage_stats", {})
                    })
                updated_context = context
                    
            print(updated_context)


        return {context_key: updated_context}

    def _normalize_evolving_context(self, context) -> dict:
        if isinstance(context, list):
            return {
                "iterations": context,
                "current_status": {},
            }

        if not isinstance(context, dict):
            return {
                "iterations": [],
                "current_status": {},
            }

        context.setdefault("iterations", [])
        context.setdefault("current_status", {})

        if not isinstance(context["iterations"], list):
            context["iterations"] = []
        if not isinstance(context["current_status"], dict):
            context["current_status"] = {}

        return context

    def _parse_json_from_response(self, response):
        match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        parsed = match.group(1) if match else response
        return re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", parsed)


    def _tool_feedback_summary(self, state: DataGenState, mode: str) -> str:
        if mode == "enhance":
            payload = {
                "coverage_met": state.get("coverage_met", False),
                "coverage_stats": state.get("coverage_stats", {}),
                "coverage_report": state.get("coverage_report", ""),
            }
        else:
            payload = {
                "testbench_runs": state.get("testbench_runs", False),
                "testbench_error": state.get("testbench_error", ""),
            }

        return json.dumps(payload, indent=2)

    def _accepted(self, state: DataGenState, mode: str) -> bool:
        if mode != "enhance":
            return bool(state.get("testbench_runs", False))

        if state.get("coverage_met", False):
            return True

        coverage_stats_list = state.get("coverage_stats_list", [])
        if len(coverage_stats_list) < 2:
            return False

        current = coverage_stats_list[-1]
        previous_best = self._best_coverage_stats(coverage_stats_list[:-1])

        return compute_avg_coverage(current) > compute_avg_coverage(previous_best)


    def _best_coverage_stats(self, stats_list: list[dict]) -> dict:
        if not stats_list:
            return {}
        return max(stats_list, key=compute_avg_coverage)

    def _build_fallback_context(
        self,
        state: DataGenState,
        mode: str,
        evolving_context,
        iteration: int,
        generator_output: str,
    ) -> dict:
        context = deepcopy(evolving_context) if isinstance(evolving_context, dict) else {}
        context.setdefault("iterations", [])
        context.setdefault("current_status", {})
        context["iterations"].append({
            "iteration": iteration,
            "generator_output": generator_output,
            "tool_feedback": self._tool_feedback_summary(state, mode)
        })
        return context



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt-4o-mini-2024-07-18')
    args = parser.parse_args()

    model = args.model

    config = FlowConfigs(
        lm_config=LMConfigs(
            prompter_lm=model,
            fixer_lm=model,
            tb_generator_lm=model,
            enhancer_lm=model,
            checker_lm=model,
            refiner_lm=model,
        )
    )

    curator = Curator(config=config)

    fix_state = {
        "reflection_mode": "fix",
        "fix_iters": 1,
        "enhance_iters": 0,
        "testbench_runs": False,
        "testbench_error": "Error: Module opt_mo is not defined. The RTL top module is opt_model.",
        "reflection_summary": """
The failure is caused by a DUT instantiation mismatch in the testbench.
The checker logic appears valid.
The next step should update the instantiated DUT name and keep the existing self-checking behavior.
""",
        "testbench": """
module testbench;
  wire zero;
  opt_mo uut (
    .zero(zero)
  );
  initial begin
    #10;
    if (zero !== 1'b0) $fatal(0, "Error: wrong result");
    $finish;
  end
endmodule
""",
        "fixing_context": {
            "iterations": [],
            "current_status": {
                "latest_issue": "",
                "stalled": False,
                "restart_recommended": False
            }
        }
    }

    fix_output = curator.forward(fix_state)
    print("=== Updated Fixing Context ===")
    print(json.dumps(fix_output["fixing_context"], indent=2))

    enhance_state = {
        "reflection_mode": "enhance",
        "fix_iters": 1,
        "enhance_iters": 2,
        "covered_executes": True,
        "coverage_met": False,
        "coverage_stats": {
            "line": 78.0,
            "branch": 45.0,
            "toggle": 56.0
        },
        "coverage_report": """
Line Coverage: 78%
Branch Coverage: 45%
Toggle Coverage: 56%

Uncovered conditions:
- Carry-out overflow cases not fully exercised
- cin=1 cases insufficient
- Boundary inputs (0, 15) not fully tested
""",
        "reflection_summary": """
Coverage is still weak on overflow-related branches and cin=1 scenarios.
The next enhancement should add targeted max-value and boundary-value tests rather than broad random expansion.
""",
        "enhancer_reasoning_trace": """
Added a few directed cases and attempted to improve branch coverage, but important overflow scenarios remain under-tested.
""",
        "testbench": """
module testbench;
reg [3:0] a, b;
reg cin;
wire [3:0] sum;
wire cout;

adder4 uut (
  .a(a),
  .b(b),
  .cin(cin),
  .sum(sum),
  .cout(cout)
);

initial begin
  a = 4'd3; b = 4'd5; cin = 0;
  #10;
  if ({cout, sum} != 5'd8) $fatal(0, "Error: wrong result");

  a = 4'd15; b = 4'd15; cin = 1;
  #10;
  if ({cout, sum} != 5'd31) $fatal(0, "Error: wrong result");

  $finish;
end
endmodule
""",
        "enhancement_context": {
            "iterations": [
                {
                    "iteration": 1,
                    "mode": "enhance",
                    "generator_output": "Added a few extra directed tests.",
                    "errors_encountered": "Coverage remained low on branch and toggle metrics.",
                    "llm_reflection": "Previous enhancement was too shallow and missed overflow-focused scenarios.",
                    "accepted": False
                }
            ],
            "current_status": {
                "latest_issue": "Coverage remains low on overflow-related logic.",
                "stalled": False,
                "restart_recommended": False
            }
        }
    }

    enhance_output = curator.forward(enhance_state)
    print("=== Updated Enhancement Context ===")
    print(json.dumps(enhance_output["enhancement_context"], indent=2))


if __name__ == "__main__":
    main()  
