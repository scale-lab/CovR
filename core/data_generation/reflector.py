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

"""Reflector takes in tool feedback to reflect on errors and what could be enhanced
"""
import os
import re 
import sys 
import json
import argparse 
import tempfile
from enum import Enum

from langchain_core.prompts import ChatPromptTemplate

from core.metrics.coverage import run_covered , report_coverage
from core.metrics.vcs import run_urg, report_coverage_vcs
from core.data_generation.state import DataGenState
from core.data_generation.config.config import FlowConfigs, LMConfigs
from core.data_generation.compiler import Compiler

from dotenv import load_dotenv
load_dotenv()


_REFLECTOR_FIX_SYS_PROMPT = """
You are an expert Verilog verification reflection agent.

Your job is to diagnose why a generated testbench failed to compile or simulate, and to produce actionable guidance for the next fix iteration.

**Instructions:**
- Carefully analyze the generator reasoning trace, current testbench, RTL, and environment feedback.
- Focus on the concrete compile/simulation blocker.
- Identify the likely root cause, not just surface-level symptoms.
- Point to concrete evidence from the compiler or simulation log.
- Propose targeted next-step changes that a testbench fixer can apply.
- Be specific about what the model should have done differently.
- Do not discuss coverage closure unless the failure feedback explicitly involves coverage tooling.
- Discuss timing, handshakes, waits, clocks, or multi-cycle sequencing only when they are relevant to the compile/simulation failure.
- For timeouts, first consider non-terminating procedural blocks, unbounded waits, handshake deadlocks, forever loops without a reachable $finish, and loop counters that cannot terminate due to width or type issues.

Your output should be a JSON object with these fields:
  - reasoning: concise evidence-based rationale grounded in the RTL, testbench, and tool feedback
  - error_identification: what specifically went wrong in the reasoning?
  - root_cause_analysis: why did this error occur? What concept was misunderstood?
  - correct_approach: what should the model have done instead?
  - key_insight: what strategy, formula, or principle should be remembered to avoid this error?
"""

_REFLECTOR_ENHANCE_SYS_PROMPT = """
You are an expert Verilog verification reflection agent.

Your job is to diagnose why a generated testbench did not close coverage, and to produce actionable guidance for the next enhancement iteration.

**Instructions:**
- Carefully analyze the generator reasoning trace, current testbench, RTL, coverage stats, and coverage report.
- Use the detailed coverage report as the primary evidence.
- Identify the likely root cause, not just surface-level symptoms.
- Propose targeted next-step changes that a testbench enhancer can apply.
- Be specific about what the model should have done differently.
- Prefer minimal, high-value stimulus changes over broad rewrites.
- Do not claim a scenario was missing if it appears in the current testbench. Cross-check every proposed missing test against the provided testbench first.

Interpret coverage metrics carefully:
- null, "--", missing, or absent metrics mean not applicable unless the report explicitly lists uncovered items.
- Do not describe null/-- condition, branch, line, or FSM coverage as a coverage failure.
- Use the detailed uncovered line/condition/toggle report as the primary evidence.
- If uncovered toggle bits appear structurally constant due to tied constants, zero-extension, width mismatch, concatenation with constants, or mathematically unreachable outputs, say so explicitly.

Before diagnosing, classify the RTL as combinational, sequential, handshake-based, or FSM-like.
- For purely combinational RTL, focus on input-space coverage, branch/condition evidence, toggle evidence, structurally unreachable bits, and deterministic stimulus values.
- Only discuss FSM states, temporal sequencing, reset/clock dwell time, or handshakes if the RTL contains state elements, clocks/resets, counters, valid/ready protocols, or explicit FSM-like control.
- When FSM coverage is weak or the RTL appears stateful, analyze likely unreached states, missing transitions, guards/prerequisites, and whether additional dwell time or multi-cycle sequencing is required.

Your output should be a JSON object with these fields:
  - reasoning: concise evidence-based rationale grounded in the RTL, testbench, and coverage feedback
  - error_identification: what specifically went wrong in the reasoning?
  - root_cause_analysis: why did this error occur? What concept was misunderstood?
  - correct_approach: what should the model have done instead?
  - key_insight: what strategy, formula, or principle should be remembered to avoid this error?
"""

_REFLECTOR_FIX_MODE_USER_PROMPT = """
Analyze why the testbench failed to compile or simulate.

RTL:
{rtl_code}

Generator Reasoning Trace:
{reasoning_trace}

Testbench:
{testbench}

Environment Feedback:
{testbench_error}

Focus on the concrete compile/simulation blocker. If the issue also hints at a deeper sequencing or DUT-interface misunderstanding, mention it briefly.

**Answer in this exact JSON format:**
{{
  "reasoning": "[Concise evidence-based rationale grounded in the RTL, testbench, and tool feedback]",
  "error_identification": "[What specifically went wrong in the reasoning?]",
  "root_cause_analysis": "[Why did this error occur? What concept was misunderstood?]",
  "correct_approach": "[What should the model have done instead?]",
  "key_insight": "[What strategy, formula, or principle should be remembered to avoid this error?]"
}}
"""

_REFLECTOR_ENHANCE_MODE_USER_PROMPT = """
Analyze why coverage is still insufficient and propose better tests.

Spec:
{natural_spec}

RTL:
{rtl_code}

Generator Reasoning Trace:
{reasoning_trace}

Testbench:
{testbench}

Coverage stats:
{coverage_stats}

Environment Feedback:
{coverage_report}

**Answer in this exact JSON format:**
{{
  "reasoning": "[Concise evidence-based rationale grounded in the RTL, testbench, and coverage feedback]",
  "error_identification": "[What specifically went wrong in the reasoning?]",
  "root_cause_analysis": "[Why did this error occur? What concept was misunderstood?]",
  "correct_approach": "[What should the model have done instead?]",
  "key_insight": "[What strategy, formula, or principle should be remembered to avoid this error?]"
}}
"""


class Reflector:
    def __init__(self, config):
        self.config = config
        self.llm = config.lm_config.reflector_llm_model

        self.fix_prompt = ChatPromptTemplate.from_messages([
            ("system", _REFLECTOR_FIX_SYS_PROMPT),
            ("user", _REFLECTOR_FIX_MODE_USER_PROMPT),
        ])

        self.enhance_prompt = ChatPromptTemplate.from_messages([
            ("system", _REFLECTOR_ENHANCE_SYS_PROMPT),
            ("user", _REFLECTOR_ENHANCE_MODE_USER_PROMPT),
        ])

    def _get_chain(self, mode: str):
        if mode == "fix":
            return self.fix_prompt | self.llm
        return self.enhance_prompt | self.llm

    def forward(self, state):
        mode = state.get("reflection_mode", "fix")
        enhance_iters = state.get("enhance_iters", 0)
        fix_iters = state.get("fix_iters", 0)
        chain = self._get_chain(mode)
        
        if mode == "fix":
            if fix_iters == 0: 
                generator_reasoning_trace = state.get("tb_generator_reasoning_trace", "")
            else: 
                generator_reasoning_trace = state.get("fixer_reasoning_trace", "")
        else:
            if enhance_iters == 0: 
                generator_reasoning_trace = state.get("tb_generator_reasoning_trace", "")
            else: 
                generator_reasoning_trace = state.get("enhancer_reasoning_trace", "")

        inputs = {
            "natural_spec": state.get("natural_spec", ""),
            "rtl_code": state.get("rtl_code", ""),
            "testbench": state.get("testbench", ""),
            "testbench_error": state.get("testbench_error", ""),
            "coverage_report": state.get("coverage_report", ""),
            "reasoning_trace": generator_reasoning_trace,
            "coverage_stats": json.dumps(state.get("coverage_stats", {}), indent=2)
        }

        response = chain.invoke(inputs)
        content = response.content if hasattr(response, "content") else str(response)
        print(f"Raw reflector response:\n{content}\n")
        parsed_json = self._parse_json_from_response(content)
        try: 
            reflection_text = json.loads(parsed_json)
        except Exception as e:
            print(f"Error parsing reflector response as JSON: {e}")
            reflection_text = parsed_json

        return {
            "reflection_summary": reflection_text,
            "reflection_history": [reflection_text],
        }

    def _parse_json_from_response(self, response):
        match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        if match:
            return match.group(1)
        return response 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Model name', default='gpt-4o-mini-2024-07-18')
    parser.add_argument('--output', type=str, help='Path to output file for storing the output of each query', default='./output/refiner_eval.log')

    args = parser.parse_args()

    model = args.model 
    output = args.output  

    config = FlowConfigs(
        lm_config=LMConfigs(
            prompter_lm = model,
            fixer_lm = model,
            tb_generator_lm = model,
            enhancer_lm = model,
            checker_lm = model,
            refiner_lm = model
        )
    ) 

    reflector = Reflector(config=config)

    state = {
        "reflection_mode": "fix",
        "natural_spec": "Design a 4-bit adder with carry in and carry out.",
        "rtl_code": """
module adder4(
    input  [3:0] a,
    input  [3:0] b,
    input        cin,
    output [3:0] sum,
    output       cout
);
assign sum = a + b;
endmodule
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
    a = 4'b0011; b = 4'b0101; cin = 1'b0;
    #10;
    if ({cout, sum} != 5'b01000)
        $fatal(0, "Error: wrong result");
    $finish;
end
endmodule
""",
        "compile_log": "Error: port cout is not being driven in the RTL.",
        "coverage_report": "",
        "coverage_stats": {},
        "reflection_history": []
    }

    updated_state = reflector.forward(state)

    print("Reflection summary:")
    print(updated_state["reflection_summary"])
    
    state = {
        "reflection_mode": "enhance",

        "natural_spec": "Design a 4-bit adder with carry in and carry out.",

        "rtl_code": """
    module adder4(
        input  [3:0] a,
        input  [3:0] b,
        input        cin,
        output [3:0] sum,
        output       cout
    );
    assign {cout, sum} = a + b + cin;
    endmodule
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
        // Only a few simple tests
        a = 4'd3; b = 4'd5; cin = 0;
        #10;
        if ({cout, sum} != 5'd8)
            $fatal(0, "Error: wrong result");

        a = 4'd7; b = 4'd1; cin = 1;
        #10;
        if ({cout, sum} != 5'd9)
            $fatal(0, "Error: wrong result");

        $finish;
    end
    endmodule
    """,

        "compile_log": "",

        "coverage_report": """
    Line Coverage: 75%
    Branch Coverage: 40%
    Toggle Coverage: 52%

    Uncovered conditions:
    - Carry-out overflow cases not fully exercised
    - cin=1 cases insufficient
    - Boundary inputs (0, 15) not tested
    - No exhaustive or randomized testing
    """,

        "coverage_stats": {
            "line": 75.0,
            "branch": 40.0,
            "toggle": 52.0,
            "fsm": 0.0
        },

        # Optional: previous failed attempt
        "reflection_history": [
            "Previous enhancement added only a few more directed tests but did not improve branch coverage significantly."
        ]
    }

    updated_state = reflector.forward(state)

    print("\n=== Reflection Summary (Enhance Mode) ===")
    print(updated_state["reflection_summary"])



if __name__ == '__main__':
    main()
