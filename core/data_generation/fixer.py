"""Fixes any syntax errors in the testbench 
"""
import re 
import json
import copy
import argparse 

from langchain_core.prompts import ChatPromptTemplate

from core.data_generation.agent import Agent 
from core.data_generation.state import DataGenState
from core.data_generation.config.config import FlowConfigs, LMConfigs
from core.data_generation.utils import _strip_testbench, extract_testbench_code

from dotenv import load_dotenv
load_dotenv()

_FIXER_SYS_PROMPT = """
You are an expert {language} verification engineer.
Your job is to fix syntax and functional errors and produce a syntactically and functionally valid, simulator-ready testbench.

Non-negotiable requirements:
- Preserve stimulus-only behavior. Do not introduce assertions, scoreboards, golden models, `$fatal`, `$error`, or correctness-checking if/else blocks.
- Never convert a stimulus-only testbench into a self-checking or monitor-heavy testbench.
- Prefer minimal targeted repairs over broad rewrites.
- Do not change the intended DUT functionality.
- `wire`, `reg`, and `int` declarations are not allowed inside `initial` blocks.
- Do not add or keep VCD or waveform dumping logic such as `$dumpfile` or `$dumpvars`.
- If simulation is timing out, reduce unnecessary stimulus while preserving the intended coverage intent.
"""


_FIXER_USER_PROMPT = """
## Inputs

### Fixing Context
{fixing_context}

### Reflection Summary
{reflection_summary}

### RTL Code
{rtl_code}

### Current Testbench
{testbench}

### Current Compiler/Simulator Error Log
{testbench_error}

## Instructions
Repair the current testbench without weakening its verification quality.

You must preserve or restore the following properties:
1. The testbench must remain stimulus-only.
2. Preserve the existing stimulus, timing structure, and DUT connectivity unless the error log requires a targeted fix.
3. Do not add correctness-checking logic, assertions, scoreboards, failure messages, or VCD/waveform dumping.
4. Do not replace stimulus with comments, monitors, or passive-only scaffolding that weakens coverage intent.

Use the fixing context as iterative memory:
- avoid repeating failed repair strategies
- preserve helpful prior fixes
- use previous accepted directions when still relevant

Use the reflection summary as high-level guidance.
If the fixing context or reflection summary conflicts with the current compiler/simulator error log, prioritize the current error log.

First briefly reason about:
- the root cause
- which stimulus, timing, and DUT connectivity must be repaired
- the minimal safe changes to make

Then output only the final fixed testbench inside a fenced code block exactly like this:

```testbench
<fixed testbench code here>
```
"""


_FIXER_USER_PROMPT_WO_REFLECTION = """
## Inputs

### Fixing Context
{fixing_context}

### RTL Code
{rtl_code}

### Current Testbench
{testbench}

### Current Compiler/Simulator Error Log
{testbench_error}

## Instructions
Repair the current testbench without weakening its verification quality.

You must preserve or restore the following properties:
1. The testbench must remain stimulus-only.
2. Preserve the existing stimulus, timing structure, and DUT connectivity unless the error log requires a targeted fix.
3. Do not add correctness-checking logic, assertions, scoreboards, failure messages, or VCD/waveform dumping.
4. Do not replace stimulus with comments, monitors, or passive-only scaffolding that weakens coverage intent.

Use the fixing context as iterative memory:
- avoid repeating failed repair strategies
- preserve helpful prior fixes
- use previous accepted directions when still relevant

If the fixing context conflicts with the current compiler/simulator error log, prioritize the current error log.

First briefly reason about:
- the root cause
- which stimulus, timing, and DUT connectivity must be repaired
- the minimal safe changes to make

Then output only the final fixed testbench inside a fenced code block exactly like this:

```testbench
<fixed testbench code here>
```
"""


class Fixer():
    def __init__(self, config):
        super().__init__()
        self.config = config

        if self.config.wo_reflector:
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", str(_FIXER_SYS_PROMPT)),
                ("user", str(_FIXER_USER_PROMPT_WO_REFLECTION)),
            ])
        else:
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", str(_FIXER_SYS_PROMPT)),
                ("user", _FIXER_USER_PROMPT),
            ])

        self.llm = config.lm_config.fixer_llm_model
        self.fixer = self.prompt | self.llm 
 
    def forward(self, state: DataGenState):
        testbench = self._strip_waveform_dumping(
            self._parse_testbench(state.get("testbench"))
        )
        language = state.get("language")
        rtl_code =  state.get("rtl_code")
        testbench_error =  state.get("testbench_error")
        fix_iters = state.get("fix_iters", 0)
        reflection_summary = state.get("reflection_summary", "")
        fixing_context = state.get("fixing_context", {"iterations": [], "current_status": {}})

        fixing_context = self._attach_testbenches(state)

        if len(testbench_error) > 2600:
            testbench_error = testbench_error[0:2600]
        
        response = self.fixer.invoke({
            "rtl_code": rtl_code, 
            "testbench": testbench,
            "language": language,
            "testbench_error": testbench_error,
            "reflection_summary": reflection_summary,
            "fixing_context": json.dumps(fixing_context, indent=2)
        })
        
        fixed_testbench = self._strip_waveform_dumping(self._parse_testbench(response.content))

        reasoning_trace = _strip_testbench(response.content)

        if fix_iters == 0:
            fixer_testbenches = [testbench, fixed_testbench]
        else:
            fixer_testbenches = [fixed_testbench]

        return {"testbench": fixed_testbench, "fix_iters": fix_iters + 1, "fixer_reasoning_trace": reasoning_trace, "fixer_testbenches": fixer_testbenches}

    def _parse_testbench(self, content):
        return extract_testbench_code(content)

    def _strip_waveform_dumping(self, testbench: str):
        if not testbench:
            return ""

        lines = []
        for line in testbench.splitlines():
            if re.search(r"\$dump(file|vars)\b", line):
                continue
            lines.append(line)
        return "\n".join(lines).strip()

    def _attach_testbenches(self, state):
        fixing_context = copy.deepcopy(
            state.get("fixing_context", {"iterations": [], "current_status": {}})
        )
        if isinstance(fixing_context, list):
            fixing_context = {"iterations": fixing_context, "current_status": {}}
        elif not isinstance(fixing_context, dict):
            fixing_context = {"iterations": [], "current_status": {}}
        fixing_context.setdefault("iterations", [])
        fixing_context.setdefault("current_status", {})
        if not isinstance(fixing_context["iterations"], list):
            fixing_context["iterations"] = []

        records = fixing_context.get("iterations", [])
        fixer_testbenches = state.get("fixer_testbenches", [])
        total_records = len(records)

        for idx, record in enumerate(records):
            if idx == len(records) - 1:
                record["testbench"] = self._strip_waveform_dumping(
                    self._parse_testbench(state.get("testbench", ""))
                )
            elif idx < len(fixer_testbenches):
                record["testbench"] = self._strip_waveform_dumping(
                    self._parse_testbench(fixer_testbenches[idx])
                )
            else:
                record["testbench"] = ""

        return fixing_context



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Model name', default='gpt-4o-mini-2024-07-18')
    parser.add_argument('--output', type=str, help='Path to output file for storing the output of each query', default='./output/refiner_eval.log')

    args = parser.parse_args()

    model = args.model 
    output = args.output 

    testbench = """
`timescale 1ns/1ps

module testbench;
  reg clk;
  wire zero;

  // DUT
  opt_mo uut (
    .zero(zero)
  );

  // Optional clock (not used by DUT)
  initial clk = 0;
  always #5 clk = ~clk;

  // Wave dump (helpful for debugging)
  initial begin
    $dumpfile("wave.vcd");
    $dumpvars(0, testbench);
  end

  initial begin
    // Basic checks
    #10;
    if (zero !== 1'b0) begin
      $display("Test failed @10ns: zero expected 0, got %b", zero);
      $finish;
    end

    #10;
    if (zero !== 1'b0) begin
      $display("Test failed @20ns: zero expected 0, got %b", zero);
      $finish;
    end

    $display("All test cases passed");
    $finish;
  end
endmodule

"""
    rtl_code = """
`timescale 1ns/1ps

module opt_model (
  output zero
);
  assign zero = 1'b0;
endmodule
"""

    testbench_error = "Module opt_mo isn't defined"

    config = FlowConfigs(
        lm_config=LMConfigs(
            prompter_lm = model,
            fixer_lm = model
        )
    )
    fixer = Fixer(config=config) 

    output = fixer.forward({
        "testbench_error": testbench_error,
        "testbench": testbench,
        "rtl_code": rtl_code    
    })

    print(output["testbench"])


if __name__ == '__main__':
    main()
