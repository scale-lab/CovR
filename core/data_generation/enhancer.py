"""Enhances TB Coverage based on the coverage report
"""
import re 
import json
import copy
import argparse 

from langchain_core.prompts import ChatPromptTemplate

from core.data_generation.agent import Agent 
from core.data_generation.state import DataGenState
from core.data_generation.config.config import FlowConfigs, LMConfigs
from core.train.prompts import SYS_PROMPT, USER_PROMPT_SECTIONED_TEMPLATE
from core.data_generation.utils import _strip_testbench, extract_testbench_code

from dotenv import load_dotenv
load_dotenv()

_ENHANCER_SYS_PROMPT = """
You are an expert Verification Engineer and Testbench Optimizer.

Your job is to improve an existing testbench so it closes more coverage on the provided RTL.

You are working in an iterative enhancement loop.
You may be given memory from previous attempts.
Use that memory to avoid repeating ineffective enhancement strategies, but always prioritize the current coverage evidence.

Important rules:
- Prefer targeted additions over full rewrites.
- Focus on uncovered or weakly covered behavior.
- Keep the testbench deterministic.
- Preserve clocks, resets when valid.
- Do not degrade the coverage quality of already-covered behavior.
- Preserve stimulus-only structure. Do not add assertions, scoreboards, golden models, `$fatal`/`$error` checks, correctness-checking branches, or VCD/waveform dumping.
- Focus on stronger stimulus only.
"""

_ENHANCER_USER_PROMPT = """
### Inputs

#### Enhancement Context
{enhancement_context}

#### Reflection Summary
{reflection_summary}

#### Natural Spec
{natural_spec}

#### RTL Code
{rtl_code}

#### Current Testbench
{testbench}

#### Current Coverage Report
{coverage_report}

### Instructions
You are improving the current testbench to increase verification coverage for the given RTL.

Use the enhancement context as iterative memory:
- identify which enhancement directions were previously attempted
- avoid repeating changes that did not improve coverage
- preserve previous additions that were useful
- reuse successful coverage-closing strategies when still relevant

Use the reflection summary as high-level guidance about:
- the most likely remaining coverage gaps
- the most promising next scenarios to add
- pitfalls from prior unsuccessful enhancement attempts

Prioritize signals in this order:
1. The current coverage report and coverage stats
2. The current RTL and current testbench
3. The reflection summary
4. The enhancement context

If the enhancement context or reflection summary conflicts with the current coverage report, trust the current coverage report.

### Your task
Analyze the remaining coverage gaps and produce an improved testbench that:
1. Increases coverage with targeted, high-value additions.
2. Preserves existing useful stimulus, timing, clocks, resets, and DUT connectivity.
3. Avoids broad rewrites unless necessary.
4. Adds new directed tests, corner cases, or randomized scenarios only where they are likely to close real gaps.
5. Maintains determinism and reproducibility.
6. Preserves the requested testbench style.
7. Preserves clocks, resets, and other valid infrastructure when possible, but do not add or keep VCD/waveform dumping logic.
8. Avoids regressing coverage or removing previously valuable tests.

When FSM coverage is weak:
- prioritize state reachability and transition closure
- add directed multi-cycle sequences for uncovered transitions
- target rare guards, recovery paths, and timeout paths
- vary predecessor state and dwell length, not just input values

When useful:
- add comments explaining why a new scenario was introduced
- add deterministic constrained-random stimulus with a fixed seed
- target edge cases, rare branches, timing-sensitive behavior, and boundary conditions, fsm states
- prefer compact, purposeful additions over noisy test expansion

### Output Format
First, reason about:
- the main remaining coverage gaps
- which prior enhancement directions should be avoided or reused
- the specific additions you will make now

Then output the final enhanced testbench inside a fenced code block exactly like this:

```testbench
<enhanced testbench code here>
```
"""


_ENHANCER_USER_PROMPT_WO_REFLECTION = """
### Inputs

#### Enhancement Context
{enhancement_context}

#### Natural Spec
{natural_spec}

#### RTL Code
{rtl_code}

#### Current Testbench
{testbench}

#### Current Coverage Report
{coverage_report}

### Instructions
You are improving the current testbench to increase verification coverage for the given RTL.

Use the enhancement context as iterative memory:
- identify which enhancement directions were previously attempted
- avoid repeating changes that did not improve coverage
- preserve previous additions that were useful
- reuse successful coverage-closing strategies when still relevant


Prioritize signals in this order:
1. The current coverage report and coverage stats
2. The current RTL and current testbench
3. The enhancement context

If the enhancement context conflicts with the current coverage report, trust the current coverage report.

### Your task
Analyze the remaining coverage gaps and produce an improved testbench that:
1. Increases coverage with targeted, high-value additions.
2. Preserves existing useful stimulus.
3. Avoids broad rewrites unless necessary.
4. Adds new directed tests, corner cases, or randomized scenarios only where they are likely to close real gaps.
5. Maintains determinism and reproducibility.
6. Preserves the requested testbench style.
7. Preserves clocks, resets, and other valid infrastructure when possible, but do not add or keep VCD/waveform dumping logic.
8. Avoids regressing coverage or removing previously valuable tests.

When FSM coverage is weak:
- prioritize state reachability and transition closure
- add directed multi-cycle sequences for uncovered transitions
- target rare guards, recovery paths, and timeout paths
- vary predecessor state and dwell length, not just input values

When useful:
- add comments explaining why a new scenario was introduced
- add deterministic constrained-random stimulus with a fixed seed
- target edge cases, rare branches, timing-sensitive behavior, and boundary conditions, fsm states
- prefer compact, purposeful additions over noisy test expansion

### Output Format
First, reason about:
- the main remaining coverage gaps
- which prior enhancement directions should be avoided or reused
- the specific additions you will make now

Then output the final enhanced testbench inside a fenced code block exactly like this:

```testbench
<enhanced testbench code here>
```
"""

class Enhancer():

    def __init__(self, config):
        super().__init__()
        self.config = config

        if not self.config.data_gen_mode:
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", str(SYS_PROMPT)),
                ("user", USER_PROMPT_SECTIONED_TEMPLATE),
            ])
        elif self.config.wo_reflector:
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", str(_ENHANCER_SYS_PROMPT)),
                ("user", _ENHANCER_USER_PROMPT_WO_REFLECTION),
            ])
        else:   
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", str(_ENHANCER_SYS_PROMPT)),
                ("user", _ENHANCER_USER_PROMPT),
            ])

        self.llm = config.lm_config.enhancer_llm_model
        self.enhancer = self.prompt | self.llm 
 
    def forward(self, state: DataGenState):
        rtl_code = state.get("rtl_code")
        natural_spec = state.get("natural_spec")
        testbench = self._strip_waveform_dumping(
            self._parse_testbench(state.get("testbench"))
        )
        language = state.get("language") 
        enhance_iters = state.get("enhance_iters", 0)
        coverage_report = state.get("coverage_report")
        reflection_summary = state.get("reflection_summary", "")
        enhancement_context = state.get("enhancement_context", {"iterations": [], "current_status": {}})

        enhancement_context = self._attach_testbenches(state)

        if self.config.data_gen_mode: 
            response = self.enhancer.invoke({
                "testbench": testbench, 
                "rtl_code": rtl_code, 
                "coverage_report": coverage_report,
                "reflection_summary": reflection_summary,
                "enhancement_context": json.dumps(enhancement_context, indent=2)
            })
        else:
            context = f"""
# Previous Enhancement Iterations
{json.dumps(enhancement_context, indent=2)}
# Reflection Summary 
{reflection_summary}
"""
            response = self.enhancer.invoke({
                "rtl_code": rtl_code, 
                "context": context,
                "natural_spec": natural_spec
            })
        
        if type(response) != str:
            response = response.content

        reasoning_trace = _strip_testbench(response)
        enhanced_testbench = self._parse_testbench(response)
        enhanced_testbench = self._strip_waveform_dumping(enhanced_testbench)

        if enhance_iters == 0:
            enhancer_testbenches = [testbench, enhanced_testbench]
        else:
            enhancer_testbenches = [enhanced_testbench]

        return {"testbench": enhanced_testbench, "enhance_iters": enhance_iters + 1, "enhancer_reasoning_trace": reasoning_trace, "enhancer_testbenches": enhancer_testbenches}

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
        enhancement_context = copy.deepcopy(
            state.get("enhancement_context", {"iterations": [], "current_status": {}})
        )
        if isinstance(enhancement_context, list):
            enhancement_context = {"iterations": enhancement_context, "current_status": {}}
        elif not isinstance(enhancement_context, dict):
            enhancement_context = {"iterations": [], "current_status": {}}
        enhancement_context.setdefault("iterations", [])
        enhancement_context.setdefault("current_status", {})
        if not isinstance(enhancement_context["iterations"], list):
            enhancement_context["iterations"] = []

        coverage_stats_list = state.get("coverage_stats_list", [])
        for idx, record in enumerate(enhancement_context["iterations"]):
            if len(state["enhancer_testbenches"]) == 0: 
                record["testbench"] = self._strip_waveform_dumping(
                    self._parse_testbench(state["testbench"])
                )
            else: 
                record["testbench"] = self._strip_waveform_dumping(
                    self._parse_testbench(state["enhancer_testbenches"][idx])
                )

            record["coverage_stats"] = coverage_stats_list[idx]

        return enhancement_context




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

    enhancer = Enhancer(
        config=config
    )
    
    testbench = """
module testbench;

    // Parameters
    localparam CLK_PERIOD = 10; // Clock period in time units
    localparam TIMEOUT = 1000;  // Timeout for simulation in time units

    // Inputs
    reg [7:0] data;
    reg clk;

    // Outputs
    wire [31:0] checksum;

    // Instantiate the Unit Under Test (UUT)
    AdlerChecksum uut (
        .data(data),
        .clk(clk),
        .checksum(checksum)
    );

    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD / 2) clk = ~clk; // Toggle clock every half period
    end

    // Stimulus process
    initial begin
        // Initialize inputs
        data = 8'h00;

        // Dump waveform
        $dumpfile("wave_AdlerChecksum.vcd");
        $dumpvars(0, testbench.uut);

        // Test various data inputs
        // Test with a sequence of increasing values
        for (int i = 0; i < 256; i++) begin
            data = i; // Send data from 0 to 255
            #(CLK_PERIOD); // Wait for one clock cycle
        end

        // Test with some edge cases
        data = 8'hFF; // Maximum value
        #(CLK_PERIOD);
        data = 8'h00; // Minimum value
        #(CLK_PERIOD);
        data = 8'h7F; // Midpoint value
        #(CLK_PERIOD);
        data = 8'h80; // Another midpoint value
        #(CLK_PERIOD);

        // Test with random values
        data = 8'h3C; // Random value
        #(CLK_PERIOD);
        data = 8'hA5; // Random value
        #(CLK_PERIOD);
        data = 8'hC3; // Random value
        #(CLK_PERIOD);

        // Finish simulation after timeout
        #(TIMEOUT);
        $finish;
    end

    // Monitor the checksum output
    initial begin
        // Monitor the checksum output
        $monitor("Time: %0t | Data: %h | Checksum: %h", $time, data, checksum);
    end

endmodule
"""
    rtl_code = """
module AdlerChecksum (
input [7:0] data,
input clk,
output reg [31:0] checksum
);

// no parameters needed

// two 16-bit registers, initialized to 1
reg [15:0] s1 = 16'h0001;
reg [15:0] s2 = 16'h0001;

// operations for each data byte
always @ (posedge clk) begin
s1 <= (s1 + data) % 65521;
s2 <= (s2 + s1) % 65521;
end

// compute Adler checksum after all bytes have been processed
always @* begin
checksum = (s2 << 16) + s1;
end

endmodule
"""
    coverage_report ="""

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   GENERAL INFORMATION   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Report generated from CDD file : /tmp/tmprn17w1_5/testbench_AdlerChecksum.v.cdd

* Reported by                    : Module

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   LINE COVERAGE RESULTS   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Module/Task/Function      Filename                 Hit/ Miss/Total    Percent hit
---------------------------------------------------------------------------------------------------------------------
  $root                   NA                         0/    0/    0      100%
  AdlerChecksum           rtl_AdlerChecksum.v        4/    1/    5       80%
---------------------------------------------------------------------------------------------------------------------
  Accumulated                                        4/    1/    5       80%
---------------------------------------------------------------------------------------------------------------------

    Module: AdlerChecksum, File: /tmp/tmprn17w1_5/rtl_AdlerChecksum.v
    -------------------------------------------------------------------------------------------------------------
    Missed Lines

           22:    checksum = ( s2 << 16) +  s1



~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   TOGGLE COVERAGE RESULTS   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                                           Toggle 0 -> 1                       Toggle 1 -> 0
Module/Task/Function      Filename                 Hit/ Miss/Total    Percent hit      Hit/ Miss/Total    Percent hit
---------------------------------------------------------------------------------------------------------------------
  $root                   NA                         0/    0/    0      100%             0/    0/    0      100%
  AdlerChecksum           rtl_AdlerChecksum.v        9/   64/   73       12%             9/   64/   73       12%
---------------------------------------------------------------------------------------------------------------------
  Accumulated                                        9/   64/   73       12%             9/   64/   73       12%
---------------------------------------------------------------------------------------------------------------------

    Module: AdlerChecksum, File: /tmp/tmprn17w1_5/rtl_AdlerChecksum.v
    -------------------------------------------------------------------------------------------------------------
    Signals not getting 100% toggle coverage

      Signal                    Toggle
      ---------------------------------------------------------------------------------------------------------
      checksum                  0->1: 32'h0000_0000
      ......................... 1->0: 32'h0000_0000 ...
      s1                        0->1: 16'h0000
      ......................... 1->0: 16'h0000 ...
      s2                        0->1: 16'h0000
      ......................... 1->0: 16'h0000 ...



~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   COMBINATIONAL LOGIC COVERAGE RESULTS   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                                                            Logic Combinations
Module/Task/Function                Filename                          Hit/Miss/Total    Percent hit
---------------------------------------------------------------------------------------------------------------------
  $root                             NA                                  0/   0/   0      100%
  AdlerChecksum                     rtl_AdlerChecksum.v                 6/  14/  20       30%
---------------------------------------------------------------------------------------------------------------------
  Accumulated                                                           6/  14/  20       30%
---------------------------------------------------------------------------------------------------------------------

    Module: AdlerChecksum, File: /tmp/tmprn17w1_5/rtl_AdlerChecksum.v
    -------------------------------------------------------------------------------------------------------------
    Missed Combinations  (* = missed value)

      =========================================================================================================
       Line #     Expression
      =========================================================================================================
             16:     s1 <= ( s1 + data) % 65521
                           |----1-----|        
                           |--------2---------|

        Expression 1   (2/4)
        ^^^^^^^^^^^^^ - +
         LR | LR | LR | LR 
        =00=|=01=|=10=|=11=
         *    *          

        Expression 2   (1/2)
        ^^^^^^^^^^^^^ - %
         E | E 
        =0=|=1=
         *    

      =========================================================================================================
       Line #     Expression
      =========================================================================================================
             17:     s2 <= ( s2 +  s1) % 65521
                           |----1----|        
                           |--------2--------|

        Expression 1   (1/4)
        ^^^^^^^^^^^^^ - +
         LR | LR | LR | LR 
        =00=|=01=|=10=|=11=
         *    *    *     

        Expression 2   (1/2)
        ^^^^^^^^^^^^^ - %
         E | E 
        =0=|=1=
         *    

      =========================================================================================================
       Line #     Expression
      =========================================================================================================
             21:    @*

        Expression 1   (0/1)
        ^^^^^^^^^^^^^ - @*
         * Event did not occur

      =========================================================================================================
       Line #     Expression
      =========================================================================================================
             22:    checksum = ( s2 << 16) +  s1
                               |----1----|      
                               |-------2-------|

        Expression 1   (0/2)
        ^^^^^^^^^^^^^ - <<
         E | E 
        =0=|=1=
         *   *

        Expression 2   (0/4)
        ^^^^^^^^^^^^^ - +
         LR | LR | LR | LR 
        =00=|=01=|=10=|=11=
         *    *    *    *



~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   FINITE STATE MACHINE COVERAGE RESULTS   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                                               State                             Arc
Module/Task/Function      Filename                Hit/Miss/Total    Percent Hit    Hit/Miss/Total    Percent hit
---------------------------------------------------------------------------------------------------------------------
  $root                   NA                        0/   0/   0      100%            0/   0/   0      100%
  AdlerChecksum           rtl_AdlerChecksum.v       0/   0/   0      100%            0/   0/   0      100%
---------------------------------------------------------------------------------------------------------------------
  Accumulated                                       0/   0/   0      100%            0/   0/   0      100%

"""

    output = enhancer.forward({
        "rtl_code": rtl_code,
        "testbench": testbench,
        "coverage_report": coverage_report
    })

    print(output["testbench"])



if __name__ == '__main__':
    main()
