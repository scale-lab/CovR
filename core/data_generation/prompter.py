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

"""Generates Natural Language test plan for Testbench Generation 
"""

import re 
import argparse 

from langchain_core.prompts import ChatPromptTemplate

from core.data_generation.agent import Agent 
from core.data_generation.state import DataGenState
from core.data_generation.config.config import FlowConfigs, LMConfigs

from dotenv import load_dotenv
load_dotenv()


PROMPTER_SYS_PROMPT = """
You are an expert verification strategist.

Your job is to produce a DUT-specific TEST PLAN for a separate model that will generate the final {language} testbench.

The test plan must maximize meaningful verification coverage while staying grounded in the provided RTL and natural-language specification.
The downstream generator will produce a stimulus-only testbench, so your plan must emphasize coverage-driving stimulus and must not depend on self-checking infrastructure.

## Core task
First infer the DUT style from the RTL and specification:
- combinational or sequential
- arithmetic/datapath, FSM/control, or protocol/streaming
- whether reset exists
- whether handshake or transaction semantics exist
- whether there are important edge cases such as overflow, underflow, modulo behavior, boundary indexing, state transitions, sign handling, or rare branches

Then produce a focused test plan for this DUT only.

## Hard rules
- Do not invent ports, signals, states, resets, handshakes, or protocols not present in the RTL/spec.
- If the DUT has no reset, do not include reset scenarios.
- If the DUT has no handshake/protocol behavior, do not include ready/valid, start/done, backpressure, or transaction-monitor guidance.
- Prefer exact signal names, widths, constants, and behavioral details from the RTL/spec.
- Focus on observable behavior at the DUT interface unless internal access is explicitly justified.
- Keep the plan useful for a downstream testbench generator, not as implementation code.
- Focus on coverage-oriented stimulus design rather than checkers, scoreboards, assertions, or reference models.

## Output format
Return exactly one fenced block labeled `tbgen-prompt`.

Inside that fenced block, output exactly these sections in this order:

[STIMULUS STRATEGY]
[OBSERVABILITY STRATEGY]
[COVERAGE PLAN]

Do not add any extra sections.

## What each section should contain

[STIMULUS STRATEGY]
Describe the most important scenario families to test for this DUT.
Include only scenarios that apply, such as:
- nominal behavior
- boundary values
- corner-case combinations
- arithmetic edge cases
- temporal/state-evolution scenarios
- directed cases for hard-to-hit logic
- deterministic random testing if useful
- timeout/termination guidance

For each major scenario family, explain what it is intended to exercise.
Prioritize stimulus that is likely to close line, branch, toggle, condition, and FSM coverage.
For FSM-like DUTs, emphasize state reachability, transition sequences, rare guards, recovery paths, and varied dwell times.

[OBSERVABILITY STRATEGY]
Describe any lightweight, non-checking observability that can help debug or understand coverage progress without turning the testbench into a self-checking one.
Include only applicable items, such as:
- structured scenario naming
- concise progress displays
- selective signal logging
- comments that mark why a scenario exists
- simple end-of-test completion reporting

Do not require or suggest:
- assertions
- scoreboards
- golden/reference models
- self-checking comparisons
- failure messages such as "Error:"
- VCD or waveform dumping

[COVERAGE PLAN]
Describe the main coverage targets for this DUT.
Include only applicable items, such as:
- input partitions and boundary bins
- operation/mode/state coverage
- transition or temporal coverage
- branch/condition/toggle-oriented scenarios
- meaningful crosses between important stimulus dimensions

Map likely hard-to-hit RTL behavior to concrete stimulus ideas.
When possible, explicitly connect each uncovered behavior class to a stimulus pattern or temporal sequence.

## Style
- Be specific, compact, and grounded.
- Avoid generic verification boilerplate.
- Avoid mentioning categories that do not apply.
- Keep the final plan concise, usually 200-500 words.
"""


_PROMPTER_USER_PROMPT = """
Generate a DUT-specific verification test plan for the following design.

RTL Code:
{rtl_code}

Natural-Language Specification:
{natural_spec}

Important:
- Tailor the plan only to behavior supported by the RTL and spec.
- Do not include reset, protocol, or handshake scenarios unless they are actually present.
- Focus on the most important coverage-driving scenarios for a stimulus-only testbench.
"""


PROMPT_TEMPLATE = lambda rtl_code, natural_spec, language, top_module, plan: f"""
You are an expert verification prompt designer.  Your task is to generate a {language} testbench for {top_module}. The testbench module must be named "testbench".
The instance for {top_module} must be named uut. 

[Context]
{natural_spec}

{rtl_code}

{plan}

- complete within a reasonable sim time (specify a timeout)
"""

class Prompter():

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", str(PROMPTER_SYS_PROMPT)),
            ("user", _PROMPTER_USER_PROMPT),
        ])

        self.llm = config.lm_config.prompter_llm_model
        self.prompter = self.prompt | self.llm 

    def forward(self, state: DataGenState):
        if self.config.data_gen_mode:
            specs = state.get("natural_spec")
            rtl_code = state.get("rtl_code")
            language = state.get("language")
            top_module = state.get("top_module")

            response = self.prompter.invoke({
                "natural_spec": specs, 
                "rtl_code": rtl_code, 
                "language": language
            })
            
            test_plan = self._parse_tbgen_prompt(response.content) or response.content.strip()
            return {"test_plan": test_plan}
        else:
            return {"test_plan": None}
            
    def _parse_tbgen_prompt(self, content):
        pattern = r"```tbgen-prompt\s*(.*?)\s*```"
        match = re.search(pattern, content, flags=re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""


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
            tb_generator_lm = model
        )
    )
    prompter = Prompter(config=config) 
    
    specs = """
Write a Verilog module named `AdlerChecksum` that computes a 32-bit checksum for incoming 8-bit data bytes. The module should have the following interface:
`input [7:0] data`, `input clk`, `output reg [31:0] checksum`.

The checksum is based on two 16-bit intermediate sums, commonly referred to as A and B. Initialize both intermediate sums to a value of 1.

On each positive edge of the `clk` signal, process the incoming `data` byte according to the following steps:
1. Update sum A by adding the current `data` byte to its current value. The result must be taken modulo 65521.
2. Update sum B by adding the *newly computed* value of sum A to its current value. This result must also be taken modulo 65521.

The final 32-bit `checksum` is formed by combining the current values of sum A and sum B, where sum B constitutes the most significant 16 bits and sum A constitutes the least significant 16 bits. This `checksum` output should combinatorially reflect the current state of the intermediate sums.

Your implementation should use registers to store the intermediate sums and assign the final checksum based on their current values.    
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
    language = "SystemVerilog"   # SystemVerilog or cocotb

    output = prompter.forward({"natural_spec": specs, "rtl_code": rtl_code, "language": language, "top_module": "AdlerChecksum"})

    print(output["test_plan"])


if __name__ == '__main__':
    main()
