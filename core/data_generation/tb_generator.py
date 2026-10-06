"""Generates the testbench using knowledge from playbook and previous reflections.
"""
import os
import re 
import sys 
import argparse 

from langchain_core.prompts import ChatPromptTemplate

from core.data_generation.state import DataGenState
from core.data_generation.config.config import FlowConfigs, LMConfigs
from core.train.prompts import SYS_PROMPT, USER_PROMPT_SECTIONED_TEMPLATE
from core.data_generation.utils import _strip_testbench, extract_testbench_code

from dotenv import load_dotenv
load_dotenv()


_TB_GENERATOR_SYS_PROMPT = """
You are an expert verification prompt designer.  Your task is to generate a {language} testbench for {top_module}. The testbench module must be named "testbench".
The instance for {top_module} must be named uut. 

- complete within a reasonable sim time (specify a timeout)
- Give full testbench implementation; Don't add comment placeholders. 
- Your objective is to maximize coverage. Reason first about suitable testing plan then generate the testbench. 
- Don't declare integer/reg/or wire variables inside initial begin blocks.
- Follow the requested testbench style exactly.
- Do not generate any VCD or waveform dumping logic such as `$dumpfile` or `$dumpvars`.
- {style_requirements}
- For very large inputs, don't use binary, use hexadecimal or a compact number representation
"""

_TB_GENERATOR_USER_PROMPT = """
[Input]
{natural_spec}

{rtl_code}

[TestPlan]
{test_plan}

Reason first about how to maximize coverage then output the testbench inside a fenced code block as follows:

```testbench
<testbench code here>
```
"""


_TB_GENERATOR_USER_PROMPT_WO_TESTPLAN = """
[Input]
{natural_spec}

{rtl_code}

Reason first about how to maximize coverage then output the testbench inside a fenced code block as follows:

```testbench
<testbench code here>
```
"""


class TBGenerator():

    def __init__(self, config):
        self.config = config 

        if self.config.data_gen_mode:
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", str(_TB_GENERATOR_SYS_PROMPT)),
                ("user", _TB_GENERATOR_USER_PROMPT),
            ])
        else:
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", str(SYS_PROMPT)),
                ("user", USER_PROMPT_SECTIONED_TEMPLATE),
            ])

        self.llm = config.lm_config.tb_generator_llm_model
        self.tb_generator = self.prompt | self.llm 
 
    def _get_style_requirements(self):
        return (
            "Generate a stimulus-only testbench. Do not add assertions, scoreboards, "
            "golden reference models, $fatal, $error, or correctness-checking if/else blocks. "
            "Keep only the DUT instantiation, clocks/resets when applicable, deterministic stimulus, "
            "and a timeout/finish path."
        )

    def forward(self, state: DataGenState):
        rtl_code =  state.get("rtl_code")
        natural_spec =  state.get("natural_spec")
        language =  state.get("language")
        test_plan = state.get("test_plan")
        top_module =  state.get("top_module")

        if self.config.data_gen_mode:
            response = self.tb_generator.invoke({
                "rtl_code": rtl_code, 
                "natural_spec": natural_spec,
                "top_module": top_module,
                "test_plan": test_plan,
                "language": language,
                "style_requirements": self._get_style_requirements()
            })
        else:
            context = ""
            response = self.tb_generator.invoke({
                "rtl_code": rtl_code, 
                "natural_spec": natural_spec,
                "context": context
            })
        
        reasoning_trace = _strip_testbench(response.content)
        testbench = self._parse_testbench(response.content)

        return {"testbench": self._strip_waveform_dumping(testbench), "tb_generator_reasoning_trace": reasoning_trace}


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

    tb_generator = TBGenerator(
        config=config
    )

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

    top_module = "AdlerChecksum"
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

    test_plan = """
[STIMULUS STRATEGY]
- Nominal/throughput: Send a sequence of 8-bit data values (0x00 to 0xFF) at a steady clock rate to observe normal operation.
- Boundary conditions: Test with data values of 0x00 (zero), 0xFF (all ones), and alternating patterns (0xAA, 0x55).
- Corner interactions: Send a sequence of maximum (0xFF) followed by minimum (0x00) values to observe behavior under rapid changes.
- Reset sequences: Apply a cold reset (initializing both sums) and warm reset (re-initializing sums while clocking) to verify state recovery.
- Back-to-back transactions: Send multiple data bytes in quick succession to check for proper checksum calculation without stalls.
- Stalls/ready deassertions: Introduce delays in the clock signal to simulate stalls and observe checksum stability.
- One-hot vs multi-hot: Test with single data values and combinations of values to ensure proper checksum calculation.
- Sign/overflow/underflow paths: Test values that approach the modulo limit (e.g., 0xFF + 0xFF) to check for correct wrapping behavior.
- X/Z propagation: Introduce X and Z states in the data input to verify that the checksum output remains stable and does not propagate X/Z.
- Protocol timing permutations: Vary the clock period to test valid/ready jitter and ensure the checksum is computed correctly.
- Constrained-random: Generate random 8-bit data values with constraints to avoid illegal combinations (e.g., no simultaneous resets). 
- Directed tests: Specifically target the modulo operation with values that cause saturation (e.g., values that sum to 65521) and check the resulting checksum.
- Test length & termination: Set a global timeout of 1000ns and a per-scenario budget of 100ns to avoid hanging simulations.

[CHECKING STRATEGY]
- Golden/reference behavior: Implement a reference function that computes the Adler checksum based on the SPEC/RTL logic, ensuring it matches the expected output for given inputs.
- Self-checking: Use `assert` statements to verify that the computed checksum matches the expected value, including scenario name in the error message.
- Protocol assertions: Ensure that the checksum output is stable after reset and that no X states are present on outputs after reset deassertion.
- Scoreboard rules: Maintain FIFO ordering of input data and ensure that the checksum is computed with the correct latency.

[COVERAGE PLAN]
- Functional coverage: Create coverpoints for input data values (0x00, 0xFF, etc.), boundary conditions, and checksum output values. Track transitions of intermediate sums A and B.
- Crosses: Monitor interactions between input data categories and checksum output, as well as between input patterns and reset conditions.
- Illegal bins: Identify and justify unreachable states, such as simultaneous resets and invalid data combinations.
- Code coverage closure guidance: Focus on rare branches in the modulo operations and ensure all paths are exercised by adjusting input values and sequences.
- Targets: Aim for functional coverage ≥ 99%, Statement/Branch/Toggle coverage ≥ 99%. Exclude any unreachable states that are not relevant to the checksum calculation.

[ASSUMPTIONS]
- The clock signal is assumed to be active high.
- Reset behavior is assumed to initialize both sums to 1.
- The module does not have any handshake signals; data is assumed to be valid on every clock cycle.
- The checksum output is expected to be stable immediately after processing the last data byte.
"""

    output = tb_generator.forward({
        "rtl_code": rtl_code,
        "test_paln": test_plan,
        "language": language,
        "top_module": top_module
    })


    print(output["testbench"])


if __name__ == '__main__':
    main()
