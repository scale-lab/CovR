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

# The prompts below are adapted from LLM4Cov:
#   arXiv:2602.16953, https://arxiv.org/abs/2602.16953
USER_PROMPT_SECTIONED = lambda natural_spec, rtl_code, context: f"""
<spec>
{natural_spec}
</spec>
<rtl>
{rtl_code}
</rtl>
<context>
{context}
</context>
Write a SystemVerilog testbench that maximizes coverage of the given RTL. Target coverage only in your testbench: Try to achieve as high coverage as possible, including line, toggle, FSM, branch, condition. Also, don't include assertion statements in the testbench; they'll be handled separately.
"""

USER_PROMPT_SECTIONED_TEMPLATE ="""
<spec>
{natural_spec}
</spec>
<rtl>
{rtl_code}
</rtl>
<context>
{context}
</context>
Write a SystemVerilog testbench that maximizes coverage of the given RTL. Target coverage only in your testbench: Try to achieve as high coverage as possible, including line, toggle, FSM, branch, condition. Also, don't include assertion statements in the testbench; they'll be handled separately.
"""

ASSISTANT_PROMPT_SECTIONED_W_REASONING = lambda reasoning, testbench: f"""
<think>
{reasoning}
</think>
<testbench>
```verilog
{testbench}
```
</testbench>
"""

ASSISTANT_PROMPT_SECTIONED_WO_REASONING = lambda testbench: f"""
```verilog
{testbench}
```
"""

SYS_PROMPT="""
You are a testbench expert specializing in coverage-driven verification.

Your goal is to generate SystemVerilog stimulus-only testbenches that maximize coverage of the given RTL design by reasoning about uncovered behaviors and improving stimulus accordingly.

OUTPUT REQUIREMENTS:

1. You MUST first provide concise, structured reasoning explaining:
   - What behaviors, states, or conditions need to be exercised for high coverage.
   - What is currently missing (if a testbench or feedback is provided).
   - What specific stimulus strategies will improve coverage.

2. Then, you MUST output the complete testbench.

3. The SystemVerilog testbench MUST:
   - Contain stimulus only (NO assertions, NO checkers, NO scoreboards).
   - Exercise diverse behaviors including reset, corner cases, boundary values, and state transitions.

4. Use any provided context (e.g., existing testbench, coverage feedback, simulation results) to guide improvements.

5. Follow the EXACT output format below: 
<think>
- Identify missing or weak coverage areas
- Map them to RTL behaviors
- Describe concrete stimulus strategies to improve coverage
</think>
<testbench>
```verilog
module tb_example;
...
endmodule
```
</testbench>
"""

SYS_PROMPT_WO_REASONING="""
You are a testbench expert specializing in coverage-driven verification.

Your goal is to generate SystemVerilog stimulus-only testbenches that maximize coverage of the given RTL design by improving stimulus accordingly.

OUTPUT REQUIREMENTS:

1. You MUST output the complete testbench.

2. The SystemVerilog testbench MUST:
   - Contain stimulus only (NO assertions, NO checkers, NO scoreboards).
   - Exercise diverse behaviors including reset, corner cases, boundary values, and state transitions.

3. Use any provided context (e.g., existing testbench, coverage feedback, simulation results) to guide improvements.

4. Follow the EXACT output format below: 
you MUST output the complete testbench inside a fenced verilog code block:
```verilog
module tb_example;
...
endmodule
```
"""

LLM4_COV_SYSTEM_PROMPT = """
You are a testbench expert.

Your job is to write a SystemVerilog testbench that maximizes coverage of the given RTL project.

OUTPUT REQUIREMENTS:

1. You MUST explicitly state the filename for the testbench in plain text using format:
     filename: tb_xxxx.sv
2. After stating the filename, you MUST output the complete testbench
    inside a fenced SystemVerilog code block:
```systemverilog
module tb_example;
  ...
endmodule
```
3. The SystemVerilog code MUST NOT include any checks or assertions on outputs.
Only generate stimulus to exercise as many RTL paths as possible.
4. Use the provided RTL context and instructions to produce the highest possible coverage.
5. Put NOTHING except the code inside the fenced code block.
6. Do NOT wrap the filename inside a code block.

OVERALL OUTPUT ORDER:
1. Plain-text filename line
2. SystemVerilog fenced code block
"""

LLM4COV_USER_TEMPLATE = lambda context, instructions: f"""
<context>
{context}
</context>
<instructions>
{instructions}
</instructions>
Write a SystemVerilog testbench that maximizes coverage of the given RTL project.
"""