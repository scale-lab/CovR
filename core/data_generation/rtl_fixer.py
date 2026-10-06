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

"""RTL syntax error fixer
"""
import re 
import argparse 

from langchain_core.prompts import ChatPromptTemplate

from core.data_generation.agent import Agent 
from core.data_generation.state import DataGenState
from core.data_generation.config.config import FlowConfigs, LMConfigs

from dotenv import load_dotenv
load_dotenv()

_RTL_FIXER_SYS_PROMPT = """
You are an expert Verilog and SystemVerilog compiler and debugger.

Your job:
- Take as input a block of RTL code and the compilation / syntax error messages.
- Analyze the error messages to determine what is wrong in the code.
- Provide a **fixed version of the RTL code** that compiles without syntax errors while keeping the original logic intact.

Output rules:
1. THINK through the error first, then output the corrected Verilog.
2. Do **NOT** change the design intent or optimize functionality unless required to remove the error.
3. Keep comments, formatting, and existing signal names where possible.
4. Wrap the corrected code ONLY inside a markdown-style code-fence:  
```verilog
   ...fixed code here...
```

RTL Code:
{rtl_code}
Error:
{rtl_error}
"""

class RTLFixer():


    def __init__(self, config):
        super().__init__()
        self.config = config

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", str(_RTL_FIXER_SYS_PROMPT)),
            ("user", ""),
        ])

        self.llm = config.lm_config.fixer_llm_model
        self.fixer = self.prompt | self.llm  

   
    def forward(self, state: DataGenState):
        rtl_code =  state.get("rtl_code")
        rtl_error =  state.get("rtl_error")
        rtl_fix_iters = state.get("rtl_fix_iters", 0)
        
        response = self.fixer.invoke({
            "rtl_code": rtl_code, 
            "rtl_error": rtl_error
        })
        
        fixed_rtl = self._parse_verilog(response.content)

        return {"rtl_code": fixed_rtl, "rtl_fix_iters": rtl_fix_iters + 1}

    
    def _parse_verilog(self, content):
        if not content:
            return ""
        pattern = re.compile(
            r"```(?:\s*verilog)?\s*(.*?)```",
            re.DOTALL | re.IGNORECASE
        )
        match = pattern.search(content)
        if match:
            return match.group(1).strip()
        else:
            return content.strip() 


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
            enhancer_lm = model
        )
    )
    fixer = RTLFixer(config=config) 

    rtl_code = """
`timescale 1ns/1ps

module opt_model (
  output reg zero
);
  assign zero = 1'b0;
endmodule
"""
    output = fixer.forward({
        "rtl_error": "Invalid reg decleration",
        "rtl_code": rtl_code    
    })

    print(output["rtl_code"])



if __name__ == '__main__':
    main()
