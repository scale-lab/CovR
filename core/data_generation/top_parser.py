"""Fixes any syntax errors in the testbench 
"""
import re 
import argparse 

from langchain_core.prompts import ChatPromptTemplate

from core.data_generation.agent import Agent 
from core.data_generation.state import DataGenState
from core.data_generation.config.config import FlowConfigs, LMConfigs

from dotenv import load_dotenv
load_dotenv()

_PARSER_SYS_PROMPT = """
You are an expert Verilog  engineer.
Your job is to output the top module name of the input verilog code. 

## Output:
Just output the top module name. 

```top_module
<top-module-name>
```
"""

_PARSER_USER_PROMPT = """
Output the top module name of the following verilog code: 
{rtl_code}
"""


class TopParser():
    
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", str(_PARSER_SYS_PROMPT)),
            ("user", _PARSER_USER_PROMPT),
        ])

        self.llm = config.lm_config.fixer_llm_model
        self.parser = self.prompt | self.llm 
 


    def forward(self, state: DataGenState):
        rtl_code =  state.get("rtl_code")
        
        response = self.parser.invoke({
            "rtl_code": rtl_code
        })
        
        top_module = self._parse_top_module(response.content)

        return top_module

    def _parse_top_module(self, content):
        if not content:
            return ""
        pattern = re.compile(
            r"```(?:\s*top_module)?\s*(.*?)```",
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
            fixer_lm = model,
            tb_generator_lm= model,
            enhancer_lm= model
        )
    )
    parser = TopParser(config=config) 

    output = parser.forward({
        "rtl_code": rtl_code    
    })

    print(output)

if __name__ == '__main__':
    main()