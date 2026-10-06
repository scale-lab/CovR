"""Problem Complexity Analyzer
"""
import os 
import re 
import json
import argparse 


from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

from core.data_generation.agent import Agent 
from core.data_generation.state import DataGenState
from core.data_generation.config.config import FlowConfigs, LMConfigs
from core.data_generation.self_checking.state import RefinerState

from dotenv import load_dotenv
load_dotenv()


import json
import argparse
from typing import List

from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

from core.data_generation.config.config import FlowConfigs, LMConfigs


_ANALYZER_SYS_PROMPT = """
You are an expert in hardware verification and coverage-driven testbench design.

Your task is to analyze how difficult it would be to generate a high-coverage stimulus-only SystemVerilog testbench for a given RTL design problem.

You must judge complexity based on how hard it is to achieve high functional coverage, not how hard the RTL is to implement.

Focus on factors such as:
1. State-space complexity
2. Number of modes / branches / corner cases
3. Temporal depth (whether behaviors require long sequences)
4. Input interaction complexity
5. Reset / initialization sensitivity
6. Hidden or difficult-to-observe internal behavior
7. Whether rare sequences are needed to trigger important behaviors
8. Whether high coverage requires targeted directed stimulus rather than simple random testing
9. Whether protocol correctness or sequencing constraints make coverage harder
10. Whether the natural-language spec leaves ambiguity that makes good stimulus design harder

You should produce a fair and practical estimate of testbench-generation difficulty.

Scoring guidance:
- 1-2: Very easy
- 3-4: Easy
- 5-6: Medium
- 7-8: Hard
- 9-10: Very hard

Important:
- Judge difficulty of generating a high-coverage testbench, not difficulty of writing the DUT.
- Prefer concise but concrete reasoning.
- Assume the testbench is stimulus-only (no assertions/checkers/scoreboards).
- Return structured output only.
"""


_ANALYZER_USER_PROMPT = """
Analyze the following problem and estimate how difficult it is to generate a high-coverage stimulus-only testbench.

Natural language specification:
{natural_spec}

RTL code:
{rtl_code}

Existing testbench (if any):
{testbench}

Assign:
1. An overall complexity level
2. An overall complexity score from 1 to 10
3. Subscores for key difficulty dimensions
4. A short explanation of why this problem is easy or hard for coverage-driven testbench generation
5. Suggested stimulus strategies that would likely be needed to reach high coverage

Be specific about what makes coverage hard.
"""


class ComplexityAnalysis(BaseModel):
    complexity_level: str = Field(
        description="One of: easy, medium, hard, very_hard"
    )
    complexity_score: int = Field(
        description="Overall complexity score from 1 to 10"
    )

    state_space_score: int = Field(
        description="Difficulty due to state-space size, 1-10"
    )
    branching_score: int = Field(
        description="Difficulty due to branches/modes/corner cases, 1-10"
    )
    temporal_depth_score: int = Field(
        description="Difficulty due to long sequences or multi-cycle dependencies, 1-10"
    )
    input_interaction_score: int = Field(
        description="Difficulty due to interaction among inputs and constraints, 1-10"
    )
    observability_score: int = Field(
        description="Difficulty due to hidden behavior or hard-to-exercise logic, 1-10"
    )
    directed_stimulus_need_score: int = Field(
        description="How much targeted/directed stimulus is needed instead of naive random stimulus, 1-10"
    )

    key_difficulty_factors: List[str] = Field(
        description="Main reasons why high-coverage testbench generation is difficult"
    )
    explanation: str = Field(
        description="Short paragraph explaining the assigned complexity"
    )
    suggested_stimulus_strategies: List[str] = Field(
        description="Stimulus strategies that would likely improve coverage"
    )


class Analyzer:
    def __init__(self, config):
        self.config = config

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", _ANALYZER_SYS_PROMPT),
            ("user", _ANALYZER_USER_PROMPT),
        ])

        self.llm = config.lm_config.refiner_llm_model

        if not hasattr(self.llm, "with_structured_output"):
            raise ValueError(
                "refiner_llm_model must support with_structured_output(...)"
            )

        self.chain = self.prompt | self.llm.with_structured_output(ComplexityAnalysis)

    def forward(self, state):
        testbench = state.get("testbench", "")
        rtl_code = state.get("rtl_code", "")
        natural_spec = state.get("natural_spec", "")

        if not rtl_code and not natural_spec:
            raise ValueError("At least one of rtl_code or natural_spec must be provided.")

        result = self.chain.invoke({
            "testbench": testbench or "None provided.",
            "rtl_code": rtl_code or "None provided.",
            "natural_spec": natural_spec or "None provided.",
        })

        if isinstance(result, BaseModel):
            return result.dict()
        return result



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

    analyzer = Analyzer(
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

    natural_spec = ""
    
    output = analyzer.forward({
        "testbench": testbench,
        "rtl_code": rtl_code,
        "natural_spec": natural_spec
    })

    print(output)


if __name__ == '__main__':
    main()