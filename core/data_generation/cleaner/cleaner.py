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

"""
Removes all self-checking behavior from a testbench:
- assertions
- $fatal / $error / $stop
- conditional checking blocks
- error display statements
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


_STRIP_SYS_PROMPT = """
You are an expert verification engineer.

Your job is to REMOVE all self-checking behavior from a SystemVerilog testbench.
"""

_STRIP_USER_PROMPT = """
Your task is to convert this testbench into a stimulus-only testbench.

REMOVE:
- assert statements (including assert property)
- $fatal, $error, $stop
- any if/else blocks that check correctness
- any display statements indicating failure (e.g., "Error:", "FAILED", "Timeout" etc.)
- any vcd dump files 
- any monitors 
- Any golden references or expected output values

KEEP:
- stimulus generation
- clock/reset logic
- DUT instantiation

DO NOT:
- modify stimulus values
- add new logic
- change structure unnecessarily

Return ONLY the cleaned testbench.

[Testbench]
{testbench}
"""


class StripSelfCheckingRefiner:
    def __init__(self, config):
        self.config = config

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", _STRIP_SYS_PROMPT),
            ("user", _STRIP_USER_PROMPT),
        ])

        self.llm = config.lm_config.refiner_llm_model
        self.refiner = self.prompt | self.llm

    def forward(self, state):
        testbench = state.get("testbench")

        response = self.refiner.invoke({
            "testbench": testbench
        })

        return {"testbench": self._parse(response.content)}

    def _parse(self, content):
        if not content:
            return ""

        pattern = re.compile(
            r"```(?:\s*(?:systemverilog|verilog))?\s*(.*?)```",
            re.DOTALL | re.IGNORECASE
        )

        matches = pattern.findall(content)
        return matches[-1].strip() if matches else content.strip()



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

    refiner = StripSelfCheckingRefiner(
        config=config
    )
 

 
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

    output = refiner.forward({
        "testbench": testbench
    })

    print(output)


    
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

    output = refiner.forward({
        "testbench": testbench,
        "rtl_code": rtl_code,
        "natural_spec": natural_spec
    })

    print(output["testbench"])
   

if __name__ == '__main__':
    main()
