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



"""Runs the refiner flow to make sure that testbenches are self checking with Error: print statements
"""

import re 
import os
import argparse 

from enum import Enum
from langgraph.graph import START, END, StateGraph

from core.data_generation.fixer import Fixer
from core.data_generation.state import DataGenState
from core.data_generation.compiler import Compiler
from core.data_generation.prompter import Prompter
from core.data_generation.reporter import Reporter
from core.data_generation.self_checking.checker import Checker
from core.data_generation.self_checking.refiner import PrintRefiner, ReferenceRefiner
from core.data_generation.self_checking.state import RefinerState
from core.data_generation.config.config import FlowConfigs, LMConfigs

from dotenv import load_dotenv
load_dotenv()



class RefinerFlowNodes(Enum):
    Checker = "Checker"
    PrintRefiner = "PrintRefiner"
    ReferenceRefiner = "ReferenceRefiner"
    Compiler = "Compiler"
    Fixer = "Fixer"
    Reporter = "Reporter"
    

class Runner():


    def __init__(self, config):
        self.config = config 
        self._start_at = None

        self.checker = Checker(
            config=config
        )

        self.print_refiner = PrintRefiner(
            config=config
        )

        self.ref_refiner = ReferenceRefiner(
            config=config
        )

        self.compiler = Compiler(
            config=config
        )

        self.fixer = Fixer(
            config=config
        )

        self.reporter = Reporter(
            config=config 
        )

        self.builder = StateGraph(RefinerState)
        
        self.builder.add_node(
            RefinerFlowNodes.Checker.value, 
            self.checker.forward
        )

        self.builder.add_node(
            RefinerFlowNodes.PrintRefiner.value, 
            self.print_refiner.forward
        )

        self.builder.add_node(
            RefinerFlowNodes.ReferenceRefiner.value, 
            self.ref_refiner.forward
        )


        self.builder.add_node(
            RefinerFlowNodes.Compiler.value, 
            self.compiler.forward
        )

        self.builder.add_node(
            RefinerFlowNodes.Fixer.value, 
            self.fixer.forward
        )

        self.builder.add_node(
            RefinerFlowNodes.Reporter.value, 
            self.reporter.forward
        )

        self.builder.add_conditional_edges(
            START,
            self._route_start,
            [
                RefinerFlowNodes.Checker.value,
                RefinerFlowNodes.PrintRefiner.value,
                RefinerFlowNodes.ReferenceRefiner.value,
                RefinerFlowNodes.Compiler.value,
                RefinerFlowNodes.Fixer.value,
                RefinerFlowNodes.Reporter.value,
            ]
        )
        self.builder.add_edge(RefinerFlowNodes.PrintRefiner.value, RefinerFlowNodes.Compiler.value)
        self.builder.add_edge(RefinerFlowNodes.ReferenceRefiner.value, RefinerFlowNodes.Compiler.value)
        self.builder.add_edge(RefinerFlowNodes.Fixer.value, RefinerFlowNodes.Compiler.value)
        self.builder.add_edge(RefinerFlowNodes.Reporter.value, END)

        self.builder.add_conditional_edges(
            RefinerFlowNodes.Checker.value, 
            self._should_refine, 
            [RefinerFlowNodes.PrintRefiner.value, RefinerFlowNodes.ReferenceRefiner.value, END]
        )

        self.builder.add_conditional_edges(
            RefinerFlowNodes.Compiler.value, 
            self._should_fix, 
            [RefinerFlowNodes.Fixer.value, RefinerFlowNodes.Reporter.value, END]
        )

        self.graph = self.builder.compile()
        png_graph = self.graph.get_graph().draw_mermaid_png()
        os.makedirs("./output", exist_ok=True)
        with open(os.path.join("output/", "refiner_flow.png"), "wb") as f:
            f.write(png_graph)

    def forward(self, state: RefinerState, start_at = None):
        self._start_at = start_at
        output = self.graph.invoke(state, {"recursion_limit":  self.config.graph_recursion_limit,  "configurable": {"start_at": start_at}})  
        return output

    def _route_start(self, state: RefinerState):
        if self._start_at is None:
            return RefinerFlowNodes.Checker.value

        if isinstance(self._start_at, RefinerFlowNodes):
            return self._start_at.value

        return str(self._start_at)

    def _should_refine(self, state: RefinerState):
        is_self_checking = state.get("is_self_checking")
        is_print_error = state.get("is_print_error")
        refine_iters = state.get("refine_iters", 0)

        if is_self_checking and not is_print_error and refine_iters <= self.config.max_refine_iters:
            return RefinerFlowNodes.PrintRefiner.value 
        
        if not is_self_checking and not is_print_error and refine_iters <= self.config.max_refine_iters:
            return RefinerFlowNodes.ReferenceRefiner.value 
        
        return END 

    def _should_fix(self, state: RefinerState):
        testbench_runs = state.get("testbench_runs")
        fix_iters = state.get("fix_iters", 0)
        if testbench_runs:
            return RefinerFlowNodes.Reporter.value
        elif  fix_iters >= self.config.max_refine_iters:
            return END
        else:
            return RefinerFlowNodes.Fixer.value


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
            enhancer_lm = model ,
            refiner_lm = model,
            checker_lm = model 
        ),
        max_refine_iters=5,
        max_enhance_iters=5,
    )
 
    natural_spec = """
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
    input        clk,         // clock
    input        rst_n,       // active-low synchronous reset
    input  [7:0] data,        // input data byte
    output reg [31:0] checksum
);

    // two 16-bit checksum registers
    reg [15:0] s1;
    reg [15:0] s2;

    // synchronous reset: clears s1 and s2 to 1
    always @(posedge clk) begin
        if (!rst_n) begin
            s1 <= 16'h0001;
            s2 <= 16'h0001;
        end
        else begin
            s1 <= (s1 + data) % 16'd65521;
            s2 <= (s2 + s1) % 16'd65521;
        end
    end

    // continuously compute the final Adler checksum
    always @* begin
        checksum = (s2 << 16) + s1;
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
    runner = Runner(
        config=config 
    )

    output = runner.forward({
        "rtl_code": rtl_code,
        "natural_spec": natural_spec,
        "testbench": testbench ,
        "language": "Verilog"
    })

    print(output)

if __name__ == '__main__':
    main()
