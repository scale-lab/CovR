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

"""Makes sure that the testbench is self checking
"""

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


class CheckerOutput(BaseModel):
    is_self_checking: bool = Field(description="True if the testbench is self-checking; False otherwise.")
    error_statements: bool = Field(description=('True if *all* error-signaling print statements begin with "Error:" for every testcase; False otherwise.'))

_CHECKER_SYS_PROMPT = """
You are an expert verification engineer, responsible for analyzing testbenches.
"""

_CHECKER_USER_PROMPT = """
Your task is to analyze the testbench and determine the following: 

- If the testbench is self-checking, i.e has golden reference outputs for each testcase. 
- If the testbench signals errors in testcases with Error statements that start with "Error:". This must be true for all the testcases applied in the testbench.

NOTE: All print statements that signal test case failure must start with "Error:". First, count how many print statements are in the testbench are used to signal test case failure. 
Then, count how many of these start with "Error:" 

Reason first about the given testbench then output your analysis result in the following json format: 

```json
{{
    "is_self_checking": <True if the testbench is self checking | False otherwise>, 
    "error_statements": <True if the testbench signals errors with statements that begin with "Error:" for all testcases | False otherwise>
}} 
```

{testbench}
"""


class Checker():


    def __init__(self, config):
        self.config = config 

        self.prompt =  ChatPromptTemplate.from_messages([
            ("system", str(_CHECKER_SYS_PROMPT)),
            ("user", _CHECKER_USER_PROMPT),
        ])

        self.base_llm = config.lm_config.checker_llm_model 
        self.llm = self.base_llm.with_structured_output(CheckerOutput)
        self.checker = self.prompt | self.llm

    
    def forward(self, state: RefinerState):
        rtl_code = state.get("rtl_code")
        testbench = state.get("testbench")

        response = self.checker.invoke({
            "testbench": testbench
        })

        # output = self._parse_json(response.content)
        # output = json.loads(output)
        return {"is_self_checking": response.is_self_checking, "is_print_error": response.error_statements, "testbench": testbench } 

    def _parse_json(self, content):
        if not content:
            return ""
            
        pattern = re.compile(
            r"```(?:\s*(?:json))?\s*(.*?)```",
            re.DOTALL | re.IGNORECASE
        )
        
        matches = pattern.findall(content)
        if matches:
            return matches[-1].strip()
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
            enhancer_lm = model,
            checker_lm = model,
            refiner_lm = model
        )
    ) 

    checker = Checker(
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

    output = checker.forward({
        "testbench": testbench
    })

    print(output)


    testbench = """
`timescale 1ns/1ps

module testbench;

    // DUT interface
    reg  [2:0] sel;
    wire [7:0] res;

    // Misc
    integer i;

    // Instantiate DUT
    decoder_3_8 uut (
        .sel(sel),
        .res(res)
    );

    // Waveform dump
    initial begin
        $dumpfile("wave_decoder_3_8.vcd");
        $dumpvars(0, testbench.uut);
    end

    // Simple watchdog to avoid hangs
    initial begin
        #200;
        $fatal(1, "Error: Timeout - simulation exceeded 200 ns");
    end

    // Self-check helpers
    task check_expected(input [2:0] exp_sel, input [7:0] exp_res);
        begin
            // Basic match
            if (res !== exp_res) begin
                $fatal(0, "Error: Decoder output mismatch for sel=%b. Expected res=%b, got res=%b",
                       exp_sel, exp_res, res);
            end

            // For valid sel values, ensure one-hot and no X/Z on res
            if (exp_sel === 3'b000 || exp_sel === 3'b001 || exp_sel === 3'b010 ||
                exp_sel === 3'b011 || exp_sel === 3'b100 || exp_sel === 3'b101 ||
                exp_sel === 3'b110 || exp_sel === 3'b111) begin
                // No unknowns on res
                if (^res === 1'bx) begin
                    $fatal(0, "Error: Output contains X/Z for sel=%b, res=%b", exp_sel, res);
                end
                // One-hot check: res != 0 and only a single bit set
                if (res == 8'b0 || (res & (res - 8'b1)) != 8'b0) begin
                    $fatal(0, "Error: Output not one-hot for sel=%b, res=%b", exp_sel, res);
                end
            end
        end
    endtask

    // Reference computation for valid sel
    function [7:0] ref_for_sel(input [2:0] v);
        begin
            case (v)
                3'b000: ref_for_sel = 8'b00000001;
                3'b001: ref_for_sel = 8'b00000010;
                3'b010: ref_for_sel = 8'b00000100;
                3'b011: ref_for_sel = 8'b00001000;
                3'b100: ref_for_sel = 8'b00010000;
                3'b101: ref_for_sel = 8'b00100000;
                3'b110: ref_for_sel = 8'b01000000;
                3'b111: ref_for_sel = 8'b10000000;
                default: ref_for_sel = 8'b10000000;
            endcase
        end
    endfunction

    // Main stimulus
    initial begin
        // Nominal sweep: cover all 0..7
        sel = 3'b000; #5; check_expected(3'b000, ref_for_sel(3'b000));
        sel = 3'b001; #5; check_expected(3'b001, ref_for_sel(3'b001));
        sel = 3'b010; #5; check_expected(3'b010, ref_for_sel(3'b010));
        sel = 3'b011; #5; check_expected(3'b011, ref_for_sel(3'b011));
        sel = 3'b100; #5; check_expected(3'b100, ref_for_sel(3'b100));
        sel = 3'b101; #5; check_expected(3'b101, ref_for_sel(3'b101));
        sel = 3'b110; #5; check_expected(3'b110, ref_for_sel(3'b110));
        sel = 3'b111; #5; check_expected(3'b111, ref_for_sel(3'b111));

        // Illegal scenario: drive X, expect DUT default branch to hold MSB
        sel = 3'bxxx; #5; check_expected(3'bxxx, 8'b10000000);

        // Boundary/transition checks
        sel = 3'b011; #5; check_expected(3'b011, ref_for_sel(3'b011));
        sel = 3'b100; #5; check_expected(3'b100, ref_for_sel(3'b100));
        sel = 3'b000; #5; check_expected(3'b000, ref_for_sel(3'b000));

        // Back-to-back toggling
        sel = 3'b001; #2; check_expected(3'b001, ref_for_sel(3'b001));
        sel = 3'b010; #2; check_expected(3'b010, ref_for_sel(3'b010));
        sel = 3'b001; #2; check_expected(3'b001, ref_for_sel(3'b001));
        sel = 3'b010; #2; check_expected(3'b010, ref_for_sel(3'b010));
        sel = 3'b011; #2; check_expected(3'b011, ref_for_sel(3'b011));
        sel = 3'b110; #2; check_expected(3'b110, ref_for_sel(3'b110));
        sel = 3'b101; #2; check_expected(3'b101, ref_for_sel(3'b101));
        sel = 3'b100; #2; check_expected(3'b100, ref_for_sel(3'b100));

        // Done
        #10;
        $finish;
    end

endmodule

"""

    output = checker.forward({
        "testbench": testbench
    })

    print(output)


    testbench = """
`timescale 1ns/1ps

module testbench;

    // DUT interface
    reg  [2:0] sel;
    wire [7:0] res;

    // Misc
    integer i;

    // Instantiate DUT
    decoder_3_8 uut (
        .sel(sel),
        .res(res)
    );

    // Waveform dump
    initial begin
        $dumpfile("wave_decoder_3_8.vcd");
        $dumpvars(0, testbench.uut);
    end

    // Simple watchdog to avoid hangs
    initial begin
        #200;
        $fatal(1, "Error: Timeout - simulation exceeded 200 ns");
    end

    // Self-check helpers
    task check_expected(input [2:0] exp_sel, input [7:0] exp_res);
        begin
            // Basic match
            if (res !== exp_res) begin
                $fatal(0, "Error: Decoder output mismatch for sel=%b. Expected res=%b, got res=%b",
                       exp_sel, exp_res, res);
            end

            // For valid sel values, ensure one-hot and no X/Z on res
            if (exp_sel === 3'b000 || exp_sel === 3'b001 || exp_sel === 3'b010 ||
                exp_sel === 3'b011 || exp_sel === 3'b100 || exp_sel === 3'b101 ||
                exp_sel === 3'b110 || exp_sel === 3'b111) begin
                // No unknowns on res
                if (^res === 1'bx) begin
                    $fatal(0, "Output contains X/Z for sel=%b, res=%b", exp_sel, res);
                end
                // One-hot check: res != 0 and only a single bit set
                if (res == 8'b0 || (res & (res - 8'b1)) != 8'b0) begin
                    $fatal(0, "Output not one-hot for sel=%b, res=%b", exp_sel, res);
                end
            end
        end
    endtask

    // Reference computation for valid sel
    function [7:0] ref_for_sel(input [2:0] v);
        begin
            case (v)
                3'b000: ref_for_sel = 8'b00000001;
                3'b001: ref_for_sel = 8'b00000010;
                3'b010: ref_for_sel = 8'b00000100;
                3'b011: ref_for_sel = 8'b00001000;
                3'b100: ref_for_sel = 8'b00010000;
                3'b101: ref_for_sel = 8'b00100000;
                3'b110: ref_for_sel = 8'b01000000;
                3'b111: ref_for_sel = 8'b10000000;
                default: ref_for_sel = 8'b10000000;
            endcase
        end
    endfunction

    // Main stimulus
    initial begin
        // Nominal sweep: cover all 0..7
        sel = 3'b000; #5; check_expected(3'b000, ref_for_sel(3'b000));
        sel = 3'b001; #5; check_expected(3'b001, ref_for_sel(3'b001));
        sel = 3'b010; #5; check_expected(3'b010, ref_for_sel(3'b010));
        sel = 3'b011; #5; check_expected(3'b011, ref_for_sel(3'b011));
        sel = 3'b100; #5; check_expected(3'b100, ref_for_sel(3'b100));
        sel = 3'b101; #5; check_expected(3'b101, ref_for_sel(3'b101));
        sel = 3'b110; #5; check_expected(3'b110, ref_for_sel(3'b110));
        sel = 3'b111; #5; check_expected(3'b111, ref_for_sel(3'b111));

        // Illegal scenario: drive X, expect DUT default branch to hold MSB
        sel = 3'bxxx; #5; check_expected(3'bxxx, 8'b10000000);

        // Boundary/transition checks
        sel = 3'b011; #5; check_expected(3'b011, ref_for_sel(3'b011));
        sel = 3'b100; #5; check_expected(3'b100, ref_for_sel(3'b100));
        sel = 3'b000; #5; check_expected(3'b000, ref_for_sel(3'b000));

        // Back-to-back toggling
        sel = 3'b001; #2; check_expected(3'b001, ref_for_sel(3'b001));
        sel = 3'b010; #2; check_expected(3'b010, ref_for_sel(3'b010));
        sel = 3'b001; #2; check_expected(3'b001, ref_for_sel(3'b001));
        sel = 3'b010; #2; check_expected(3'b010, ref_for_sel(3'b010));
        sel = 3'b011; #2; check_expected(3'b011, ref_for_sel(3'b011));
        sel = 3'b110; #2; check_expected(3'b110, ref_for_sel(3'b110));
        sel = 3'b101; #2; check_expected(3'b101, ref_for_sel(3'b101));
        sel = 3'b100; #2; check_expected(3'b100, ref_for_sel(3'b100));

        // Done
        #10;
        $finish;
    end

endmodule

"""

    output = checker.forward({
        "testbench": testbench
    })

    print(output)

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

    output = checker.forward({
        "testbench": testbench
    })

    print(output)


if __name__ == '__main__':
    main()
