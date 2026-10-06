
"""Refines testbenches to be self checking with Error print statements
"""


import re 
import argparse 

from langchain_core.prompts import ChatPromptTemplate

from core.data_generation.agent import Agent 
from core.data_generation.state import DataGenState
from core.data_generation.config.config import FlowConfigs, LMConfigs

from dotenv import load_dotenv
load_dotenv()

_PRINT_REFINER_SYS_PROMPT = """
You are an expert verification engineer, responsible for analyzing testbenches.
"""

_PRINT_REFINER_USER_PROMPT = """
Your task is to refine the print statements in this testbench

- For print statements that signal testcase failure, modify it so that the statement starts with "Error:". For example, 

if (zero !== 1'b0) begin
    $display("Test failed @10ns: zero expected 0, got %b", zero);
    $finish;
end

should be modified to: 

if (zero !== 1'b0) begin
    $display("Error: Test failed @10ns: zero expected 0, got %b", zero);
    $finish;
end

Instructions: 
- Only modify the print statements. 
- Don't modify anything else in the testbench. 

Reason first about the given testbench then output the refined testbench following instructions: 

{testbench}

"""


class PrintRefiner():
    
    def __init__(self, config):
        self.config = config 

        self.prompt =  ChatPromptTemplate.from_messages([
            ("system", str(_PRINT_REFINER_SYS_PROMPT)),
            ("user", _PRINT_REFINER_USER_PROMPT),
        ])

        self.llm = config.lm_config.refiner_llm_model 
        self.refiner = self.prompt | self.llm

    
    def forward(self, state: DataGenState):
        testbench = state.get("testbench")

        response = self.refiner.invoke({
            "testbench": testbench
        })

        output = self._parse_testbench(response.content)

        return {"testbench": output} 
 
    def _parse_testbench(self, content):
        if not content:
            return ""
            
        pattern = re.compile(
            r"```(?:\s*(?:testbench|systemverilog|verilog))?\s*(.*?)```",
            re.DOTALL | re.IGNORECASE
        )
        
        matches = pattern.findall(content)
        if matches:
            return matches[-1].strip()
        else:
            return content.strip()




_REF_REFINER_SYS_PROMPT = """
You are an expert verification engineer, responsible for refining testbenches to be self-checking.
"""

_REF_REFINER_USER_PROMPT = """
Your task is to refine this testbench to be self-checking

- The testbench must be self-checking and display error statements must start with "Error:" for example: $fatal(0, "Error: Test failed for A=%b B=%b C=%b D=%b", A, B, C, D);
- $fatal first argument must be integer indicating severity, for example: $fatal(0, "Error: Test failed for A=%b B=%b C=%b D=%b", A, B, C, D);
- Don't display the errors count at the end. 
- Don't change the testcases given in the testbench; only add golden references for each testcases. 

Reason first about the given testbench then output the refined testbench following instructions: 

[Natural Spec]
{natural_spec}

[RTL Code]
{rtl_code}

[Testbench]
{testbench}
"""


class ReferenceRefiner():
    
    def __init__(self, config):
        self.config = config 

        self.prompt =  ChatPromptTemplate.from_messages([
            ("system", str(_REF_REFINER_SYS_PROMPT)),
            ("user", _REF_REFINER_USER_PROMPT),
        ])

        self.llm = config.lm_config.refiner_llm_model 
        self.refiner = self.prompt | self.llm

    
    def forward(self, state: DataGenState):
        testbench = state.get("testbench")
        rtl_code = state.get("rtl_code")
        natural_spec = state.get("natural_spec")

        response = self.refiner.invoke({
            "testbench": testbench,
            "rtl_code": rtl_code,
            "natural_spec": natural_spec
        })

        output = self._parse_testbench(response.content)

        return {"testbench": output} 
 
    def _parse_testbench(self, content):
        if not content:
            return ""
            
        pattern = re.compile(
            r"```(?:\s*(?:testbench|systemverilog|verilog))?\s*(.*?)```",
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

    print_refiner = PrintRefiner(
        config=config
    )
 

    ref_refiner = ReferenceRefiner(
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

    output = print_refiner.forward({
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

    output = ref_refiner.forward({
        "testbench": testbench,
        "rtl_code": rtl_code,
        "natural_spec": natural_spec
    })

    print(output)
   

if __name__ == '__main__':
    main()
