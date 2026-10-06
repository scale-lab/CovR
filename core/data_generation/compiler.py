"""Compiles the testbench and reports any errors
"""

import os
import re 
import sys 
import argparse 
import tempfile

from core.metrics.coverage import run_testbench 
from core.metrics.vcs import run_testbench_vcs
from core.data_generation.state import DataGenState
from core.data_generation.config.config import FlowConfigs, LMConfigs

class Compiler():

    def __init__(self, config):
        self.config = config  

    def forward(self, state: DataGenState):
        testbench = state.get("testbench")
        rtl_code = state.get("rtl_code")
        language = state.get("language")
        top_module = state.get("top_module")

        tb_file_path = os.path.join(self.config.output_dir, f"testbench_{top_module}.v")
        with open(tb_file_path, "w") as f:
            f.write(testbench)
        
        rtl_file_path = os.path.join(self.config.output_dir, f"rtl_{top_module}.v")
        with open(rtl_file_path, "w") as f:
            f.write(rtl_code)

        if language == "SystemVerilog" or language == "Verilog":
            correct, output = run_testbench_vcs(
                testbench=tb_file_path,
                verilog_files=[rtl_file_path],
                top_module_name=top_module,
                output_dir=self.config.output_dir
            )

            if correct:               
                return {"testbench_runs": True, "testbench_error": ""}
            else:
                return {"testbench_runs": False, "testbench_error": output, "vcd_file": "", "reflection_mode": "fix"}

        elif language == "cocotb":
            # compile cocotb 
            pass 
        else:
            print("[ERROR] Invalid Language : ", language)
            sys.exit(0)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Model name', default='gpt-4o-mini-2024-07-18')
    parser.add_argument('--output', type=str, help='Path to output file for storing the output of each query', default='./output/refiner_eval.log')

    args = parser.parse_args()

    model = args.model 
    output = args.output 

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
        $display("Testbench finished");
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
    language = "SystemVerilog"

    config = FlowConfigs(
        lm_config=LMConfigs(
            prompter_lm = model,
            fixer_lm = model,
            tb_generator_lm= model,
            enhancer_lm  = model
        )
    )


    compiler = Compiler(
        config=config
    )

    output = compiler.forward({
        "testbench": testbench,
        "rtl_code": rtl_code,
        "language": language,
        "top_module": "AdlerChecksum"
    })

    print(output["testbench_runs"])
    print(output["testbench_error"])


if __name__ == '__main__':
    main()
