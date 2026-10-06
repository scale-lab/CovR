"""Testbench Coverage Reporter 
"""

import os
import re 
import sys 
import argparse 
import tempfile
from enum import Enum

from core.metrics.coverage import run_covered , report_coverage
from core.metrics.vcs import run_urg, report_coverage_vcs
from core.data_generation.state import DataGenState
from core.data_generation.config.config import FlowConfigs, LMConfigs
from core.data_generation.compiler import Compiler

class Reporter():

    def __init__(self, config):
        self.config = config  
        self.output_dir = tempfile.mkdtemp()         # TODO: move this to config class
        self.max_coverage_report_tokens = 6000

    def _estimate_tokens(self, text: str) -> int:
        if not text:
            return 0

        words = len(text.split())
        special_chars = len(re.findall(r"[^\w\s]", text))
        return max(1, int(words * 0.75) + special_chars // 4)

    def _extract_first_float(self, text: str, default: float = 100.0) -> float:
        match = re.search(r"(\d+\.\d+)", text)
        return float(match.group(1)) if match else default

    def _smart_truncate_coverage_report(self, report: str, max_tokens: int | None = None) -> str:
        if not report:
            return ""

        token_budget = max_tokens or self.max_coverage_report_tokens
        if self._estimate_tokens(report) <= token_budget:
            return report

        raw_sections = [section.strip() for section in re.split(r"=+", report) if section.strip()]
        if not raw_sections:
            return report

        sections = raw_sections[:]
        while len(sections) > 1:
            candidate = "\n\n===============================================================================\n".join(sections)
            if self._estimate_tokens(candidate) <= token_budget:
                return candidate
            sections.pop()

        return sections[0]

    def forward(self, state: DataGenState):
        testbench = state.get("testbench")
        rtl_code = state.get("rtl_code")
        language = state.get("language")
        top_module = state.get("top_module")
        vcd_file = state.get("vcd_file")

        tb_file_path = os.path.join(self.config.output_dir, f"testbench_{top_module}.v")
        with open(tb_file_path, "w") as f:
            f.write(testbench)
        
        rtl_file_path = os.path.join(self.config.output_dir, f"rtl_{top_module}.v")
        with open(rtl_file_path, "w") as f:
            f.write(rtl_code)

        if language == "SystemVerilog" or language == "Verilog":
            # correct, report_file = run_covered(
            #     verilog_files = [rtl_file_path],
            #     testbench=tb_file_path,
            #     vcd_file=vcd_file,
            #     singularity_image="/oscar/data/sreda/mabdelat/LLM4TB/sing.sif",
            #     output_dir=self.config.output_dir,
            #     top_module_name=top_module,
            #     top_module_tree_path="testbench.uut"
            # )

            correct, report_file = run_urg(
                output_dir=self.config.output_dir,
                top_module_name=top_module,
                testbench=tb_file_path
            )

            print("Coverage Report is: ", correct, report_file)
            if correct: 
                stats = report_coverage_vcs(report_file)

                metrics = ['line', 'toggle', 'cond', 'fsm', 'branch']
                coverage_met = all(
                    (stats[m] is None) or (stats[m] == 100)
                    for m in metrics if m in stats
                )
                
                coverage_report = None 
                with open(report_file, 'r') as f:
                    coverage_report = self._smart_truncate_coverage_report(
                        f.read(),
                        max_tokens=self.max_coverage_report_tokens,
                    )

                if not coverage_met:
                    reflection_mode = "enhance"
                else:
                    reflection_mode = ""

                return {"covered_executes": True, "coverage_stats": stats, "coverage_report": coverage_report, "coverage_met": coverage_met, "testbenches": [testbench], "coverage_reports": [coverage_report], "coverage_stats_list": [stats], "coverage_met_list": [coverage_met], "reflection_mode": reflection_mode}
            else:
                return {"covered_executes": False}

        elif language == "cocotb":
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
 
    config = FlowConfigs(
        lm_config=LMConfigs(
            prompter_lm = model,
            fixer_lm = model,
            tb_generator_lm = model,
            enhancer_lm = model,
            curator_lm = model,
            reflector_lm = model,
            max_seq_length = 4028,
            max_new_tokens = 1028,
            checker_lm = model
        )
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

    testbench_enhanced = """
`timescale 1ns / 1ps
module testbench;

    localparam CLK_PERIOD = 10;
    localparam TIMEOUT    = 5000;

    // Inputs
    reg [7:0] data;
    reg clk;
    reg rst_n;

    // Output
    wire [31:0] checksum;

    // DUT
    AdlerChecksum uut (
        .data(data),
        .clk(clk),
        .checksum(checksum)
    );

    // Clock generation
    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // Stimulus
    initial begin
        $dumpfile("wave_AdlerChecksum.vcd");
        $dumpvars(0, testbench.uut);

        // Reset-like sequence
        data = 0;
        repeat (5) @(posedge clk);

        // Incremental sweep (0 → 255)
        for (int i = 0; i < 256; i++) begin
            data = i;
            @(posedge clk);
        end

        // Edge patterns
        data = 8'hFF; @(posedge clk);
        data = 8'h00; @(posedge clk);
        data = 8'h7F; @(posedge clk);
        data = 8'h80; @(posedge clk);
        data = 8'hAA; @(posedge clk);
        data = 8'h55; @(posedge clk);

        data = 8'hAA; @(posedge clk);
        data = 8'h55; @(posedge clk);
        data = 8'hF0; @(posedge clk);
        data = 8'h0F; @(posedge clk);

        // Random bursts
        repeat (300) begin
            data = $urandom_range(0, 255);
            @(posedge clk);
        end

       // Force s1 overflow
        repeat (150000) begin
            data = 8'hFF;      // maximize addition each cycle
            @(posedge clk);
        end
        repeat (75000) begin
            data = 8'hFF;     // max value to drive s1 up fast
            @(posedge clk);
        end


        $display("Final checksum = %h", checksum);
        #(CLK_PERIOD*10);
        $finish;
    end

    // Monitor
    initial
        $monitor("T=%0t ns | data=%h | checksum=%h", $time, data, checksum);

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
assign checksum = (s2 << 16) + s1;


endmodule
"""
    language = "SystemVerilog"
    top_module = "AdlerChecksum"
    
    reporter = Reporter(
        config=config
    )

    compiler = Compiler(
        config=config
    )

    output = compiler.forward({
        "testbench": testbench_enhanced,
        "rtl_code": rtl_code,
        "language": language,
        "top_module": top_module
    })

    output = reporter.forward({
        "testbench": testbench_enhanced,
        "rtl_code": rtl_code,
        # "vcd_file": output["vcd_file"],
        "language": language,
        "top_module": top_module
    })

    print(output["coverage_report"])
    print(output["coverage_stats"])


if __name__ == '__main__':
    main()
