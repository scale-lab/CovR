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

"""Compiles the rtl code and reports any errors
"""

import os
import re 
import sys 
import shutil
import argparse 
import tempfile
import subprocess

from core.data_generation.state import DataGenState
from core.data_generation.config.config import FlowConfigs, LMConfigs

def run_ivl(
    verilog_files: list[str],
    output_dir: str,
    singularity_image: str = "sing.sif",
): 
    iverilog_executable = shutil.which("iverilog")
    
    os.makedirs(output_dir, exist_ok=True)

    file_name = os.path.basename(verilog_files[0])
    out_file = os.path.join(f"{file_name}.out")

    if iverilog_executable:
        compile_cmd = ["iverilog", "-g2012", "-o", out_file] + verilog_files
        run_cmd = ["vvp", out_file]
    else:
        compile_cmd = [
            "singularity", 
            "exec",
            "-B", "/oscar/data:/oscar/data",
            "-B", f"{os.getcwd()}:{os.getcwd()}",
            "-W", output_dir,                           
            singularity_image, 
            "iverilog", 
            "-g2012", 
            "-o", out_file
        ] + verilog_files
        run_cmd = ["singularity", "exec", singularity_image, "vvp", out_file]

    compile_result = subprocess.run(
        compile_cmd, 
        cwd=output_dir,
        capture_output=True, 
        text=True
    )
    if compile_result.stderr:
        print(f"[Error] Running iverilog: {compile_result.stderr}")
        return False, compile_result.stderr 
    
    try:
        run_result = subprocess.run(
            run_cmd, 
            cwd=output_dir,
            capture_output=True, 
            text=True, 
            timeout=60*2
        )
    except subprocess.TimeoutExpired:
        print(f"Simulation timed out after {60*2} seconds")
        return False, ""

    if run_result.stderr:
        print(f"[Error] Running vvp: {run_result.stderr}")
        return False, run_result.stderr 
    
    return True, run_result.stdout



class RTLCompiler():

    def __init__(self, config):
        self.config = config  

    def forward(self, state: DataGenState):
        rtl_code = state.get("rtl_code")
        top_module = state.get("top_module")
        
        rtl_file_path = os.path.join(self.config.output_dir, f"rtl_{top_module}.v")
        with open(rtl_file_path, "w") as f:
            f.write(rtl_code)

        correct, output = run_ivl(
            verilog_files=[rtl_file_path],
            singularity_image="/oscar/data/sreda/mabdelat/LLM4TB/sing.sif",
            output_dir=self.config.output_dir
        )

        print("RTL Execution Result is: ", correct, output)
        if correct: 
            return {"rtl_runs": True, "rtl_error": ""}
        else:
            return {"rtl_runs": False, "rtl_error": output}

    



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Model name', default='gpt-4o-mini-2024-07-18')
    parser.add_argument('--output', type=str, help='Path to output file for storing the output of each query', default='./output/refiner_eval.log')

    args = parser.parse_args()

    model = args.model 
    output = args.output 

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
            enhancer_lm = model
        )
    )


    compiler = RTLCompiler(
        config=config
    )

    output = compiler.forward({
        "rtl_code": rtl_code,
        "top_module": "AdlerChecksum"
    })

    print(output["rtl_runs"])
    print(output["rtl_error"])


if __name__ == '__main__':
    main()
