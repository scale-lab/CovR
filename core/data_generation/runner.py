"""Runs main data generation workflow.. 
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
from core.data_generation.enhancer import Enhancer
from core.data_generation.tb_generator import TBGenerator
from core.data_generation.reflector import Reflector
from core.data_generation.curator import Curator
from core.data_generation.config.config import FlowConfigs, LMConfigs

from dotenv import load_dotenv
load_dotenv()


class DataGenNodes(Enum):
    Prompter = "Prompter"
    Compiler = "Compiler"
    Fixer = "Fixer"
    Reporter = "Reporter"
    Enhancer = "Enhancer"
    TBGenerator = "TBGenerator"
    Reflector = "Reflector"
    Curator = "Curator"


class DataGenRunner():


    def __init__(self, config):
        self.config = config
        self._start_at = None
        self.prompter = Prompter(
            config=config
        ) 

        self.tb_generator = TBGenerator(
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

        self.enhancer = Enhancer(
            config=config
        )
        
        self.reflector = Reflector(
            config=config
        )

        self.curator = Curator(
            config=config
        )

        self.builder = StateGraph(DataGenState)

        self.builder.add_node(
            DataGenNodes.Prompter.value, 
            self.prompter.forward
        )

        self.builder.add_node(
            DataGenNodes.TBGenerator.value, 
            self.tb_generator.forward
        )

        self.builder.add_node(
            DataGenNodes.Compiler.value, 
            self.compiler.forward
        )

        self.builder.add_node(
            DataGenNodes.Fixer.value, 
            self.fixer.forward
        )

        self.builder.add_node(
            DataGenNodes.Reporter.value, 
            self.reporter.forward
        )

        self.builder.add_node(
            DataGenNodes.Enhancer.value, 
            self.enhancer.forward
        )

        self.builder.add_node(
            DataGenNodes.Reflector.value,
            self.reflector.forward
        )

        self.builder.add_node(
            DataGenNodes.Curator.value,
            self.curator.forward
        )

        self.builder.add_conditional_edges(
            START,
            self._route_start,
            [
                DataGenNodes.Prompter.value,
                DataGenNodes.TBGenerator.value,
                DataGenNodes.Compiler.value,
                DataGenNodes.Fixer.value,
                DataGenNodes.Reporter.value,
                DataGenNodes.Enhancer.value,
                DataGenNodes.Reflector.value,
                DataGenNodes.Curator.value,
            ]
        )
        self.builder.add_edge(DataGenNodes.Prompter.value, DataGenNodes.TBGenerator.value)
        self.builder.add_edge(DataGenNodes.TBGenerator.value, DataGenNodes.Compiler.value)
        self.builder.add_edge(DataGenNodes.Fixer.value, DataGenNodes.Compiler.value)

        self.builder.add_edge(DataGenNodes.Enhancer.value, DataGenNodes.Compiler.value)

        self.builder.add_conditional_edges(
            DataGenNodes.Compiler.value,
            self._after_compile,
            [DataGenNodes.Reflector.value, DataGenNodes.Curator.value, DataGenNodes.Reporter.value, END]
        )

        self.builder.add_conditional_edges(
            DataGenNodes.Reporter.value,
            self._after_report,
            [DataGenNodes.Reflector.value, DataGenNodes.Curator.value, END]
        )

        self.builder.add_edge(DataGenNodes.Reflector.value, DataGenNodes.Curator.value)

        self.builder.add_conditional_edges(
            DataGenNodes.Curator.value,
            self._route_after_reflection,
            [DataGenNodes.Fixer.value, DataGenNodes.Enhancer.value, END]
        )

        self.graph = self.builder.compile()
        png_graph = self.graph.get_graph().draw_mermaid_png()
        os.makedirs("./output", exist_ok=True)
        with open(os.path.join("output/", "data_gen_flow.png"), "wb") as f:
            f.write(png_graph)

    def forward(self, state: DataGenState, start_at = None):
        self._start_at = start_at
        output = self.graph.invoke(state, {"recursion_limit":  self.config.graph_recursion_limit,  "configurable": {"start_at": start_at}})  
        return output

    def _route_start(self, state: DataGenState):
        if self._start_at is None:
            return DataGenNodes.Prompter.value

        if isinstance(self._start_at, DataGenNodes):
            return self._start_at.value

        return str(self._start_at)

    def _should_fix(self, state: DataGenState):
        testbench_runs = state.get("testbench_runs")
        fix_iters = state.get("fix_iters", 0)
        if testbench_runs:
            return DataGenNodes.Reporter.value
        elif  fix_iters >= self.config.max_refine_iters:
            return END
        else:
            return DataGenNodes.Fixer.value

    def _should_enhance(self, state: DataGenState):
        coverage_stats = state.get("coverage_stats")
        enhance_iters = state.get("enhance_iters", 0)
        coverage_met = state.get("coverage_met")
        if coverage_met:
            return END
        elif enhance_iters >= self.config.max_enhance_iters:
            return END
        else:
            return DataGenNodes.Enhancer.value

    def _after_report(self, state: DataGenState):
        coverage_met = state.get("coverage_met", False)
        enhance_iters = state.get("enhance_iters", 0)

        if coverage_met:
            return END
        elif enhance_iters >= self.config.max_enhance_iters:
            return END
        else:
            state["reflection_mode"] = "enhance"
        
        if self.config.wo_reflector: 
            return DataGenNodes.Curator.value

        return DataGenNodes.Reflector.value


    def _route_after_reflection(self, state: DataGenState):
        mode = state["reflection_mode"]
        if mode == "fix":
            return DataGenNodes.Fixer.value
        elif mode == "enhance":
            return DataGenNodes.Enhancer.value
        return END


    def _after_compile(self, state: DataGenState):
        testbench_runs = state["testbench_runs"]
        fix_iters = state.get("fix_iters", 0)

        if testbench_runs:
            return DataGenNodes.Reporter.value
        elif fix_iters >= self.config.max_refine_iters:
            return END
        else:
            if self.config.wo_reflector:
                return DataGenNodes.Curator.value
            return DataGenNodes.Reflector.value



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Model name', default='gpt-4o-mini-2024-07-18')
    parser.add_argument('--output', type=str, help='Path to output file for storing the output of each query', default='./output/refiner_eval.log')
    parser.add_argument(
        '--testbench_style',
        type=str,
        choices=['self_checking', 'stimulus_only'],
        default='stimulus_only',
        help='Select whether the workflow generates self-checking or stimulus-only testbenches.'
    )

    args = parser.parse_args()

    model = args.model 
    output = args.output  
    testbench_style = args.testbench_style

    config = FlowConfigs(
        lm_config=LMConfigs(
            prompter_lm = model,
            fixer_lm = model,
            tb_generator_lm = model,
            enhancer_lm = model,
            checker_lm = model,
            refiner_lm = model,
        ),
        testbench_style=testbench_style
    )

    runner = DataGenRunner(
        config = config
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

#     rtl_code = """
# module AdlerChecksum (
#     input        clk,         // clock
#     input        rst_n,       // active-low synchronous reset
#     input  [7:0] data,        // input data byte
#     output reg [31:0] checksum
# );

#     // two 16-bit checksum registers
#     reg [15:0] s1;
#     reg [15:0] s2;

#     // synchronous reset: clears s1 and s2 to 1
#     always @(posedge clk) begin
#         if (!rst_n) begin
#             s1 <= 16'h0001;
#             s2 <= 16'h0001;
#         end
#         else begin
#             s1 <= (s1 + data) % 16'd65521;
#             s2 <= (s2 + s1) % 16'd65521;
#         end
#     end

#     // continuously compute the final Adler checksum
#     always @* begin
#         checksum = (s2 << 16) + s1;
#     end

# endmodule
# """

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
    language = "SystemVerilog"   # SystemVerilog or cocotb


    output = runner.forward({
        "rtl_code": rtl_code,
        "natural_spec": natural_spec,
        "language": language,
        "top_module": "AdlerChecksum",
        "testbench_style": testbench_style
    })
    
    print(output['testbench'])



if __name__ == '__main__':
    main()
