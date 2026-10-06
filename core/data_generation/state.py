import operator
from typing_extensions import TypedDict
from typing import List, Annotated, TypedDict, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing import List, Annotated, TypedDict, Sequence, Literal, Dict, Any


class IterationRecord(TypedDict):
    iteration: int
    mode: Literal["fix", "enhance", "generate"]
    generator_output: str
    testbench: str
    errors_encountered: str
    llm_reflection: str
    tool_feedback_outcome: str
    accepted: bool


class EvolvingContext(TypedDict):
    iterations: List[IterationRecord]
    current_status: Dict[str, Any]


class DataGenState(TypedDict):
    rtl_code: str 
    top_module: str
    testbench_style: str
    rtl_error: str
    rtl_fix_iters: int
    natural_spec: str 
    language: str 
    tbgen_prompt: str
    test_plan: str
    testbench: str
    fixer_reasoning_trace: str
    enhancer_reasoning_trace: str
    reflection_mode: str 
    reflection_summary: str 
    testbenches : Annotated[List[str], operator.add]
    fixer_testbenches : Annotated[List[str], operator.add]
    enhancer_testbenches : Annotated[List[str], operator.add]
    testbench_runs: bool
    testbenches_status: Annotated[List[bool], operator.add]
    testbench_error: str
    fix_iters: int
    enhance_iters: int
    coverage_stats: dict
    coverage_stats_list: Annotated[List[dict], operator.add]
    coverage_report: str
    coverage_reports: Annotated[List[str], operator.add]
    covered_executes: bool
    coverage_met: bool
    coverage_met_list: Annotated[List[str], operator.add]
    tb_generator_reasoning_trace: str
    reflection_history: Annotated[List[str], operator.add]
    fixing_context: EvolvingContext
    enhancement_context: EvolvingContext
