import operator
from typing_extensions import TypedDict
from typing import List, Annotated, TypedDict, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing import List


class RefinerState(TypedDict):
    rtl_code: str 
    top_module: str
    rtl_error: str
    rtl_fix_iters: int
    natural_spec: str 
    language: str 
    testbench: str
    testbench_runs: bool
    is_self_checking: bool 
    is_print_error: bool
    testbenches_status: Annotated[List[bool], operator.add]
    testbench_error: str
    fix_iters: int
    refine_iters: int
    coverage_stats: dict
    coverage_report: str
    covered_executes: bool
    coverage_met: bool
