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

"""Merge two JSONL files into a single JSONL file."""

import argparse
import json
import os

from core.data_generation.utils import read_jsonl


def merge_two_jsonl(input_a: str, input_b: str, output_path: str) -> None:
    for input_path in [input_a, input_b]:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        if not input_path.endswith(".jsonl"):
            raise ValueError(f"Expected a .jsonl file: {input_path}")

    merged_data = []
    merged_data.extend(read_jsonl(input_a))
    merged_data.extend(read_jsonl(input_b))

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for entry in merged_data:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")

    print(f"Merged {len(merged_data)} entries into {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Merge two JSONL files together.")
    parser.add_argument("--input_a", type=str, required=True, help="First input JSONL file")
    parser.add_argument("--input_b", type=str, required=True, help="Second input JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file")
    args = parser.parse_args()

    merge_two_jsonl(
        input_a=args.input_a,
        input_b=args.input_b,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
