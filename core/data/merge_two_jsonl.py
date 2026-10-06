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
