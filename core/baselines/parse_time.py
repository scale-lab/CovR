#!/usr/bin/env python3
import re
import sys
from datetime import datetime

TASK_HEADER_RE = re.compile(
    r'^(?P<ts>\d{4}-\d{2}-\d{2} [0-9:.]+) \| .*task (?P<idx>\d+)\/(?P<total>\d+) \[(?P<name>[^\]]+)\]'
)
TIMESTAMP_RE = re.compile(r'^(?P<ts>\d{4}-\d{2}-\d{2} [0-9:.]+)')

TIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"


def parse_timestamp(ts_str: str) -> datetime:
    return datetime.strptime(ts_str, TIME_FORMAT)


def compute_task_times(log_path: str):
    tasks = []
    current_task = None
    last_timestamp = None

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            # Track last timestamp in file
            m_ts = TIMESTAMP_RE.match(line)
            if m_ts:
                last_timestamp = parse_timestamp(m_ts.group("ts"))

            # Detect start of new task
            m = TASK_HEADER_RE.match(line)
            if m:
                ts = parse_timestamp(m.group("ts"))
                idx = int(m.group("idx"))
                name = m.group("name")

                # Close previous task if exists
                if current_task is not None:
                    current_task["end"] = ts
                    tasks.append(current_task)

                current_task = {
                    "idx": idx,
                    "name": name,
                    "start": ts,
                    "end": None,
                }

    # Close last task using final timestamp
    if current_task and last_timestamp:
        current_task["end"] = last_timestamp
        tasks.append(current_task)

    return tasks


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} LOG_FILE")
        sys.exit(1)

    log_file = sys.argv[1]
    tasks = compute_task_times(log_file)

    print("task_idx,task_name,start_time,end_time,duration_seconds")

    total = 0
    count = 0

    for t in tasks:
        duration = (t["end"] - t["start"]).total_seconds()
        total += duration
        count += 1

        print(
            f"{t['idx']},{t['name']},"
            f"{t['start'].isoformat(' ')},{t['end'].isoformat(' ')},"
            f"{duration:.3f}"
        )

    avg = total / count if count > 0 else 0

    print("\n=== SUMMARY ===")
    print(f"Total tasks: {count}")
    print(f"Average time: {avg:.3f} seconds  ({avg/60:.2f} min)")


if __name__ == "__main__":
    main()
