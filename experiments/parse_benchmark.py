#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# ///
"""Parse a benchmark log file produced by benchmark.py and print stats.

Usage: ./parse_benchmark.py [logfile]   # parse a specific log
       ./parse_benchmark.py             # pick from recent logs in logs/
"""

import os
import re
import sys
import statistics
import time

def parse_log(path):
    with open(path) as f:
        text = f.read()

    # Extract header info
    model = re.search(r"Model:\s+(.+)", text)
    bench = re.search(r"Benchmark:\s+(.+)", text)
    k_val = re.search(r"K:\s+(\d+)", text)
    tokens = re.search(r"Tokens:\s+(\d+)", text)

    print(f"Model:     {model.group(1) if model else '?'}")
    print(f"Benchmark: {bench.group(1) if bench else '?'}")
    print(f"K:         {k_val.group(1) if k_val else '?'}")
    print(f"Tokens:    {tokens.group(1) if tokens else '?'}")
    print()

    # Parse server-reported timing from [server] lines
    prefills = re.findall(r"prefill=(\d+) tokens in (\d+)ms", text)
    generations = re.findall(r"generated=(\d+) tokens in (\d+)ms \(([0-9.]+) tok/s\)", text)

    # Parse runner-measured elapsed times
    elapsed_times = re.findall(r"\[runner\] elapsed=([0-9.]+)s", text)

    # Parse query banners to get prompts
    queries = re.findall(r'--- Query (\d+)/\d+: "(.+?)" ---', text)

    n = max(len(generations), len(elapsed_times), len(queries))
    if n == 0:
        print("No query results found in log.")
        return

    print(f"{'Q':>2}  {'gen':>4}  {'tok/s':>7}  {'TTFT':>6}  {'wall':>6}  prompt")
    print(f"{'':>2}  {'tok':>4}  {'server':>7}  {'ms':>6}  {'s':>6}  ")
    print("-" * 72)

    tok_s_list = []
    ttft_list = []
    gen_list = []
    wall_list = []

    for i in range(n):
        qn = i + 1
        prompt = queries[i][1] if i < len(queries) else "?"

        if i < len(generations):
            gen_tok = int(generations[i][0])
            gen_ms = int(generations[i][1])
            tok_s = float(generations[i][2])
        else:
            gen_tok, gen_ms, tok_s = 0, 0, 0.0

        if i < len(prefills):
            pfill_tok = int(prefills[i][0])
            ttft_ms = int(prefills[i][1])
        else:
            pfill_tok, ttft_ms = 0, 0

        wall = float(elapsed_times[i]) if i < len(elapsed_times) else 0.0

        print(f"{qn:>2}  {gen_tok:>4}  {tok_s:>7.2f}  {ttft_ms:>6}  {wall:>6.2f}  {prompt[:40]}")

        if tok_s > 0:
            tok_s_list.append(tok_s)
        if ttft_ms > 0:
            ttft_list.append(ttft_ms)
        gen_list.append(gen_tok)
        wall_list.append(wall)

    print("-" * 72)
    print()

    # Summary
    total_tok = sum(gen_list)
    total_wall = sum(wall_list)

    print("--- Summary ---")
    print(f"Queries:     {n}")
    print(f"Total tok:   {total_tok}")
    print(f"Total wall:  {total_wall:.1f}s")

    if tok_s_list:
        print(f"tok/s:       min={min(tok_s_list):.2f}  avg={statistics.mean(tok_s_list):.2f}"
              f"  median={statistics.median(tok_s_list):.2f}  max={max(tok_s_list):.2f}")
    else:
        print("tok/s:       (no server timing found — check log for errors)")

    if ttft_list:
        print(f"TTFT:        min={min(ttft_list)}ms  avg={int(statistics.mean(ttft_list))}ms"
              f"  median={int(statistics.median(ttft_list))}ms  max={max(ttft_list)}ms")

    if wall_list:
        print(f"Wall/query:  min={min(wall_list):.2f}s  avg={statistics.mean(wall_list):.2f}s"
              f"  median={statistics.median(wall_list):.2f}s  max={max(wall_list):.2f}s")

def pick_log(logs_dir):
    """Show the 10 most recent logs, let user pick. Default is latest."""
    logs = []
    for name in os.listdir(logs_dir):
        path = os.path.join(logs_dir, name)
        if os.path.isfile(path) and name.endswith(".log"):
            logs.append((os.path.getmtime(path), name, path))
    if not logs:
        print(f"No .log files in {logs_dir}/")
        sys.exit(1)
    logs.sort(reverse=True)
    logs = logs[:10]

    print("Recent logs:")
    for i, (mtime, name, _) in enumerate(logs):
        age = time.time() - mtime
        if age < 3600:
            ago = f"{int(age / 60)}m ago"
        elif age < 86400:
            ago = f"{int(age / 3600)}h ago"
        else:
            ago = f"{int(age / 86400)}d ago"
        default = " (latest)" if i == 0 else ""
        print(f"  {i + 1}. {name}  ({ago}){default}")
    choice = input(f"\nChoose [1]: ").strip() or "1"
    return logs[int(choice) - 1][2]


def main():
    if len(sys.argv) >= 2:
        paths = sys.argv[1:]
    else:
        logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        paths = [pick_log(logs_dir)]

    for path in paths:
        print(f"\n=== {os.path.basename(path)} ===\n")
        parse_log(path)
        print()

if __name__ == "__main__":
    main()
