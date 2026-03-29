#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# ///
"""Flash-MoE benchmark runner.

Starts the inference server, lets you pick a benchmark, sends queries one by one.
All server output (stdout+stderr) and SSE responses are printed live and written to a log file.
No parsing — that's parse_benchmark.py's job.
"""

import argparse
import contextlib
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request


@contextlib.contextmanager
def managed_server(cmd, log):
    """Launch a subprocess and guarantee it gets killed on exit (including Ctrl-C)."""
    server = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
    try:
        yield server
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()


def find_models(models_dir):
    """Find model directories that have a config.json and model_weights.bin."""
    models = {}
    if not os.path.isdir(models_dir):
        return models
    for name in sorted(os.listdir(models_dir)):
        config_path = os.path.join(models_dir, name, "config.json")
        weights_path = os.path.join(models_dir, name, "model_weights.bin")
        if os.path.isfile(config_path) and os.path.isfile(weights_path):
            with open(config_path) as f:
                cfg = json.load(f)
            # num_experts_per_tok may be top-level or under text_config
            tc = cfg.get("text_config", cfg)
            k = tc.get("num_experts_per_tok", 8)
            num_experts = tc.get("num_experts", 0)
            num_layers = tc.get("num_hidden_layers", 0)
            models[name] = {
                "path": os.path.join(models_dir, name),
                "k": k,
                "num_experts": num_experts,
                "num_layers": num_layers,
            }
    return models


def load_benchmarks(bench_dir):
    """Load all benchmark files from the benchmarks/ directory. Each file is one benchmark, each line is a query."""
    benchmarks = {}
    if not os.path.isdir(bench_dir):
        return benchmarks
    for name in sorted(os.listdir(bench_dir)):
        path = os.path.join(bench_dir, name)
        if os.path.isfile(path) and not name.startswith("."):
            with open(path) as f:
                queries = [line.strip() for line in f if line.strip()]
            if queries:
                benchmarks[name] = queries
    return benchmarks


def wait_for_server(port, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2)
            return True
        except Exception:
            time.sleep(0.5)
    return False


def send_query(port, prompt, max_tokens, session_id, log):
    """Send a chat completion request. Print and log the raw SSE stream."""
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "session_id": session_id,
    }).encode()

    req = urllib.request.Request(
        f"http://localhost:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )

    try:
        resp = urllib.request.urlopen(req, timeout=300)
    except Exception as e:
        msg = f"  [ERROR] Request failed: {e}\n"
        sys.stdout.write(msg)
        log.write(msg)
        return None

    for raw_line in resp:
        line = raw_line.decode("utf-8", errors="replace")
        sys.stdout.write(line)
        sys.stdout.flush()
        log.write(line)


def pick(label, items, fmt):
    """Prompt user to pick from a list. Returns (name, value)."""
    names = list(items.keys())
    print(f"\n{label}:")
    for i, name in enumerate(names):
        print(f"  {i + 1}. {fmt(name, items[name])}")
    choice = input(f"\nChoose [1]: ").strip() or "1"
    name = names[int(choice) - 1]
    return name, items[name]


def main():
    parser = argparse.ArgumentParser(description="Flash-MoE benchmark runner")
    parser.add_argument("--k", type=int, default=None, help="Experts per layer (default: from model config)")
    parser.add_argument("--tokens", type=int, default=10, help="Max tokens per query (default: 10)")
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument("--log", default=None, help="Log file path (default: auto)")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.join(script_dir, "..")
    models_dir = os.path.join(repo_dir, "models")
    infer_bin = os.path.join(repo_dir, "metal_infer", "infer")

    if not os.path.isfile(infer_bin):
        print("Building infer...")
        subprocess.run(["make", "-C", os.path.join(repo_dir, "metal_infer")], check=True)

    # Pick model
    models = find_models(models_dir)
    if not models:
        print(f"No models found in {models_dir}/")
        print("Each model needs config.json and model_weights.bin.")
        sys.exit(1)

    model_name, model_info = pick("Models", models,
        lambda n, m: f"{n}  (K={m['k']}, {m['num_experts']} experts, {m['num_layers']} layers)")
    model_path = model_info["path"]
    k = args.k if args.k is not None else model_info["k"]

    # Pick benchmark
    benchmarks = load_benchmarks(os.path.join(script_dir, "benchmarks"))
    if not benchmarks:
        print(f"No benchmark files found in {os.path.join(script_dir, 'benchmarks')}/")
        print("Create a text file with one query per line.")
        sys.exit(1)

    bench_name, prompts = pick("Benchmarks", benchmarks,
        lambda n, q: f"{n} ({len(q)} queries)")

    # Setup log
    logs_dir = os.path.join(script_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_path = args.log or os.path.join(logs_dir, f"benchmark_{model_name}_{bench_name}_{int(time.time())}.log")
    log = open(log_path, "w")

    header = (
        f"=== Flash-MoE Benchmark ===\n"
        f"Model:     {model_name}\n"
        f"Path:      {model_path}\n"
        f"Benchmark: {bench_name}\n"
        f"K:         {k}\n"
        f"Tokens:    {args.tokens}\n"
        f"Port:      {args.port}\n"
        f"Time:      {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"===========================\n\n"
    )
    sys.stdout.write(header)
    log.write(header)

    # Start server
    cmd = [infer_bin, "--model", model_path, "--serve", str(args.port), "--k", str(k)]
    log.write(f"[runner] cmd: {' '.join(cmd)}\n")
    log.flush()

    with managed_server(cmd, log) as server:
        server_ready = threading.Event()

        def drain_server():
            for raw_line in iter(server.stdout.readline, b""):
                line = raw_line.decode("utf-8", errors="replace")
                tagged = f"[server] {line}"
                sys.stdout.write(tagged)
                sys.stdout.flush()
                log.write(tagged)
                log.flush()
                if "[serve] Listening on" in line or "Endpoints:" in line:
                    server_ready.set()
            server_ready.set()

        threading.Thread(target=drain_server, daemon=True).start()

        print("Waiting for server...")
        if not wait_for_server(args.port):
            print("Server failed to start!")
            sys.exit(1)
        server_ready.wait(timeout=5)
        print("Server ready.\n")
        log.write("[runner] Server ready\n")
        log.flush()

        # Run queries
        total_start = time.time()
        for i, prompt in enumerate(prompts):
            qn = i + 1
            banner = f"\n--- Query {qn}/{len(prompts)}: \"{prompt[:50]}\" ---\n"
            sys.stdout.write(banner)
            log.write(banner)
            log.flush()

            t0 = time.time()
            log.write(f"[runner] query_start time={t0:.3f}\n")
            send_query(args.port, prompt, args.tokens, f"bench-{qn}", log)
            t1 = time.time()
            log.write(f"[runner] query_end time={t1:.3f} elapsed={t1-t0:.3f}s\n")
            log.flush()

            elapsed = t1 - t0
            sys.stdout.write(f"\n  [{elapsed:.2f}s elapsed]\n")
            log.write(f"[runner] elapsed={elapsed:.3f}s\n\n")
            log.flush()

        total = time.time() - total_start
        footer = f"\n=== Done: {len(prompts)} queries in {total:.1f}s ===\n"
        sys.stdout.write(footer)
        log.write(footer)

    log.close()
    print(f"Log saved to: {log_path}")
    print(f"Run: ./parse_benchmark.py {log_path}")


if __name__ == "__main__":
    main()
