# cli/cli_app.py
import argparse
import subprocess
import os

SCRIPT_MAP = {
    "generate": "generate_cli.py",
    "chat": "chat_cli.py",
    "stream": "stream_cli.py",
    "log": "log_cli.py",
    "batch": "generate_batch.py",
    "rerank": "rerank_eval.py",
    "build_jsonl": "rerank_jsonl_builder.py"
}

def main():
    parser = argparse.ArgumentParser(description="CLI Toolkit for LLM Inference")
    parser.add_argument("mode", choices=SCRIPT_MAP.keys(), help="Which tool to run")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments to pass to the tool script")

    args = parser.parse_args()
    script = SCRIPT_MAP[args.mode]

    script_path = os.path.join(os.path.dirname(__file__), script)
    cmd = ["python", script_path] + args.args

    print(f"\nRunning: {script}\n")
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
