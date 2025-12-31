#!/usr/bin/env python3
"""
Minimal Python example that calls the local llama.cpp server `/infill` endpoint
and prints the JSON response.

Usage:
  python examples/copilot/tool_calling.py

Notes:
  - This script prefers the `requests` library; if not available it falls back to
    calling `curl` via subprocess.
  - For real code prefer `libcurl` bindings or `requests` with proper error handling.
"""

import json
import sys
import subprocess
import requests



def post_with_requests(url, payload):
    r = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=30)
    r.raise_for_status()
    return r.text


def post_with_curl(url, payload):
    # Use curl as a fallback when `requests` is unavailable
    cmd = [
        "curl",
        "--silent",
        "--no-buffer",
        "--request", "POST",
        "--header", "Content-Type: application/json",
        "--data", json.dumps(payload),
        url,
    ]
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    return out.decode("utf-8")


def main():
    url = "http://127.0.0.1:8033/v1/chat/completions"

    payload = {
        "prompt": "Hello from Python",
        "n_predict": 32,
        "stream": False,
    }

    try:
        if requests is not None:
            resp_text = post_with_requests(url, payload)
        else:
            resp_text = post_with_curl(url, payload)
    except subprocess.CalledProcessError as e:
        print("External command failed:", e, file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print("Request failed:", e, file=sys.stderr)
        sys.exit(1)

    # Try to pretty-print JSON, otherwise print raw
    try:
        obj = json.loads(resp_text)
        print(json.dumps(obj, indent=2, ensure_ascii=False))
    except Exception:
        print(resp_text)


if __name__ == "__main__":
    main()