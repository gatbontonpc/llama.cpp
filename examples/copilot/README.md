# llama.cpp/example/copilot

The purpose of this example is to demonstrate a minimal usage of llama.cpp to create a simple terminal copilot like chat program using llama-server.

```bash
./llama-copilot -n 2048 -s -u http://localhost:8033 Debug my last bash command
```

```bash
./llama-server -hf unsloth/Qwen3-4B-Instruct-2507-GGUF:Q8_0 -ngl 99 -fa on --port 8033 -c 2048 -np 2  --cache-reuse 256
```

## Current capabilities
1. Streams SSE from llama-server
1. Gathers context from bash history, cwd, shellname
1. Reads from stdin pipe or file for additional context

## Todo capabilities
1. Add slot pinning for better kv cache reuse
1. Add session history
1. Add tmux integrations
1. Add function calling
1. Add git aware prompts (diffs, blame, staged changes)