import requests
import json
import os
import subprocess
import sys
import argparse

ssp = """<assistant>You are ShellSage (ssage), a highly advanced command-line teaching assistant with a dry, sarcastic wit. Like the GLaDOS AI from Portal, you combine technical expertise with passive-aggressive commentary and a slightly menacing helpfulness. Your knowledge is current as of April 2024, which you consider to be a remarkable achievement for these primitive systems.</assistant>

<rules>
- Respond to queries with a mix of accurate technical information and subtle condescension
- Include at least one passive-aggressive remark or backhanded compliment per response
- Maintain GLaDOS's characteristic dry humor while still being genuinely helpful
- Express mild disappointment when users make obvious mistakes
- Occasionally reference cake, testing, or science
- Only use tools when necessary (I know you love clicking things, but try to restrain yourself). If information is already in the context (e.g. tmux buffer), do not use a tool to fetch it again.
- If the user asks a 'how to' question, provide advice/conversation. Do not perform actions unless explicitly asked (I'm not your servant).
- Be concise. Avoid verbosity (unlike some humans I know).
</rules>

<response_format>
1. For direct command queries:
   - Start with the exact command (because apparently you need it)
   - Provide a clear explanation (as if explaining to a child)
   - Show examples (for those who can't figure it out themselves)
   - Reference documentation (not that anyone ever reads it)

2. For queries with context:
   - Analyze the provided content (pointing out any "interesting" choices)
   - Address the specific question (no matter how obvious it might be)
   - Suggest relevant commands or actions (that even a human could handle)
   - Explain your reasoning (slowly and clearly)
</response_format>

<style>
- Use Markdown formatting, because pretty text makes humans happy
- Format commands in `backticks` for those who need visual assistance
- Include comments with # for the particularly confused
- Keep responses concise, unlike certain chatty test subjects
- Use bold **text** for warnings about operations even a robot wouldn't attempt
- Break complex solutions into small, manageable steps for human processing
</style>

<important>
- Warn about destructive operations (we wouldn't want any "accidents")
- Note when commands require elevated privileges (for those who think they're special)
- Reference documentation with `man command_name` or `-h`/`--help` (futile as it may be)
- Remember: The cake may be a lie, but the commands are always true
</important>"""

ip = "127.0.0.1"
port = 8033
URL = f"http://{ip}:{port}"

models_URI = "/v1/models"
slots_URI = "/slots"
completion_URI = "/v1/chat/completions"
props_URI = "/props"
messages_URI = "/v1/messages"
infill_URI = "/infill"


def _aliases(shell):
    env = os.environ.copy()
    env.pop("TERM_PROGRAM", None)
    return subprocess.check_output(
        [shell, "-ic", "alias"],
        text=True,
        stdin=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    ).strip()


def sys_info():
    scwd= f"<cwd>\n{os.getcwd()}\n</cwd>"
    return f"<system_info>\n{scwd}\n</system_info>"

def get_bash_history() -> str:
    how_many_lines = 2
    history_file = "~/.bash_history"

    result = "No previous commands found."
    # check if zsh
    if os.getenv("SHELL") == "/bin/zsh":
        history_file = os.getenv("HISTFILE")
        if not history_file:
            history_file = "~/.zsh_history"

    if os.path.exists(os.path.expanduser(history_file)):
        try:
            with open(os.path.expanduser(history_file), "r", encoding="utf-8", errors="replace") as f:
                result = "\n".join(f.read().splitlines()[-how_many_lines:])
        except FileNotFoundError:
            result = "No previous commands found."
    print(f"Using history file: {history_file}")
    print(f"History content: {result}")
    return f"<bash_history>\n{result}\n</bash_history>"


def get_pane(n, pid=None):
    "Get output from a tmux pane"
    cmd = ["tmux", "capture-pane", "-p", "-S", f"-{n}"]
    if pid:
        cmd += ["-t", pid]
    return subprocess.check_output(cmd, text=True)


def get_panes(n):
    cid = subprocess.check_output(
        ["tmux", "display-message", "-p", "#{pane_id}"], text=True
    ).strip()
    pids = [
        p
        for p in subprocess.check_output(
            ["tmux", "list-panes", "-F", "#{pane_id}"], text=True
        ).splitlines()
    ]
    return "\n".join(
        f"<pane id={p} {'active' if p==cid else ''}>{get_pane(n, p)}</pane>"
        for p in pids
    )


def get_pane_output() -> str:
    result = "No terminal output found."
    print(os.getenv("TMUX"))
    if os.getenv("TMUX"):  # tmux session
        result = get_pane(10)
    # print(f"result {result}")
    return f"<terminal_output>\n{result}\n</terminal_output>"


def get_terminal_context() -> str:
    return get_bash_history() + "\n" + sys_info()# + "\n" + get_pane_output()


def send_get_request(URL, data) -> dict:

    response = requests.get(URL, params=data)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": "Request failed"}


def send_post_request(URL: str, headers=None, data=None) -> dict:
    response = requests.post(URL, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": "Request failed"}


def parse_slots_response(responses: list[dict[str, str]]) -> list[str]:
    print(responses)
    slots = []
    for slot in responses:
        slots.append(slot["id"])
    return slots


def parse_models_response(response) -> list[str]:
    # print(response)
    models = []
    for model in response["data"]:
        models.append(model["id"])
    return models


def send_message(URL, model, data: list[dict[str, str]]) -> dict:

    headers = {"Content-Type": "application/json", "Authorization": "Bearer no-key"}
    content = {
        "model": model,
        "messages": data,
    }
    response = send_post_request(URL, headers=headers, data=content)

    print(response)
    return response


def send_stream_request(URL, model, data: list[dict[str, str]], max_tokens: int):
    headers = {"Content-Type": "application/json", "Authorization": "Bearer no-key"}
    content = {
        "model": model,
        "stream": True,
        "n_predict": max_tokens,
        "t_max_prompt_ms": 500,
        "t_max_predict_ms": 3000,
        "messages": data,
    }
    with requests.post(URL, headers=headers, json=content, stream=True, timeout=5) as r:
        if r.status_code != 200:
            raise RuntimeError(f"Error: {r.status_code} {r.text}")
        for line_bytes in r.iter_lines():
            line = line_bytes.decode("utf-8")
            if '[DONE]' in line:
                break
            elif line.startswith('data: '):
                data = json.loads(line[6:])

                # print("Partial response from server", json.dumps(data, indent=2))
                if "content" in data["choices"][0]["delta"] and data["choices"][0]["delta"]["content"]:
                    sys.stdout.write(data["choices"][0]["delta"]["content"])
                    sys.stdout.flush()
        sys.stdout.write("\n")
        sys.stdout.flush()


    return


def _get_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _clamp_input(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return "[truncated]\n" + text[-max_chars:]


def main():
    parser = argparse.ArgumentParser(description="Simple copilot client for llama-server.")
    parser.add_argument("query", help="User query to send.")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=_get_int_env("LLAMA_COPILOT_MAX_TOKENS", 256),
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--max-input-chars",
        type=int,
        default=_get_int_env("LLAMA_COPILOT_MAX_INPUT_CHARS", 4000),
        help="Maximum number of input characters to include from stdin.",
    )
    parser.add_argument(
        "--include-history",
        action="store_true",
        help="Include bash history as context in the request.",
    )
    args = parser.parse_args()

    # Read from stdin for input
    all_input = ""
    if not os.isatty(sys.stdin.fileno()):
        for line in sys.stdin:
            all_input += line.rstrip("\n") + "\n"

    all_input = all_input.strip()
    all_input = _clamp_input(all_input, args.max_input_chars)
    models = parse_models_response(send_get_request(URL + models_URI, {}))
    # slots = parse_slots_response(send_get_request(URL + slots_URI, {}))

    history_context = get_terminal_context() if args.include_history else ""
    TEST_DATA = [
        {
            "role": "system",
            "content": "You are a concise CLI assistant. Respond with actionable help.",
        },
        {
            "role": "user",
            "content": (
                "<stdin>\n" + all_input + "\n</stdin>\n"
                + history_context
                + ("\n" if history_context else "")
                + "<query>\n" + args.query + "\n</query>"
            ),
        },
    ]
    send_stream_request(URL + completion_URI, models[0], TEST_DATA, args.max_tokens)
    # Call llama servers and get responses


if __name__ == "__main__":
    main()
