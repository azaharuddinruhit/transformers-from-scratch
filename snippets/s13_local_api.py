import json
import time
import urllib.request

# Ollama and vLLM both serve the OpenAI chat-completions API, so one client works
# with either: only the base URL and the model name change.
SERVERS = {
    "ollama": ("http://localhost:11434/v1", "gpt-oss:20b"),      # ollama pull gpt-oss:20b
    "vllm": ("http://localhost:8000/v1", "openai/gpt-oss-20b"),  # vllm serve openai/gpt-oss-20b
}
BASE_URL, MODEL = SERVERS["ollama"]


def call(path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(BASE_URL + path, data=data,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)


try:
    models = [m["id"] for m in call("/models")["data"]]
except OSError as error:
    raise SystemExit(f"No server at {BASE_URL}: start Ollama or vLLM first ({error})")
print("models on the server:", ", ".join(models))

start = time.perf_counter()
reply = call("/chat/completions", {
    "model": MODEL,
    "messages": [{"role": "user", "content": "Explain the KV cache in one sentence."}],
    "temperature": 0.2,
    "max_tokens": 1000,     # reasoning models spend tokens thinking before they answer
})
seconds = time.perf_counter() - start
used = reply["usage"]
print("reply:", reply["choices"][0]["message"]["content"])
print(f"{used['prompt_tokens']} prompt tokens, {used['completion_tokens']} output tokens "
      f"in {seconds:.1f} s, {used['completion_tokens'] / seconds:.0f} tokens/s including prefill")
