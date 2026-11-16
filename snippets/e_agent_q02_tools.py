import json

# 1. Describe each tool: name, description and a JSON schema for its arguments
TOOLS = {
    "get_order_status": {
        "description": "Look up the delivery status of an order by its ID.",
        "parameters": {"type": "object",
                       "properties": {"order_id": {"type": "string"}},
                       "required": ["order_id"]},
        "run": lambda order_id: {"order_id": order_id, "status": "shipped",
                                 "eta": "2 days"},
    },
}


def fake_model(messages, tools):
    """Stands in for an LLM API call. Real providers return the same shape:
    either a tool call (name + JSON arguments) or a final text answer."""
    if messages[-1]["role"] == "user":
        return {"tool_call": {"name": "get_order_status",
                              "arguments": json.dumps({"order_id": "A-1042"})}}
    result = json.loads(messages[-1]["content"])
    return {"text": f"Order {result['order_id']} has {result['status']}; "
                    f"it should arrive in {result['eta']}."}


def validate(args, schema):
    missing = [k for k in schema["required"] if k not in args]
    if missing:
        raise ValueError(f"missing arguments: {missing}")


def run_agent(user_message, max_steps=5):
    messages = [{"role": "user", "content": user_message}]
    specs = {n: {k: v for k, v in t.items() if k != "run"} for n, t in TOOLS.items()}
    for step in range(max_steps):                    # always cap the loop
        reply = fake_model(messages, specs)
        if "text" in reply:                          # 4. final answer: stop
            return reply["text"]
        call = reply["tool_call"]                    # 2. model asks for a tool
        tool = TOOLS[call["name"]]
        args = json.loads(call["arguments"])
        validate(args, tool["parameters"])           # 3. your code checks ...
        result = tool["run"](**args)                 #    ... and runs it
        print(f"step {step}: {call['name']}({args}) -> {result}")
        messages.append({"role": "tool", "content": json.dumps(result)})
    return "Stopped: step limit reached."


print(run_agent("Where is my order A-1042?"))
