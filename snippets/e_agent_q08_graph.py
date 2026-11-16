import copy

# What a graph runtime such as LangGraph automates, in plain Python: nodes that
# update shared state, a conditional edge, a checkpoint after every step, and a
# pause for human approval. The "model" is a scripted stand-in, not an LLM.


def call_model(state):
    if "order" not in state:
        return {"action": "lookup_order"}
    if state["order"]["status"] == "damaged" and "refund" not in state:
        return {"action": "refund", "amount": 40}
    return {"action": "finish", "answer": state.get("refund", "No refund needed.")}


class Interrupt(Exception):
    """Raised by a node to pause the run until a person answers."""


def model_node(state, resume):
    state["next"] = call_model(state)


def lookup_node(state, resume):
    state["order"] = {"id": "A-1042", "status": "damaged"}   # stands in for a tool


def refund_node(state, resume):
    # On resume the node runs again from its start, so nothing before the pause
    # may have side effects (LangGraph's interrupt() behaves the same way)
    if resume is None:
        raise Interrupt(f"approve a refund of {state['next']['amount']}?")
    state["refund"] = "Refund issued." if resume == "yes" else "Refund declined."


NODES = {"model": model_node, "lookup_order": lookup_node, "refund": refund_node}
EDGES = {"lookup_order": "model", "refund": "model"}       # tools return to the model


def route(state):                                          # conditional edge
    action = state["next"]["action"]
    return "END" if action == "finish" else action


CHECKPOINTS = {}                                           # thread_id -> (state, node)


def run(thread_id, state=None, resume=None, max_steps=10):
    if state is None:                                      # resume from the checkpoint
        state, node = copy.deepcopy(CHECKPOINTS[thread_id])
    else:
        node = "model"
    for _ in range(max_steps):
        print("  step:", node)
        try:
            NODES[node](state, resume)
        except Interrupt as pause:
            CHECKPOINTS[thread_id] = (copy.deepcopy(state), node)
            return f"paused, {pause}"
        resume = None
        node = route(state) if node == "model" else EDGES[node]
        CHECKPOINTS[thread_id] = (copy.deepcopy(state), node)
        if node == "END":
            return state["next"]["answer"]
    return "stopped at the step limit"


print(run("thread-1", state={"goal": "My order arrived damaged"}))
print("...a person reviews the request, then the same thread resumes...")
print(run("thread-1", resume="yes"))
