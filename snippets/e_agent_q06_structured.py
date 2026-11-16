import json
from dataclasses import dataclass


@dataclass
class Ticket:
    title: str
    priority: str          # one of PRIORITIES
    component: str


PRIORITIES = {"low", "medium", "high"}


def parse_ticket(raw: str) -> Ticket:
    data = json.loads(raw)                              # may raise on bad JSON
    missing = {"title", "priority", "component"} - data.keys()
    if missing:
        raise ValueError(f"missing fields: {sorted(missing)}")
    if data["priority"] not in PRIORITIES:
        raise ValueError(f"priority must be one of {sorted(PRIORITIES)}")
    return Ticket(data["title"], data["priority"], data["component"])


# Stands in for an LLM: two flawed replies, then a valid one
replies = iter([
    'Sure! {"title": "Login fails"',                                  # broken
    '{"title": "Login fails", "priority": "urgent", "component": "auth"}',
    '{"title": "Login fails", "priority": "high", "component": "auth"}',
])
fake_model = lambda prompt: next(replies)


def extract_ticket(text, max_attempts=3):
    prompt = f"Return JSON with title, priority, component for: {text}"
    for attempt in range(1, max_attempts + 1):
        raw = fake_model(prompt)
        try:
            return parse_ticket(raw)
        except (json.JSONDecodeError, ValueError) as err:
            print(f"attempt {attempt} rejected: {err}")
            prompt += f"\nYour last reply was invalid: {err}. Try again."
    raise RuntimeError("no valid output")


print(extract_ticket("Users cannot log in since this morning's release."))
