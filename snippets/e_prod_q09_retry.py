import random
import time


class TransientError(Exception):
    """Rate limit, timeout or 5xx: worth retrying."""


def call_with_retries(fn, max_attempts=5, base_delay=0.05, max_delay=2.0):
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except TransientError as err:
            if attempt == max_attempts:
                raise                                  # give up: let a fallback run
            # Exponential backoff with full jitter spreads retries out
            delay = random.uniform(0, min(max_delay, base_delay * 2 ** attempt))
            print(f"attempt {attempt} failed ({err}); retrying in {delay:.3f} s")
            time.sleep(delay)


random.seed(0)
failures = iter([TransientError("429 rate limited"), TransientError("timeout")])


def flaky_llm_call():                                 # stands in for an API call
    err = next(failures, None)
    if err:
        raise err
    return "answer"


def with_fallback(primary, fallback):
    try:
        return call_with_retries(primary)
    except TransientError:
        return fallback()


print(with_fallback(flaky_llm_call, lambda: "answer from the backup model"))
