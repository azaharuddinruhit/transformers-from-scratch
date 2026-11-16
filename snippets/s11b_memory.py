def kv_cache_bytes(L, T, d_model, bytes_per_number=2, B=1):
    return 2 * L * T * d_model * bytes_per_number * B  # 2 = keys and values


def weight_bytes(N, bytes_per_param=2):
    return N * bytes_per_param


def response_seconds(t_first_token, n_tokens, tokens_per_second):
    return t_first_token + n_tokens / tokens_per_second


def rough_speed(N, P, B, F, R, bytes_per_param=2):
    """Order-of-magnitude bounds; real systems are slower."""
    m_weights = weight_bytes(N, bytes_per_param)
    throughput = min(F / (2 * N), B * R / m_weights)    # tokens/s, all requests
    t_first = max(2 * N * P * B / F, m_weights / R)     # seconds
    return throughput, throughput / B, t_first


print(f"KV cache, GPT-2 small, T=1024: {kv_cache_bytes(12, 1024, 768):,} bytes")
print(f"weights, GPT-2 small:          {weight_bytes(124_439_808):,} bytes")
print(f"KV cache, L=32, d=4096, T=8192: {kv_cache_bytes(32, 8192, 4096):,} bytes")
print("response time:", response_seconds(0.4, 200, 50), "s")

# Generic hardware numbers, not a specific product
F, R = 100e12, 1e12            # 100 TFLOP/s of compute, 1 TB/s of bandwidth
for B in (1, 16, 256):
    total, per_request, ttft = rough_speed(N=7e9, P=1000, B=B, F=F, R=R)
    print(f"7B model, batch {B:3}: {total:7.0f} tok/s total, "
          f"{per_request:5.1f} tok/s per request, first token ~{ttft:.2f} s")
