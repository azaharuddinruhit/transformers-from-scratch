def monthly_cost(requests_per_day, input_tokens, output_tokens,
                 price_in_per_m, price_out_per_m, cached_share=0.0,
                 cached_discount=0.9):
    """Prices are per million tokens. cached_share: fraction of input tokens
    served from a prompt cache, billed at (1 - cached_discount)."""
    effective_in = input_tokens * (1 - cached_share * cached_discount)
    per_request = (effective_in * price_in_per_m
                   + output_tokens * price_out_per_m) / 1e6
    return per_request, per_request * requests_per_day * 30


# Example prices only; use your provider's current price list
prices = dict(price_in_per_m=3.0, price_out_per_m=15.0)

for label, kwargs in [
    ("baseline", dict(input_tokens=6000, output_tokens=500)),
    ("shorter prompt", dict(input_tokens=3000, output_tokens=500)),
    ("+ prompt cache 70%", dict(input_tokens=3000, output_tokens=500,
                                cached_share=0.7)),
    ("+ capped output", dict(input_tokens=3000, output_tokens=250,
                             cached_share=0.7)),
]:
    per_req, month = monthly_cost(20_000, **kwargs, **prices)
    print(f"{label:20} ${per_req:.4f} per request   ${month:,.0f} per month")
