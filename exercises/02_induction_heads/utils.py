import torch
from torch import Tensor
from transformer_lens import HookedTransformer, ActivationCache


TEST_STRING = "The quick brown fox jumps over the lazy dog.The quick brown fox jumps over the lazy dog."


def _ref_average_over_condition(tensor, condition):
    n_head, n_query, n_key = tensor.shape
    return torch.tensor([
        sum(tensor[h, q, k] for q in range(n_query) for k in range(n_key) if condition(q, k))
        / sum(1 for q in range(n_query) for k in range(n_key) if condition(q, k))
        for h in range(n_head)
    ])


def _ref_heads_above_threshold(cache, condition, threshold):
    result = []
    for layer, pattern in enumerate(cache.stack_activation("pattern")):
        scores = _ref_average_over_condition(pattern, condition)
        for head, score in enumerate(scores):
            if score > threshold:
                result.append(f"L{layer+1}H{head}")
    return result


def _find_repeating_rows(tokens):
    last_occurrence = {}
    repeats = {}
    for pos, token in enumerate(tokens[0]):
        tid = token.item()
        if tid in last_occurrence:
            repeats[pos] = last_occurrence[tid]
        last_occurrence[tid] = pos
    return repeats


def test_average_over_condition(fn):
    print("Testing average_over_condition...", end=" ")
    tensor = torch.tensor([
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        [[0.7, 0.8, 0.9], [0.1, 0.2, 0.3]],
    ])
    condition = lambda q, k: q == k
    result = fn(tensor, condition)
    expected = _ref_average_over_condition(tensor, condition)
    if not isinstance(result, torch.Tensor):
        print(f"FAIL (expected Tensor, got {type(result).__name__})")
        return
    if result.shape != expected.shape:
        print(f"FAIL (expected shape {tuple(expected.shape)}, got {tuple(result.shape)})")
        return
    if not torch.allclose(result.float(), expected.float(), atol=1e-4):
        print(f"FAIL (values don't match, expected {expected}, got {result})")
        return
    print("PASS")


def test_head_detectors(current_fn, prev_fn, first_fn, induction_fn, model):
    print("Testing head detectors...", end=" ")
    tokens = model.to_tokens(TEST_STRING)
    _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
    threshold = 0.2

    expected_current = _ref_heads_above_threshold(cache, lambda q, k: q == k, threshold)
    expected_prev = _ref_heads_above_threshold(cache, lambda q, k: q == k + 1, threshold)
    expected_first = _ref_heads_above_threshold(cache, lambda q, k: k == 0, threshold)

    got_current = current_fn(cache, threshold=threshold)
    got_prev = prev_fn(cache, threshold=threshold)
    got_first = first_fn(cache, threshold=threshold)

    for name, got, expected in [
        ("current_attn_detector", got_current, expected_current),
        ("prev_attn_detector", got_prev, expected_prev),
        ("first_attn_detector", got_first, expected_first),
    ]:
        if got != expected:
            print(f"FAIL ({name}: expected {expected}, got {got})")
            return

    repeat_dict = _find_repeating_rows(tokens)
    def induction_cond(q, k):
        return q in repeat_dict and repeat_dict[q] + 1 == k
    expected_induction = _ref_heads_above_threshold(cache, induction_cond, threshold)
    got_induction = induction_fn(cache, tokens, threshold=threshold)
    if got_induction != expected_induction:
        print(f"FAIL (induction_attn_detector: expected {expected_induction}, got {got_induction})")
        return
    print("PASS")


def test_logit_attribution(fn, model):
    print("Testing logit_attribution...", end=" ")
    text = "The quick brown fox jumps over the lazy dog."
    tokens = model.to_tokens(text)
    _, cache = model.run_with_cache(tokens, remove_batch_dim=True)

    result = fn(tokens, model, cache, token_position=3)

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    if not isinstance(result, torch.Tensor):
        print(f"FAIL (expected Tensor, got {type(result).__name__})")
        return
    if result.shape != (n_layers, n_heads):
        print(f"FAIL (expected shape ({n_layers}, {n_heads}), got {tuple(result.shape)})")
        return

    # Verify against reference
    results_ref = torch.stack([cache[f"blocks.{i}.attn.hook_result"] for i in range(n_layers)], dim=1)
    results_ref = results_ref[3]
    logits_ref = model.unembed(results_ref)
    next_token = tokens[0, 4]
    expected = logits_ref[:, :, next_token]
    if not torch.allclose(result, expected, atol=1e-4):
        print(f"FAIL (values don't match, max diff: {(result - expected).abs().max():.2e})")
        return
    print("PASS")
