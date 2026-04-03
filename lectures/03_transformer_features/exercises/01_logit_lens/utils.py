import torch
import numpy as np


def test_get_residual_streams(fn, model):
    print("Testing get_residual_streams...", end=" ")
    tokens = model.to_tokens("Hello world")
    result = fn(model, tokens)

    n_layers = model.cfg.n_layers
    seq_len = tokens.shape[1]
    d_model = model.cfg.d_model

    if not isinstance(result, torch.Tensor):
        print(f"FAIL (expected torch.Tensor, got {type(result).__name__})")
        return
    expected_shape = (seq_len, n_layers, d_model)
    if result.shape != expected_shape:
        print(f"FAIL (expected shape {expected_shape}, got {tuple(result.shape)})")
        return
    if result.abs().sum() == 0:
        print("FAIL (all zeros, did you forget to extract activations?)")
        return
    print("PASS")


def test_logit_lens(fn, model):
    print("Testing logit_lens...", end=" ")
    tokens = model.to_tokens("The cat sat on")
    top_tokens, kl_divs = fn(model, tokens)

    n_layers = model.cfg.n_layers
    seq_len = tokens.shape[1]

    if not isinstance(top_tokens, torch.Tensor) or not isinstance(kl_divs, torch.Tensor):
        print(f"FAIL (expected two torch.Tensors)")
        return
    expected_shape = (seq_len, n_layers)
    if top_tokens.shape != expected_shape:
        print(f"FAIL (top_tokens shape {tuple(top_tokens.shape)}, expected {expected_shape})")
        return
    if kl_divs.shape != expected_shape:
        print(f"FAIL (kl_divs shape {tuple(kl_divs.shape)}, expected {expected_shape})")
        return
    if top_tokens.dtype not in (torch.int64, torch.int32):
        print(f"FAIL (top_tokens should be integer dtype, got {top_tokens.dtype})")
        return
    if (kl_divs < -1e-4).any():
        print(f"FAIL (KL divergence should be non-negative, got min {kl_divs.min():.4e})")
        return
    last_layer_kl = kl_divs[:, -1]
    if not torch.allclose(last_layer_kl, torch.zeros_like(last_layer_kl), atol=1e-3):
        print(f"FAIL (KL divergence at last layer should be ~0, got max {last_layer_kl.max():.4f})")
        return
    print("PASS")


def test_top_k_tokens(fn, model):
    print("Testing top_k_tokens...", end=" ")
    tokens = model.to_tokens("Hello")
    _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
    residual = cache[f"blocks.{model.cfg.n_layers - 1}.hook_resid_post"]
    last_pos = residual[-1]

    result = fn(model, last_pos, k=5)

    if not isinstance(result, list):
        print(f"FAIL (expected list, got {type(result).__name__})")
        return
    if len(result) != 5:
        print(f"FAIL (expected 5 items, got {len(result)})")
        return
    if not all(isinstance(item, tuple) and len(item) == 2 for item in result):
        print("FAIL (each item should be a (token_str, probability) tuple)")
        return
    probs = [p for _, p in result]
    if not all(0 <= p <= 1 for p in probs):
        print("FAIL (probabilities should be between 0 and 1)")
        return
    if probs != sorted(probs, reverse=True):
        print("FAIL (results should be sorted by descending probability)")
        return
    print("PASS")
