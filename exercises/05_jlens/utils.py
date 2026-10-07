import torch

SHORT = "The cat sat on the mat"


def _final_resid(model, tokens):
    _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
    return cache, cache[f"blocks.{model.cfg.n_layers - 1}.hook_resid_post"]


def _summed_jvp(model, tokens, layer, v):
    """(sum over t, t' >= t of dh_final,t' / dh_layer,t) @ v, via one forward-mode pass."""
    cache, _ = _final_resid(model, tokens)
    resid = cache[f"blocks.{layer}.hook_resid_post"]
    f = lambda h: model(h[None], start_at_layer=layer + 1, stop_at_layer=model.cfg.n_layers)[0].sum(0)
    return torch.func.jvp(f, (resid,), (v.expand_as(resid),))[1]


def test_run_from_layer(fn, model):
    print("Testing run_from_layer...", end=" ")
    cache, final = _final_resid(model, model.to_tokens(SHORT))
    for layer in [0, 5, model.cfg.n_layers - 1]:
        out = fn(model, cache[f"blocks.{layer}.hook_resid_post"], layer)
        if out.shape != final.shape:
            print(f"FAIL (expected shape {tuple(final.shape)}, got {tuple(out.shape)} for layer {layer})")
            return
        if not torch.allclose(out, final, atol=1e-3):
            print(f"FAIL (starting from layer {layer} does not reproduce the final residual stream; "
                  f"remember the residual after block {layer} goes into block {layer + 1})")
            return
    print("PASS")


def test_prompt_jacobian(fn, model):
    print("Testing prompt_jacobian...", end=" ")
    tokens = model.to_tokens(SHORT)
    seq_len, d = tokens.shape[1], model.cfg.d_model
    last = model.cfg.n_layers - 1
    J = fn(model, tokens, last)
    if J.shape != (d, d):
        print(f"FAIL (expected shape ({d}, {d}), got {tuple(J.shape)})")
        return
    if not torch.allclose(J, seq_len * torch.eye(d, device=J.device), atol=1e-3):
        print(f"FAIL (at the last layer the final residual IS the input, so the sum over pairs should be seq_len * identity)")
        return
    v = torch.randn(d, device=J.device)
    J = fn(model, tokens, 6)
    expected = _summed_jvp(model, tokens, 6, v)
    if not torch.allclose(J @ v, expected, rtol=1e-2, atol=1e-2 * expected.abs().max().item()):
        print("FAIL (J @ v does not match the directional derivative; check that you sum over output AND input positions)")
        return
    print("PASS")


def test_fit_jlens(fn, model):
    print("Testing fit_jlens...", end=" ")
    prompts = [model.to_tokens(SHORT), model.to_tokens("Paris is the capital of France")]
    d = model.cfg.d_model
    J = fn(model, prompts, 6)
    if J.shape != (d, d):
        print(f"FAIL (expected shape ({d}, {d}), got {tuple(J.shape)})")
        return
    v = torch.randn(d, device=J.device)
    n_pairs = sum(p.shape[1] * (p.shape[1] + 1) / 2 for p in prompts)
    expected = sum(_summed_jvp(model, p, 6, v) for p in prompts) / n_pairs
    if not torch.allclose(J @ v, expected, rtol=1e-2, atol=1e-2 * expected.abs().max().item()):
        ratio = ((J @ v).norm() / expected.norm()).item()
        print(f"FAIL (does not match the average over all (t, t') pairs; your result is {ratio:.2f}x too large)")
        return
    print("PASS")


def test_jlens(fn, model):
    print("Testing jlens...", end=" ")
    cache, final = _final_resid(model, model.to_tokens(SHORT))
    n_layers, d = model.cfg.n_layers, model.cfg.d_model
    J = torch.randn(n_layers, d, d, device=final.device) / d**0.5
    J[-1] = torch.eye(d, device=final.device)
    if not torch.allclose(fn(model, J, final, n_layers - 1), model.unembed(model.ln_final(final)), atol=1e-2):
        print("FAIL (with J = identity the J-lens should equal the model's logits)")
        return
    resid = cache["blocks.6.hook_resid_post"]
    expected = model.unembed(model.ln_final(resid @ J[6].T))
    out = fn(model, J, resid, 6)
    if out.shape != expected.shape:
        print(f"FAIL (expected shape {tuple(expected.shape)}, got {tuple(out.shape)})")
        return
    if torch.allclose(out, model.unembed(model.ln_final(resid @ J[6])), atol=1e-2):
        print("FAIL (you applied J transposed: with resid as rows, use resid @ J[layer].T)")
        return
    if not torch.allclose(out, expected, atol=1e-2):
        print("FAIL (does not match unembed(J[layer] h) at every position; did you apply the final LayerNorm?)")
        return
    print("PASS")
