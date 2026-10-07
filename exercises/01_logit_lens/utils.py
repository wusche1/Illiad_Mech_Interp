import torch

PROMPT = "The Eiffel Tower is located in the city of"


def _reference(model):
    tokens = model.to_tokens(PROMPT)
    logits, cache = model.run_with_cache(tokens, remove_batch_dim=True)
    resids = torch.stack([cache["blocks.0.hook_resid_pre"]] + [cache[f"blocks.{l}.hook_resid_post"] for l in range(model.cfg.n_layers)])
    log_probs = model.unembed(model.ln_final(resids)).log_softmax(-1)
    return tokens, logits[0], cache, resids, log_probs


def test_unembed(fn, model):
    print("Testing unembed...", end=" ")
    tokens, logits, cache, resids, _ = _reference(model)
    final = cache[f"blocks.{model.cfg.n_layers - 1}.hook_resid_post"]
    out = fn(model, final)
    if out.shape != logits.shape:
        print(f"FAIL (expected shape {tuple(logits.shape)}, got {tuple(out.shape)})")
        return
    if torch.allclose(out, model.unembed(final), atol=1e-3):
        print("FAIL (output matches W_U @ resid without the final LayerNorm: apply model.ln_final first)")
        return
    if not torch.allclose(out, logits, atol=1e-3):
        print(f"FAIL (does not reproduce the model's logits, max difference {(out - logits).abs().max():.3f})")
        return
    if fn(model, final[-1]).shape != (model.cfg.d_vocab,):
        print("FAIL (should also work on a single (d_model,) vector)")
        return
    print("PASS")


def test_get_resid_stack(fn, model):
    print("Testing get_resid_stack...", end=" ")
    tokens, _, cache, resids, _ = _reference(model)
    out = fn(model, tokens)
    if out.shape != resids.shape:
        print(f"FAIL (expected shape {tuple(resids.shape)} = (n_layers + 1, seq_len, d_model), got {tuple(out.shape)})")
        return
    if not torch.allclose(out[0], resids[0], atol=1e-4):
        print("FAIL (index 0 should be the embeddings, blocks.0.hook_resid_pre)")
        return
    if not torch.allclose(out, resids, atol=1e-4):
        bad = [i for i in range(len(out)) if not torch.allclose(out[i], resids[i], atol=1e-4)]
        print(f"FAIL (wrong residual stream at indices {bad}; index l+1 should be blocks.l.hook_resid_post)")
        return
    print("PASS")


def test_logit_lens(fn, model):
    print("Testing logit_lens...", end=" ")
    tokens, logits, _, _, log_probs = _reference(model)
    out = fn(model, tokens)
    if out.shape != log_probs.shape:
        print(f"FAIL (expected shape {tuple(log_probs.shape)} = (n_layers + 1, seq_len, d_vocab), got {tuple(out.shape)})")
        return
    if not torch.allclose(out.exp().sum(-1), torch.ones_like(out[..., 0]), atol=1e-3):
        print("FAIL (output should be log-probabilities: exp() should sum to 1 over the vocabulary)")
        return
    if not torch.allclose(out[-1], logits.log_softmax(-1), atol=1e-3):
        print("FAIL (the last layer should reproduce the model's own log-probabilities)")
        return
    if not torch.allclose(out, log_probs, atol=1e-3):
        print("FAIL (intermediate layers do not match the reference)")
        return
    print("PASS")


def test_kl_from_final(fn):
    print("Testing kl_from_final...", end=" ")
    # one position, two layers, vocabulary of size 2: P_0 = [0.9, 0.1], P_final = [0.5, 0.5]
    log_probs = torch.tensor([[[0.9, 0.1]], [[0.5, 0.5]]]).log()
    expected = torch.tensor([[0.5108], [0.0]])
    out = fn(log_probs)
    if out.shape != (2, 1):
        print(f"FAIL (expected shape (n_layers + 1, seq_len) = (2, 1), got {tuple(out.shape)})")
        return
    if torch.allclose(out[0], torch.tensor([0.3681]), atol=1e-3):
        print("FAIL (you computed KL(P_layer || P_final); the formula is KL(P_final || P_layer))")
        return
    if not torch.allclose(out, expected, atol=1e-3):
        print(f"FAIL (for P_layer = [0.9, 0.1], P_final = [0.5, 0.5] expected KL = 0.511, got {out[0, 0]:.3f})")
        return
    print("PASS")
