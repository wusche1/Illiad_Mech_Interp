import torch
import torch.nn.functional as F


def test_standard_sae(cls, activations, device):
    print("Testing StandardSAE...", end=" ")
    d_model, d_dict = 512, 2048
    sae = cls(d_model, d_dict).to(device)

    # Check parameters exist with correct shapes
    if sae.W_enc.shape != (d_dict, d_model):
        print(f"FAIL (W_enc shape {tuple(sae.W_enc.shape)}, expected ({d_dict}, {d_model}))")
        return
    if sae.W_dec.shape != (d_model, d_dict):
        print(f"FAIL (W_dec shape {tuple(sae.W_dec.shape)}, expected ({d_model}, {d_dict}))")
        return
    if sae.b_enc.shape != (d_dict,):
        print(f"FAIL (b_enc shape {tuple(sae.b_enc.shape)}, expected ({d_dict},))")
        return
    if sae.b_dec.shape != (d_model,):
        print(f"FAIL (b_dec shape {tuple(sae.b_dec.shape)}, expected ({d_model},))")
        return

    # Forward pass
    batch = activations[:16].to(device)
    x_hat, z, loss = sae(batch, l1_coeff=1.0)
    if x_hat.shape != (16, d_model):
        print(f"FAIL (x_hat shape {tuple(x_hat.shape)}, expected (16, {d_model}))")
        return
    if z.shape != (16, d_dict):
        print(f"FAIL (z shape {tuple(z.shape)}, expected (16, {d_dict}))")
        return
    if (z < -1e-6).any():
        print(f"FAIL (z has negative values, ReLU should make all non-negative)")
        return
    if loss.ndim != 0:
        print(f"FAIL (loss should be scalar, got shape {tuple(loss.shape)})")
        return

    # Verify loss computation
    mse = (batch - x_hat).pow(2).mean()
    l1 = z.abs().mean()
    expected_loss = mse + 1.0 * l1
    if not torch.allclose(loss, expected_loss, atol=1e-4):
        print(f"FAIL (loss={loss.item():.4f}, expected MSE+L1={expected_loss.item():.4f})")
        return

    # Test normalize_decoder
    sae.normalize_decoder()
    col_norms = sae.W_dec.data.norm(dim=0)
    if not torch.allclose(col_norms, torch.ones_like(col_norms), atol=1e-5):
        print(f"FAIL (after normalize_decoder, column norms should be 1, got mean={col_norms.mean():.4f})")
        return

    print("PASS")


def test_topk_sae(cls, activations, device):
    print("Testing TopKSAE...", end=" ")
    d_model, d_dict, k = 512, 2048, 32
    sae = cls(d_model, d_dict, k).to(device)

    batch = activations[:16].to(device)
    x_hat, z, loss = sae(batch)

    if x_hat.shape != (16, d_model):
        print(f"FAIL (x_hat shape {tuple(x_hat.shape)}, expected (16, {d_model}))")
        return
    if z.shape != (16, d_dict):
        print(f"FAIL (z shape {tuple(z.shape)}, expected (16, {d_dict}))")
        return

    # Check exactly k nonzeros per sample
    nnz_per_sample = (z > 0).sum(dim=-1)
    if not (nnz_per_sample == k).all():
        print(f"FAIL (expected exactly {k} nonzeros per sample, got {nnz_per_sample.tolist()})")
        return

    # Loss should be MSE only
    mse = (batch - x_hat).pow(2).mean()
    if not torch.allclose(loss, mse, atol=1e-4):
        print(f"FAIL (loss={loss.item():.4f} should equal MSE={mse.item():.4f}, no sparsity penalty)")
        return

    print("PASS")


def test_batch_topk_sae(cls, activations, device):
    print("Testing BatchTopKSAE...", end=" ")
    d_model, d_dict, k = 512, 2048, 32
    sae = cls(d_model, d_dict, k).to(device)

    batch_size = 16
    batch = activations[:batch_size].to(device)
    x_hat, z, loss = sae(batch)

    if x_hat.shape != (batch_size, d_model):
        print(f"FAIL (x_hat shape {tuple(x_hat.shape)}, expected ({batch_size}, {d_model}))")
        return
    if z.shape != (batch_size, d_dict):
        print(f"FAIL (z shape {tuple(z.shape)}, expected ({batch_size}, {d_dict}))")
        return

    # Total nonzeros should be k * batch_size
    total_nnz = (z > 0).sum().item()
    expected_nnz = k * batch_size
    if total_nnz != expected_nnz:
        print(f"FAIL (total nonzeros={total_nnz}, expected k*batch_size={expected_nnz})")
        return

    # Individual samples should have variable sparsity (not all exactly k)
    nnz_per_sample = (z > 0).sum(dim=-1)
    if (nnz_per_sample == k).all():
        print(f"FAIL (all samples have exactly {k} nonzeros, batch top-k should allow variable per-sample sparsity)")
        return

    print("PASS")


def test_jumprelu_sae(cls, activations, device):
    print("Testing JumpReLUSAE...", end=" ")
    d_model, d_dict = 512, 2048
    sae = cls(d_model, d_dict).to(device)

    # Check threshold is positive
    if (sae.threshold <= 0).any():
        print(f"FAIL (threshold should be positive, got min={sae.threshold.min().item():.4f})")
        return

    batch = activations[:16].to(device)
    x_hat, z, loss = sae(batch)

    if x_hat.shape != (16, d_model):
        print(f"FAIL (x_hat shape {tuple(x_hat.shape)}, expected (16, {d_model}))")
        return
    if z.shape != (16, d_dict):
        print(f"FAIL (z shape {tuple(z.shape)}, expected (16, {d_dict}))")
        return
    if loss.ndim != 0:
        print(f"FAIL (loss should be scalar)")
        return

    # Check JumpReLU property: nonzero z values should be > threshold
    threshold = sae.threshold.detach()
    for i in range(16):
        active = z[i] > 0
        if active.any():
            min_active = z[i][active].min().item()
            min_threshold = threshold[active].min().item()
            if min_active < min_threshold - 1e-5:
                print(f"FAIL (active feature value {min_active:.4f} < threshold {min_threshold:.4f})")
                return

    print("PASS")


def test_train_sae(train_fn, sae_cls, activations, device):
    print("Testing train_sae...", end=" ")
    sae = sae_cls(512, 2048).to(device)
    losses = train_fn(sae, activations.to(device), n_steps=200, batch_size=128, lr=3e-4, log_every=1000, l1_coeff=1.0)

    if not isinstance(losses, list):
        print(f"FAIL (expected list, got {type(losses).__name__})")
        return
    if len(losses) != 200:
        print(f"FAIL (expected 200 losses, got {len(losses)})")
        return
    if not all(isinstance(l, float) for l in losses):
        print(f"FAIL (losses should be floats)")
        return
    if any(not torch.isfinite(torch.tensor(l)) for l in losses):
        print(f"FAIL (non-finite loss values detected)")
        return

    # Check loss decreases (compare first 10 avg to last 10 avg)
    early = sum(losses[:10]) / 10
    late = sum(losses[-10:]) / 10
    if late >= early:
        print(f"FAIL (loss should decrease: early avg={early:.4f}, late avg={late:.4f})")
        return

    print("PASS")


def test_dead_neuron_tracker(cls, d_dict=2048):
    print("Testing DeadNeuronTracker...", end=" ")
    tracker = cls(d_dict, threshold=0.01, decay=0.99)

    # Simulate: only first 100 neurons fire
    z = torch.zeros(32, d_dict)
    z[:, :100] = torch.randn(32, 100).abs()
    for _ in range(100):
        tracker.update(z)

    dead = tracker.dead_mask
    if not isinstance(dead, torch.Tensor):
        print(f"FAIL (dead_mask should be a tensor, got {type(dead).__name__})")
        return
    if dead.shape != (d_dict,):
        print(f"FAIL (dead_mask shape {tuple(dead.shape)}, expected ({d_dict},))")
        return
    if dead.dtype != torch.bool:
        print(f"FAIL (dead_mask should be bool, got {dead.dtype})")
        return

    # First 100 should be alive (EMA converges up from 0), rest should be dead (stay at 0)
    if dead[:100].any():
        print(f"FAIL (neurons 0-99 fired every step but {dead[:100].sum().item()} marked dead)")
        return
    if not dead[100:].all():
        print(f"FAIL (neurons 100+ never fired but {(~dead[100:]).sum().item()} marked alive)")
        return

    print("PASS")


def score_sae(sae, activations, l0_budget=50, batch_size=1024):
    """Score an SAE: returns FVU if L0 <= budget, else penalized FVU."""
    sae.eval()
    with torch.no_grad():
        idx = torch.randint(0, len(activations), (batch_size,))
        batch = activations[idx].to(next(sae.parameters()).device)
        z = sae.encode(batch)
        x_hat = sae.decode(z)
        l0 = (z > 0).float().sum(dim=-1).mean().item()
        mse = (batch - x_hat).pow(2).mean().item()
        fvu = mse / batch.var().item()
        dead_frac = ((z > 0).any(dim=0) == False).float().mean().item()

    penalty = max(0, (l0 - l0_budget) / l0_budget)
    score = fvu * (1 + 10 * penalty)
    print(f"L0={l0:.1f}  FVU={fvu:.4f}  dead={dead_frac:.1%}  score={score:.4f}" +
          (f"  (PENALTY: L0 exceeds budget of {l0_budget})" if penalty > 0 else ""))
    return score
