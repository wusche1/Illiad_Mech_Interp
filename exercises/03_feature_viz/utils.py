"""Test functions for the feature visualization exercise."""

import torch
import torch.nn as nn
from torchvision import models


def get_model():
    model = models.vgg19(weights='DEFAULT').features
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def get_activation(model, layer_index, img):
    for i, layer in enumerate(model):
        img = layer(img)
        if i == layer_index:
            return img


def test_feature_viz_step(step_fn):
    torch.manual_seed(42)
    model = get_model()
    img = 0.01 * torch.randn(1, 3, 64, 64, requires_grad=True)

    new_img = step_fn(img, model, layer_index=5, channel=10, lr=0.05)

    if new_img is None:
        print("FAIL: returned None -- not yet implemented?")
        return

    if not isinstance(new_img, torch.Tensor):
        print(f"FAIL: expected torch.Tensor, got {type(new_img)}")
        return

    if new_img.shape != (1, 3, 64, 64):
        print(f"FAIL: expected shape (1, 3, 64, 64), got {new_img.shape}")
        return

    if not new_img.requires_grad:
        print("FAIL: returned tensor must require gradients (for the next step)")
        return

    act_before = get_activation(model, 5, img)[0, 10].mean().item()
    act_after = get_activation(model, 5, new_img)[0, 10].mean().item()
    if act_after <= act_before:
        print(f"FAIL: activation should increase (before: {act_before:.4f}, after: {act_after:.4f})")
        return

    print(f"PASS: step works correctly (activation {act_before:.4f} -> {act_after:.4f})")


def test_feature_viz(viz_fn):
    torch.manual_seed(0)
    model = get_model()

    result = viz_fn(model, layer_index=5, channel=10, size=64, steps=50, lr=0.05)

    if result is None:
        print("FAIL: returned None -- not yet implemented?")
        return

    if not isinstance(result, torch.Tensor):
        print(f"FAIL: expected torch.Tensor, got {type(result)}")
        return

    if result.dim() != 3 or result.shape[0] != 3:
        print(f"FAIL: expected shape (3, H, W), got {result.shape}")
        return

    act = get_activation(model, 5, result.unsqueeze(0))[0, 10].mean().item()
    torch.manual_seed(0)
    baseline = get_activation(model, 5, 0.01 * torch.randn(1, 3, 64, 64))[0, 10].mean().item()

    if act > baseline * 5:
        print(f"PASS: optimized activation ({act:.2f}) >> random baseline ({baseline:.4f})")
    else:
        print(f"FAIL: optimized activation ({act:.2f}) not much higher than baseline ({baseline:.4f})")
