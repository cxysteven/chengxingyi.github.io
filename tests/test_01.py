"""Tests for problems/01_mha.md.

Black-box only: the interface and observable behaviour are checked, nothing about
how the module is implemented. Run from the repo root:

    python -m pytest tests/test_01.py -q
"""
import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
SOLUTION = ROOT / "solutions" / "01_mha.py"

B, T, D, H = 2, 7, 32, 4  # batch, seq_len, d_model, num_heads


def _load_class():
    if not SOLUTION.exists():
        pytest.skip(f"{SOLUTION.relative_to(ROOT)} not found yet")
    spec = importlib.util.spec_from_file_location("solution_01_mha", SOLUTION)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.MultiHeadAttention


@pytest.fixture(autouse=True)
def _no_grad():
    with torch.no_grad():
        yield


@pytest.fixture
def mha():
    torch.manual_seed(0)
    return _load_class()(D, H).eval()


@pytest.fixture
def x():
    torch.manual_seed(1)
    return torch.randn(B, T, D)


def _right_padding_mask():
    # sample 0: no padding. sample 1: last 3 positions are padding.
    # Every query row still has at least one attendable key, even with causal.
    m = torch.zeros(B, T, dtype=torch.bool)
    m[1, -3:] = True
    return m


def _left_padding_mask():
    # sample 0: first 2 positions are padding. sample 1: the whole sequence is padding.
    # Combined with causal this produces fully-masked query rows.
    m = torch.zeros(B, T, dtype=torch.bool)
    m[0, :2] = True
    m[1, :] = True
    return m


def _reference_from(mha):
    """torch.nn.MultiheadAttention carrying the same weights as the candidate module."""
    ref = nn.MultiheadAttention(D, H, bias=True, batch_first=True)
    ref.in_proj_weight.copy_(torch.cat([mha.q_proj.weight, mha.k_proj.weight, mha.v_proj.weight], 0))
    ref.in_proj_bias.copy_(torch.cat([mha.q_proj.bias, mha.k_proj.bias, mha.v_proj.bias], 0))
    ref.out_proj.weight.copy_(mha.out_proj.weight)
    ref.out_proj.bias.copy_(mha.out_proj.bias)
    return ref.eval()


# --------------------------------------------------------------------------- interface

def test_output_shapes(mha, x):
    out, attn = mha(x)
    assert out.shape == (B, T, D)
    assert attn.shape == (B, H, T, T)
    assert out.dtype == x.dtype


def test_projection_layers_follow_the_contract(mha):
    for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
        layer = getattr(mha, name)
        assert isinstance(layer, nn.Linear), name
        assert layer.weight.shape == (D, D), name
        assert layer.bias is not None, name


def test_rejects_d_model_not_divisible_by_num_heads():
    with pytest.raises((AssertionError, ValueError)):
        _load_class()(30, 4)


# --------------------------------------------------------------------------- distribution

@pytest.mark.parametrize("causal", [False, True])
def test_attention_rows_are_distributions(mha, x, causal):
    _, attn = mha(x, causal=causal)
    assert (attn >= 0).all()
    torch.testing.assert_close(attn.sum(-1), torch.ones(B, H, T), atol=1e-5, rtol=0)


def test_attention_rows_are_distributions_with_padding(mha, x):
    _, attn = mha(x, padding_mask=_right_padding_mask())
    assert (attn >= 0).all()
    torch.testing.assert_close(attn.sum(-1), torch.ones(B, H, T), atol=1e-5, rtol=0)


# --------------------------------------------------------------------------- causal

def test_causal_upper_triangle_is_zero(mha, x):
    _, attn = mha(x, causal=True)
    upper = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    assert (attn[..., upper] == 0).all()
    assert (attn[..., ~upper] > 0).all()  # lower triangle is actually used


# --------------------------------------------------------------------------- padding

def test_padding_keys_receive_zero_weight(mha, x):
    pm = _right_padding_mask()
    _, attn = mha(x, padding_mask=pm)
    masked_key_cols = pm[:, None, None, :].expand(B, H, T, T)
    assert (attn[masked_key_cols] == 0).all()
    # keys that are not padding must still be attended (mask must not leak onto the query axis)
    assert (attn[~masked_key_cols] > 0).all()


def test_padding_in_one_sample_does_not_affect_another(mha, x):
    pm = _right_padding_mask()  # sample 0 has no padding
    out_masked, attn_masked = mha(x, padding_mask=pm)
    out_plain, attn_plain = mha(x)
    torch.testing.assert_close(attn_masked[0], attn_plain[0])
    torch.testing.assert_close(out_masked[0], out_plain[0])


# --------------------------------------------------------------------------- fully-masked rows

def test_fully_masked_rows_causal_plus_left_padding_no_nan(mha, x):
    pm = _left_padding_mask()
    out, attn = mha(x, causal=True, padding_mask=pm)
    assert torch.isfinite(out).all()
    assert torch.isfinite(attn).all()

    allowed = (~pm)[:, None, None, :] & torch.tril(torch.ones(T, T, dtype=torch.bool))  # (B,1,T,T)
    allowed = allowed.expand(B, H, T, T)
    row_has_key = allowed.any(-1)  # (B,H,T)
    assert row_has_key.any() and (~row_has_key).any()  # the fixture really covers both kinds of rows

    # rows with at least one attendable key: still a distribution, and blocked keys get exactly 0
    torch.testing.assert_close(
        attn.sum(-1)[row_has_key], torch.ones(int(row_has_key.sum())), atol=1e-5, rtol=0
    )
    blocked_in_live_rows = ~allowed & row_has_key[..., None]
    assert (attn[blocked_in_live_rows] == 0).all()


def test_fully_padded_sequence_no_nan(mha, x):
    pm = torch.zeros(B, T, dtype=torch.bool)
    pm[1] = True  # sample 1 is entirely padding
    out, attn = mha(x, padding_mask=pm)
    assert torch.isfinite(out).all()
    assert torch.isfinite(attn).all()
    _, attn_plain = mha(x)
    torch.testing.assert_close(attn[0], attn_plain[0])


# --------------------------------------------------------------------------- numerics

def test_softmax_is_numerically_stable_on_large_inputs(mha):
    torch.manual_seed(2)
    big = torch.randn(B, T, D) * 1e4
    out, attn = mha(big)
    assert torch.isfinite(out).all()
    assert torch.isfinite(attn).all()
    torch.testing.assert_close(attn.sum(-1), torch.ones(B, H, T), atol=1e-5, rtol=0)


# --------------------------------------------------------------------------- vs torch

def test_matches_torch_without_mask(mha, x):
    ref = _reference_from(mha)
    out, attn = mha(x)
    ref_out, ref_attn = ref(x, x, x, need_weights=True, average_attn_weights=False)
    torch.testing.assert_close(out, ref_out, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(attn, ref_attn, atol=1e-5, rtol=1e-4)


def test_matches_torch_with_causal_and_right_padding(mha, x):
    # right padding + causal leaves no fully-masked row, so torch's reference is NaN-free
    ref = _reference_from(mha)
    pm = _right_padding_mask()
    blocked = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)  # torch: True = not allowed
    out, attn = mha(x, causal=True, padding_mask=pm)
    ref_out, ref_attn = ref(
        x, x, x, attn_mask=blocked, key_padding_mask=pm, need_weights=True, average_attn_weights=False
    )
    torch.testing.assert_close(out, ref_out, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(attn, ref_attn, atol=1e-5, rtol=1e-4)
