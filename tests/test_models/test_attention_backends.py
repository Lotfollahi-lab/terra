"""SDPA backend presets: the logged description gives the order in which
PyTorch tries the enabled backends, and the presets agree with the eager
attention on CPU; without torch.nn.attention (PyTorch < 2.3) the backend is
left to PyTorch."""
import importlib
import importlib.util
import sys

import pytest
import torch

import terra.models.modules as modules


def test_description_lists_backends_in_pytorch_priority_order():
    from torch.nn.attention import SDPBackend
    order = [int(b) for b in torch._C._get_sdp_priority_order()]
    backends = modules.resolve_sdpa_backends('auto')
    listed = modules.describe_attention(True, 'auto').split(': ')[-1]
    names = listed.rstrip(')').split(' > ')
    assert sorted(names) == sorted(b.name for b in backends)
    ranks = [order.index(int(getattr(SDPBackend, name))) for name in names]
    assert ranks == sorted(ranks)


@pytest.mark.parametrize('preset', ['auto', 'math', 'flash', 'torch'])
@pytest.mark.parametrize('mask_kind', ['none', 'key_padding', 'full_bool',
                                       'additive'])
def test_presets_match_eager_attention(preset, mask_kind):
    torch.manual_seed(0)
    attention = modules.Attention(32, num_heads=4, qkv_bias=True,
                                  use_flash_attention=True,
                                  attention_backend=preset).eval()
    x = torch.randn(3, 10, 32)
    if mask_kind == 'none':
        masks = None
    elif mask_kind == 'key_padding':
        masks = torch.ones(3, 1, 1, 10, dtype=torch.bool)
        masks[:, :, :, 7:] = False
    elif mask_kind == 'full_bool':
        masks = torch.rand(3, 1, 10, 10) > 0.3
        masks[..., torch.arange(10), torch.arange(10)] = True
    else:
        masks = -torch.rand(3, 4, 10, 10) * 3
    with torch.no_grad():
        out, _ = attention(x, masks=masks)
        attention.use_flash_attention = False
        eager, _ = attention(x, masks=masks)
    torch.testing.assert_close(out, eager, rtol=1e-5, atol=1e-5)


def test_without_backend_selection(monkeypatch):
    # PyTorch < 2.3 has no torch.nn.attention: importing works and SDPA picks
    # its backend itself (instead of failing at import). A separate copy of
    # the module is loaded, so that the real one is left untouched.
    monkeypatch.setitem(sys.modules, 'torch.nn.attention', None)
    spec = importlib.util.spec_from_file_location(
        'terra.models._modules_without_backend_selection', modules.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.SDPBackend is None
    assert module.resolve_sdpa_backends('auto') is None
    attention = module.Attention(16, num_heads=2,
                                 attention_backend='auto').eval()
    with torch.no_grad():
        out, _ = attention(torch.randn(2, 5, 16))
    assert out.shape == (2, 5, 16)
