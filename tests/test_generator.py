import torch

from sparc.block import SoftClamp
from sparc.generator import HiFiGANGenerator

CFG = dict(in_channels=15, channels=32, upsample_scales=(5, 4, 3, 2),
           upsample_kernel_sizes=(10, 8, 6, 4), paddings=["default"] * 4,
           output_paddings=["default"] * 4, spk_emb_size=8)


def _gen(**kw):
    return HiFiGANGenerator(**{**CFG, **kw}).eval()


def test_forward_does_not_mutate_input():
    g = _gen()
    c = torch.randn(2, 15, 10)
    c[:, 12] = 150.0
    c0 = c.clone()
    spk = torch.randn(2, 8)
    out1 = g(c, spk)
    assert torch.equal(c, c0)
    assert torch.equal(g(c, spk), out1)
    assert out1.shape[-1] == 10 * 120


def test_init_applies_to_weight_norm_params():
    torch.manual_seed(0)
    g = _gen(use_weight_norm=True)
    m = g.input_conv
    assert hasattr(m, "weight_v")
    # N(0, 0.01) direction vectors, not PyTorch's default kaiming init
    assert m.weight_v.std().item() < 0.02


def test_state_dict_load_gives_identical_outputs():
    a, b = _gen(), _gen()
    b.load_state_dict(a.state_dict())
    c, spk = torch.randn(1, 15, 8), torch.randn(1, 8)
    assert torch.equal(a(c, spk), b(c, spk))


def test_softclamp_temp():
    x = torch.linspace(-20, 20, 11)
    assert torch.allclose(SoftClamp()(x), torch.tanh(x * 0.2) / 0.2)
    assert torch.allclose(SoftClamp(temp=1.0)(x), torch.tanh(x))


def test_unsupported_paddings_raise():
    for kw in (dict(paddings=[1, 1, 1, 1]), dict(output_paddings=[0, 0, 0, 0])):
        try:
            _gen(**kw)
        except NotImplementedError:
            continue
        raise AssertionError("expected NotImplementedError")


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f()
            print("ok", n)
    print("SKIP pretrained-checkpoint comparison: no en+ checkpoint in local cache")
