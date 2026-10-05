"""Tests for the vocoder discriminators, adversarial helpers and spectral losses (CPU, no data)."""

from pathlib import Path

import pytest
import torch
import torchaudio
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

import sparc
from sparc.vocoders.losses import discriminators as D
from sparc.vocoders.losses import losses as L
from sparc.vocoders.losses import ops

CROP = 30720  # 64 frames at 24 kHz
CONF_DIR = str(Path(sparc.__file__).parent / "conf")


def n_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


@pytest.fixture(scope="module")
def waves():
    g = torch.Generator().manual_seed(0)
    return torch.randn(1, 1, CROP, generator=g) * 0.1, torch.randn(1, 1, CROP, generator=g) * 0.1


@pytest.fixture(scope="module")
def mpd_out(waves):
    return D.MultiPeriodDiscriminator().eval()(*waves)


@pytest.fixture(scope="module")
def mrd_out(waves):
    return D.MultiResolutionDiscriminator().eval()(*waves)


@pytest.fixture(scope="module")
def msd_out(waves):
    return D.MultiScaleDiscriminator().eval()(*waves)


def test_parameter_counts():
    assert n_params(D.MultiPeriodDiscriminator()) == 41_105_770
    assert n_params(D.MultiResolutionDiscriminator()) == 1_413_990
    assert n_params(D.MultiScaleDiscriminator()) == 29_618_821


def test_mpd_shapes(mpd_out):
    real, fake, fmap_r, fmap_f = mpd_out
    assert len(real) == len(fake) == len(fmap_r) == len(fmap_f) == 5
    for period, r, f, fr, ff in zip((2, 3, 5, 7, 11), real, fake, fmap_r, fmap_f):
        assert r.dim() == 2 and r.shape == f.shape
        assert len(fr) == len(ff) == 5  # four convolutions after the skipped first one, plus conv_post
        assert fr[0].shape[-1] == period and fr[-1].shape[1] == 1
        assert all(a.shape == b.shape for a, b in zip(fr, ff))
    assert real[0].shape == (1, 380)  # ceil(30720 / 2 / 3 ** 4) = 190 rows of 2 columns


def test_mrd_shapes(mrd_out):
    real, fake, fmap_r, fmap_f = mrd_out
    assert [tuple(r.shape) for r in real] == [(1, 1, 61, 130), (1, 1, 121, 66), (1, 1, 241, 34)]
    assert [r.shape for r in real] == [f.shape for f in fake]
    assert [len(f) for f in fmap_r] == [21, 21, 21]  # 5 bands x 4 skipped-first convolutions + conv_post
    assert all(a.shape == b.shape for fr, ff in zip(fmap_r, fmap_f) for a, b in zip(fr, ff))


def test_msd_shapes_and_scale_lengths(msd_out):
    real, fake, fmap_r, _ = msd_out
    assert [tuple(r.shape) for r in real] == [(1, 480), (1, 241), (1, 121)]
    assert [len(f) for f in fmap_r] == [8, 8, 8]
    # scale 1, 1/2, 1/4: the first feature map has the length of the (pooled) input
    assert [f[0].shape[-1] for f in fmap_r] == [CROP, CROP // 2 + 1, CROP // 4 + 1]
    # the fork's cumulative strides (2, 4) gave a third scale of 1/8
    out = D.MultiScaleDiscriminator()(torch.randn(1, 1, 16320), torch.randn(1, 1, 16320))[0]
    assert [r.shape[-1] for r in out] == [255, 128, 64]


def test_msd_rejects_bad_scales():
    with pytest.raises(ValueError):
        D.MultiScaleDiscriminator(scales=(1, 3, 4))


def test_mpd_handles_length_not_divisible_by_period():
    real, fake, _, _ = D.MultiPeriodDiscriminator(periods=(3, 7))(torch.randn(2, 1, 1001), torch.randn(2, 1, 1001))
    assert all(r.shape[0] == 2 and torch.isfinite(r).all() for r in real + fake)


def test_mrd_is_blind_to_level():
    mrd = D.MultiResolutionDiscriminator(fft_sizes=(512,), channels=8).eval()
    y = torch.randn(1, 1, 8192) * 0.2
    base = mrd(y, y)[0][0]
    scaled = mrd(0.25 * y, 0.25 * y)[0][0]
    assert torch.allclose(base, scaled, atol=1e-4)


def test_mrd_band_count():
    mrd = D.MultiResolutionDiscriminator(fft_sizes=(512,), num_bands=4, channels=8)
    assert len(mrd.discriminators[0].band_convs) == 4
    assert mrd.discriminators[0].bands[0][0] == 0 and mrd.discriminators[0].bands[-1][1] == 257
    real, _, fmap_r, _ = mrd(torch.randn(1, 1, 8192), torch.randn(1, 1, 8192))
    assert len(fmap_r[0]) == 4 * 4 + 1


def test_mrd_under_bf16_autocast_returns_finite_logits():
    mrd = D.MultiResolutionDiscriminator(fft_sizes=(512,), channels=8)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        real, fake, _, _ = mrd(torch.randn(1, 1, 8192), torch.randn(1, 1, 8192))
    assert torch.isfinite(real[0].float()).all() and torch.isfinite(fake[0].float()).all()


def test_generator_gradient_reaches_waveform(waves):
    y, y_hat = waves[0], waves[1].clone().requires_grad_(True)
    mrd = D.MultiResolutionDiscriminator(fft_sizes=(512,), channels=8)
    _, fake, fmap_r, fmap_f = mrd(y, y_hat)
    (L.hinge_g_loss(fake)[0] + L.feature_matching_loss(fmap_r, fmap_f)).backward()
    assert y_hat.grad is not None and y_hat.grad.abs().sum() > 0


R = [torch.tensor([2.0, 0.0]), torch.tensor([0.0])]
F_ = [torch.tensor([-2.0, 0.5]), torch.tensor([0.0])]


def check(result, total, per_disc):
    got_total, got_per = result
    assert got_total.item() == pytest.approx(total)
    assert [p.item() for p in got_per] == pytest.approx(per_disc)


def test_hinge_values():
    check(L.hinge_d_loss(R, F_), 1.625, [1.25, 2.0])
    check(L.hinge_g_loss(F_), 1.375, [1.75, 1.0])


def test_lsgan_values():
    check(L.lsgan_d_loss(R, F_), 2.0625, [3.125, 1.0])
    check(L.lsgan_g_loss(F_), 2.8125, [4.625, 1.0])


def test_adversarial_registry():
    assert L.ADVERSARIAL_LOSSES["hinge"] == (L.hinge_d_loss, L.hinge_g_loss)
    assert L.ADVERSARIAL_LOSSES["lsgan"] == (L.lsgan_d_loss, L.lsgan_g_loss)


def test_feature_matching_value():
    real = [[torch.ones(2, 3), torch.zeros(4)], [torch.ones(5)]]
    fake = [[torch.zeros(2, 3), torch.full((4,), 0.5)], [torch.full((5,), 3.0)]]
    assert L.feature_matching_loss(real, fake).item() == pytest.approx((1.0 + 0.5 + 2.0) / 2)
    assert L.feature_matching_loss(real, real).item() == 0.0


def test_mel_loss_zero_for_identical_and_positive_otherwise(waves):
    mel = L.MelSpectrogramLoss()
    y, y_hat = waves
    assert mel(y, y).item() == 0.0
    assert mel(y_hat, y).item() > 0.0
    assert mel(y_hat[:, 0], y[:, 0]).item() == pytest.approx(mel(y_hat, y).item())


def test_mel_matches_torchaudio(waves):
    x = waves[0][:, 0]
    reference = torchaudio.transforms.MelSpectrogram(
        sample_rate=24000, n_fft=1024, win_length=1024, hop_length=256, n_mels=100, f_min=0.0, f_max=12000.0,
        power=1.0, center=True, mel_scale="htk",
    )
    expected = torch.log(reference(x).clamp(min=1e-5))
    got = L.MelSpectrogramLoss().log_mel(x)
    assert got.shape == (1, 100, CROP // 256 + 1)
    assert torch.allclose(got, expected, atol=1e-5)


def test_mel_buffers_not_in_state_dict_and_autocast_is_float32(waves):
    mel = L.MelSpectrogramLoss()
    assert len(mel.state_dict()) == 0
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert mel(*waves).dtype == torch.float32


def reference_mss(y_hat, y, fft_sizes, alpha=1.0, overlap=0.75):
    total = 0.0
    for n in fft_sizes:
        spec = torchaudio.transforms.Spectrogram(n_fft=n, win_length=n, hop_length=int(n * (1 - overlap)), power=1)
        a, b = spec(y_hat), spec(y)
        total = total + torch.nn.functional.l1_loss(a, b) + alpha * torch.nn.functional.l1_loss(
            torch.log(a + 1e-7), torch.log(b + 1e-7)
        )
    return total


def test_mss_matches_ddsp_av_form(waves):
    y, y_hat = waves[0][:, 0], waves[1][:, 0]
    mss = L.MultiScaleSpectralLoss()
    assert mss(y, y).item() == 0.0
    assert mss(y_hat, y).item() == pytest.approx(reference_mss(y_hat, y, mss.fft_sizes).item(), rel=1e-5)
    two = L.MultiScaleSpectralLoss(fft_sizes=(512, 64), alpha=0.5)
    assert two(y_hat, y).item() == pytest.approx(reference_mss(y_hat, y, (512, 64), alpha=0.5).item(), rel=1e-5)


def test_mss_gradient_is_finite_for_digital_silence(waves):
    y_hat = torch.zeros(1, 1, 8192, requires_grad=True)
    L.MultiScaleSpectralLoss()(y_hat, waves[0][..., :8192]).backward()
    assert torch.isfinite(y_hat.grad).all()


def test_mr_stft_distance(waves):
    y, y_hat = waves
    assert L.mr_stft_distance(y, y).item() == pytest.approx(0.0, abs=1e-6)
    near = L.mr_stft_distance(y + 0.01 * torch.randn_like(y), y).item()
    far = L.mr_stft_distance(y_hat, y).item()
    assert 0.0 < near < far
    assert not L.mr_stft_distance(y_hat, y).requires_grad


@pytest.fixture(scope="module")
def loss_cfgs():
    with initialize_config_dir(config_dir=CONF_DIR, version_base=None):
        return {n: compose(config_name=None, overrides=[f"+loss={n}"]).loss for n in
                ("shared", "mss_aux", "msd", "native_hifigan")}


def test_loss_config_schema(loss_cfgs):
    for name, cfg in loss_cfgs.items():
        assert cfg.name == name
        assert cfg.adversarial in L.ADVERSARIAL_LOSSES
        assert {"mel_weight", "mss_weight", "mel", "mss", "discriminators"} <= set(cfg)
        assert cfg.mel._target_.endswith("losses.MelSpectrogramLoss")
        assert cfg.mss._target_.endswith("losses.MultiScaleSpectralLoss")
        for spec in cfg.discriminators.values():
            assert {"module", "adv_weight", "fm_weight"} == set(spec)


def test_loss_config_values(loss_cfgs):
    shared, aux, msd, native = (loss_cfgs[n] for n in ("shared", "mss_aux", "msd", "native_hifigan"))
    assert shared.adversarial == "hinge" and shared.mel_weight == 45.0 and shared.mss_weight == 0.0
    assert list(shared.discriminators) == ["mpd", "mrd"]
    assert (shared.discriminators.mpd.adv_weight, shared.discriminators.mpd.fm_weight) == (1.0, 1.0)
    assert (shared.discriminators.mrd.adv_weight, shared.discriminators.mrd.fm_weight) == (0.1, 0.1)
    assert aux.mss_weight == 1.0 and aux.discriminators == shared.discriminators and aux.mel == shared.mel
    assert msd.adversarial == "hinge" and list(msd.discriminators) == ["mpd", "msd"]
    assert native.adversarial == "lsgan" and list(native.discriminators) == ["mpd", "msd"]
    # the fork sums over sub-discriminators: adversarial 1 and feature matching 2 per sub-discriminator
    assert native.discriminators.mpd.adv_weight == 5 * 1.0 and native.discriminators.mpd.fm_weight == 5 * 2.0
    assert native.discriminators.msd.adv_weight == 3 * 1.0 and native.discriminators.msd.fm_weight == 3 * 2.0
    assert native.mel_weight == 45.0 and native.mel == shared.mel


def test_loss_configs_instantiate(loss_cfgs):
    for cfg in loss_cfgs.values():
        assert isinstance(instantiate(cfg.mel), L.MelSpectrogramLoss)
        assert isinstance(instantiate(cfg.mss), L.MultiScaleSpectralLoss)
    shared = loss_cfgs["shared"]
    assert n_params(instantiate(shared.discriminators.mrd.module)) == 1_413_990
    assert isinstance(instantiate(loss_cfgs["msd"].discriminators.msd.module), D.MultiScaleDiscriminator)


@pytest.mark.parametrize("left,right", [(0, 5), (7, 0), (512, 512), (3, 9), (0, 0)])
def test_reflect_pad_matches_functional_value_and_gradient(left, right):
    x = torch.randn(2, 1, 600, dtype=torch.float64, requires_grad=True)
    expected = torch.nn.functional.pad(x, (left, right), "reflect")
    got = ops.reflect_pad_1d(x, left, right)
    assert torch.equal(got, expected)
    weight = torch.randn_like(got)
    (g1,) = torch.autograd.grad((got * weight).sum(), x)
    (g2,) = torch.autograd.grad((expected * weight).sum(), x)
    assert torch.allclose(g1, g2)


def test_reflect_pad_rejects_padding_not_shorter_than_signal():
    with pytest.raises(ValueError):
        ops.reflect_pad_1d(torch.randn(1, 8), 0, 8)


@pytest.mark.parametrize("n_fft,win_length", [(1024, 1024), (1024, 512), (64, 64)])
def test_stft_matches_torch_centred_stft(n_fft, win_length):
    x = torch.randn(2, 5000)
    window = torch.hann_window(win_length)
    expected = torch.stft(x, n_fft, 256 if n_fft > 64 else 16, win_length=win_length, window=window, center=True,
                          pad_mode="reflect", return_complex=True)
    got = ops.stft(x, n_fft, 256 if n_fft > 64 else 16, win_length, window)
    assert torch.allclose(got, expected, atol=1e-5)


def test_losses_and_discriminators_avoid_nondeterministic_reflect_pad(monkeypatch, waves):
    """Reflect padding and ``torch.stft(center=True)`` have no deterministic CUDA backward (``trainer.deterministic``)."""
    real_pad, real_stft = torch.nn.functional.pad, torch.stft

    def pad(x, pad_, mode="constant", value=None):
        assert mode != "reflect"
        return real_pad(x, pad_, mode, value)

    def stft(*args, **kwargs):
        assert kwargs.get("center") is False
        return real_stft(*args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "pad", pad)
    monkeypatch.setattr(torch, "stft", stft)
    y, y_hat = waves[0][..., :8001], waves[1][..., :8001]
    D.MultiPeriodDiscriminator(periods=(7, 11))(y, y_hat)
    D.MultiResolutionDiscriminator(fft_sizes=(512,), channels=8)(y, y_hat)
    L.MelSpectrogramLoss()(y_hat, y)
    L.MultiScaleSpectralLoss()(y_hat, y)
    L.mr_stft_distance(y_hat, y)


def test_mismatched_waveform_shapes_are_rejected(waves):
    y, y_hat = waves
    short = y_hat[..., :-100]  # same number of mel frames is possible; the shapes must still match
    for loss in (L.MelSpectrogramLoss(), L.MultiScaleSpectralLoss(), L.mr_stft_distance):
        with pytest.raises(ValueError):
            loss(short, y)
    for disc in (D.MultiPeriodDiscriminator(periods=(2,)), D.MultiResolutionDiscriminator(fft_sizes=(512,), channels=8),
                 D.MultiScaleDiscriminator()):
        with pytest.raises(ValueError):
            disc(y, short)
    L.MelSpectrogramLoss()(y_hat[:, 0], y)  # [B, L] against [B, 1, L] is the same waveform


def test_adversarial_and_feature_matching_losses_reduce_in_float32():
    logits = [torch.tensor([0.3, -0.2], dtype=torch.bfloat16, requires_grad=True)]
    fmaps = [[torch.full((3,), 0.5, dtype=torch.bfloat16)]]
    for fn in (L.hinge_d_loss, L.lsgan_d_loss):
        total, per_disc = fn(logits, logits)
        assert total.dtype == per_disc[0].dtype == torch.float32
    assert L.hinge_g_loss(logits)[0].dtype == L.lsgan_g_loss(logits)[0].dtype == torch.float32
    assert L.feature_matching_loss(fmaps, [[torch.zeros(3, dtype=torch.bfloat16)]]).dtype == torch.float32
    L.hinge_g_loss(logits)[0].backward()
    assert logits[0].grad.dtype == torch.bfloat16


def test_inputs_are_not_modified_and_outputs_do_not_depend_on_batch_mates(waves):
    y = torch.cat([waves[0], waves[1]])
    y_hat = torch.cat([waves[1], waves[0]]) * 0.5
    before = y.clone(), y_hat.clone()
    modules = [D.MultiPeriodDiscriminator(periods=(2, 7)).eval(),
               D.MultiResolutionDiscriminator(fft_sizes=(512,), channels=8).eval(), D.MultiScaleDiscriminator().eval()]
    for module in modules:
        batched = module(y, y_hat)
        single = module(y[:1], y_hat[:1])
        for a, b in zip(batched[0] + batched[1], single[0] + single[1]):
            assert torch.allclose(a[:1], b, atol=1e-4)
    for loss in (L.MelSpectrogramLoss(), L.MultiScaleSpectralLoss()):
        loss(y_hat, y)
    L.mr_stft_distance(y_hat, y)
    assert torch.equal(y, before[0]) and torch.equal(y_hat, before[1])
