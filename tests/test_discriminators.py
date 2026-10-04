import torch

from sparc.training.discriminators import MultiScaleDiscriminator


def test_msd_scales_are_cumulative_1_2_4():
    msd = MultiScaleDiscriminator().eval()
    y = torch.randn(1, 1, 16320)
    with torch.no_grad():
        y_d_rs, y_d_gs, _, _ = msd(y, y)
    assert [o.shape[-1] for o in y_d_rs] == [255, 128, 64]
    assert [o.shape[-1] for o in y_d_gs] == [255, 128, 64]


def test_msd_loads_existing_state_dict():
    msd = MultiScaleDiscriminator()
    # AvgPool has no parameters, so the state dict has no meanpool keys.
    sd = msd.state_dict()
    assert not any(k.startswith("meanpools") for k in sd)
    MultiScaleDiscriminator().load_state_dict(sd, strict=True)


if __name__ == "__main__":
    test_msd_scales_are_cumulative_1_2_4()
    test_msd_loads_existing_state_dict()
    print("ok")
