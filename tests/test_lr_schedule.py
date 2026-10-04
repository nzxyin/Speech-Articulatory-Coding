from sparc.training.lightning_module import lr_decay_warning, lr_lambda


def test_default_multipliers():
    assert lr_lambda(0) == 1.0
    assert lr_lambda(199999) == 1.0
    assert lr_lambda(200000) == 0.5
    assert lr_lambda(1_500_000) == 0.5 ** 7


def test_static_after():
    assert lr_lambda(10**7, halve_every=8000, static_after=16000) == 0.25
    assert lr_lambda(10**7, halve_every=8000, static_after=None) == 0.5 ** 1250


def test_warning():
    assert lr_decay_warning(5000, 200000) is None
    assert lr_decay_warning(1_500_000, 200000) is None
    assert lr_decay_warning(1_500_000, 8000, 320000) is not None  # old defaults: 0.5**40
    assert lr_decay_warning(None, 8000) is None


if __name__ == "__main__":
    test_default_multipliers(); test_static_after(); test_warning(); print("ok")
