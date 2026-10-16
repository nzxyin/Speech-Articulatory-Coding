import numpy as np

from sparc.compression import nested


def test_argsel_ties_go_shallow():
    assert nested.argsel({3: 1.0, 5: 1.0, 7: 2.0}, "min") == 3
    assert nested.argsel({3: 1.0, 5: 2.0, 7: 2.0}, "max") == 5


def test_most_frequent_lr():
    assert nested.most_frequent_lr([1e-3, 3e-4, 1e-3]) == 1e-3
    assert nested.most_frequent_lr([1e-3, 3e-4]) == 3e-4


def test_paired_stats_sign():
    s = nested.paired_stats([0.7, 0.8, 0.75, 0.9], [0.6, 0.7, 0.7, 0.85], 1)
    assert s["won"] == 4 and s["mean"] > 0
    s = nested.paired_stats([0.7, 0.8, 0.75, 0.9], [0.6, 0.7, 0.7, 0.85], -1)
    assert s["won"] == 0
