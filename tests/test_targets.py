"""Target reparameterizations of the multi-speaker EMA datasets (sparc.compression.datasets.target_matrices)."""

import numpy as np
import pytest

from sparc.compression.datasets import _sorted_eig, split_target, target_matrices


def rot(deg):
    t = np.radians(deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def block_cov(blocks, rng):
    C = np.zeros((12, 12))
    for a, B in enumerate(blocks):
        C[2 * a:2 * a + 2, 2 * a:2 * a + 2] = B
    return C


def random_blocks(rng):
    out = []
    for _ in range(6):
        L = np.diag(rng.uniform(1, 9, 2))
        R = rot(rng.uniform(-90, 90))
        out.append(R @ L @ R.T)
    return out


@pytest.mark.parametrize("kind", ["z", "zca", "pca", "zca12"])
def test_inverse_and_whitening(kind):
    rng = np.random.default_rng(0)
    M = rng.normal(size=(12, 12))
    C = M @ M.T + 12 * np.eye(12) if kind == "zca12" else block_cov(random_blocks(rng), rng)
    std = np.sqrt(np.diag(C))
    ref = [_sorted_eig(C[2 * a:2 * a + 2, 2 * a:2 * a + 2])[1] for a in range(6)]
    A, Ainv = target_matrices(kind, C, std, ref)
    assert np.allclose(A @ Ainv, np.eye(12))
    Cz = C / np.outer(std, std)  # covariance of the z-scored frame
    Cy = A @ Cz @ A.T
    if kind == "z":
        assert np.allclose(Cy, Cz)
    elif kind == "zca12":
        assert np.allclose(Cy, np.eye(12), atol=1e-8)
    else:
        for a in range(6):
            i = slice(2 * a, 2 * a + 2)
            assert np.allclose(Cy[i, i], np.eye(2), atol=1e-8)


def test_zca_keeps_orientation():
    """zca applied to mm deviations is symmetric positive definite: no rotation of the frame."""
    rng = np.random.default_rng(1)
    C = block_cov(random_blocks(rng), rng)
    std = np.sqrt(np.diag(C))
    A, _ = target_matrices("zca", C, std)
    W = A @ np.diag(1 / std)
    assert np.allclose(W, W.T)
    assert np.all(np.linalg.eigvalsh(W) > 0)


@pytest.mark.parametrize("deg", [20.0, -35.0])
def test_pca_undoes_a_group_rotation(deg):
    """Two groups whose articulator distributions differ by a rotation map to the same target distribution
    and their transforms differ exactly by that rotation."""
    rng = np.random.default_rng(2)
    B1 = random_blocks(rng)
    B1 = [rot(10) @ np.diag([9.0, 1.0]) @ rot(10).T for _ in B1]  # clearly anisotropic
    R = rot(deg)
    B2 = [R @ B @ R.T for B in B1]
    C1, C2 = block_cov(B1, rng), block_cov(B2, rng)
    ref = [_sorted_eig(B)[1] for B in B1]
    s1, s2 = np.sqrt(np.diag(C1)), np.sqrt(np.diag(C2))
    A1, _ = target_matrices("pca", C1, s1, ref)
    A2, _ = target_matrices("pca", C2, s2, ref)
    W1, W2 = A1 @ np.diag(1 / s1), A2 @ np.diag(1 / s2)  # maps from mm deviations
    for a in range(6):
        i = slice(2 * a, 2 * a + 2)
        assert np.allclose(W2[i, i], W1[i, i] @ R.T, atol=1e-8)


def test_pca_axes_matched_by_direction_not_variance():
    """Axes are paired with the reference by direction, so the per-group rotation never exceeds 45 degrees
    (a near-isotropic articulator is not spun by an arbitrary angle when its variance order flips)."""
    ref = [np.eye(2)] * 6
    B = rot(80) @ np.diag([4.0, 1.0]) @ rot(80).T  # major axis at 80 deg: close to the reference's 2nd axis
    C = block_cov([B] * 6, None)
    s = np.sqrt(np.diag(C))
    A, _ = target_matrices("pca", C, s, ref)
    W = A[:2, :2] @ np.diag(1 / s[:2])
    # rotation part of W (polar decomposition) is within 45 degrees of identity
    U, _, Vt = np.linalg.svd(W)
    Q = U @ Vt
    assert np.degrees(np.arccos(np.clip(Q[0, 0], -1, 1))) <= 45 + 1e-6


def test_split_target():
    assert split_target("ema_loso_usc_M1_pca") == ("ema_loso_usc_M1", "pca")
    assert split_target("ema_loso_5emo_jn_zca12") == ("ema_loso_5emo_jn", "zca12")
    assert split_target("ema_loso_5emo_jn_zca") == ("ema_loso_5emo_jn", "zca")
    assert split_target("ema_multi") == ("ema_multi", "z")
