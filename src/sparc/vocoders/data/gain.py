"""Gain augmentation: whole-file peak normalization to a target level, equivalent to ``sox norm <db>``."""

import numpy as np

DEFAULT_GAIN_DB_RANGE = (-6.0, -1.0)


def sample_gain_db(rng: np.random.Generator, db_range: tuple[float, float] = DEFAULT_GAIN_DB_RANGE) -> float:
    """Draws a target peak level in dBFS, uniform over ``db_range`` and rounded to two decimals."""
    low, high = db_range
    return round(float(rng.uniform(low, high)), 2)


def gain_factor(peak24: float, db: float) -> float:
    """Linear factor that moves a file whose absolute peak is ``peak24`` to a peak of ``db`` dBFS.

    ``peak24`` is the maximum absolute sample of the whole 24 kHz file, so a crop is scaled by the same factor as
    the full utterance would be.
    """
    if not peak24 > 0.0:
        raise ValueError(f"cannot normalize a file with peak {peak24}")
    return 10.0 ** (db / 20.0) / float(peak24)


def apply_gain(x: np.ndarray, g: float) -> np.ndarray:
    """Returns ``x * g`` as float32, multiplied in float64."""
    return (np.asarray(x, dtype=np.float64) * g).astype(np.float32)
