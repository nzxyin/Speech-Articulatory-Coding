"""Shared constants for the 24 kHz articulatory vocoders (see docs/vocoders/INTERFACES.md)."""

SAMPLE_RATE = 24000
FRAME_RATE = 50
HOP = SAMPLE_RATE // FRAME_RATE  # 480 samples per feature frame
EXTRACTOR_SAMPLE_RATE = 16000
EXTRACTOR_HOP = EXTRACTOR_SAMPLE_RATE // FRAME_RATE  # 320

N_EMA = 12
EMA_NAMES = ("TDX", "TDY", "TBX", "TBY", "TTX", "TTY", "LIX", "LIY", "ULX", "ULY", "LLX", "LLY")
N_FEATURES = 15
F0_CHANNEL = 12
LOUDNESS_CHANNEL = 13
PERIODICITY_CHANNEL = 14

SPEAKER_RAW_DIM = 1024
SPEAKER_DIM = 64

LOUDNESS_EPS = 1e-4
F0_MIN_HZ = 50.0
F0_MAX_HZ = 550.0

# Feature-frame centres inside SPARC, in 24 kHz samples relative to 480 * t.
LOUDNESS_CENTRE_OFFSET = 0
F0_CENTRE_OFFSET = 120
WAVLM_CENTRE_OFFSET = 300


def feature_length(n24: int) -> int:
    """Number of SPARC feature frames for an utterance of ``n24`` samples at 24 kHz.

    SPARC resamples to 16 kHz (``ceil(2 * n24 / 3)`` samples) and truncates every stream to the WavLM length
    ``floor((L16 - 80) / 320)``.
    """
    l16 = -((-2 * n24) // 3)
    return (l16 - 80) // EXTRACTOR_HOP
