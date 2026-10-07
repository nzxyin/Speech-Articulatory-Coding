"""Reader for mview-format EMA .mat files (Haskins mview; used by USC-TIMIT EMA and USC EMA_5EMO).

Each file holds one struct array named after the file. Element 0 is AUDIO (SIGNAL: samples, SRATE: Hz); the
others are sensors (NAME, SRATE ~100 Hz, SIGNAL: (T, 3) positions in mm, NaN during dropouts). Sensor order
varies between files, so sensors are always addressed by NAME. Audio and EMA start together: every sample
index i is at time i / SRATE from the start of the file.
"""

from dataclasses import dataclass

import numpy as np
import scipy.io as sio

SENSORS = ("TT", "TB", "TD", "UL", "LL", "JAW")


@dataclass
class MviewRecord:
    audio: np.ndarray  # (n,) float64
    audio_sr: float
    ema: dict  # sensor name -> (T, 3) float64, mm
    ema_sr: float

    @property
    def audio_dur(self):
        return len(self.audio) / self.audio_sr

    @property
    def ema_dur(self):
        return len(next(iter(self.ema.values()))) / self.ema_sr


def read_mview(path, sensors=SENSORS):
    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    names = [k for k in m if not k.startswith("__")]
    if len(names) != 1:
        raise ValueError(f"{path}: expected one struct variable, got {names}")
    elems = {str(e.NAME): e for e in np.atleast_1d(m[names[0]])}
    if "AUDIO" not in elems:
        raise ValueError(f"{path}: no AUDIO element")
    rates = {float(elems[s].SRATE) for s in sensors}
    if len(rates) != 1:
        raise ValueError(f"{path}: sensors have different sampling rates {rates}")
    ema = {}
    for s in sensors:
        sig = np.asarray(elems[s].SIGNAL, dtype=np.float64)
        if sig.ndim != 2 or sig.shape[1] != 3:
            raise ValueError(f"{path}: sensor {s} has shape {sig.shape}, expected (T, 3)")
        ema[s] = sig
    lengths = {len(v) for v in ema.values()}
    if len(lengths) != 1:
        raise ValueError(f"{path}: sensors have different lengths {lengths}")
    return MviewRecord(audio=np.asarray(elems["AUDIO"].SIGNAL, dtype=np.float64).ravel(),
                       audio_sr=float(elems["AUDIO"].SRATE), ema=ema, ema_sr=rates.pop())
