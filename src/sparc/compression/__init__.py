"""Structural compression of SSL speech encoders for speech-to-EMA prediction on MNGU0.

Modules:
    mngu0     -- standard split, label-derived EMA/audio alignment, EMA in mm
    metrics   -- RMSE/MAE/PCC/velocity error, per articulator, bootstrap CIs, per phone class
    encoders  -- SSL model registry and layer-subset (truncated / non-contiguous) encoders
    compute   -- parameters, FLOPs/MACs, activation memory and latency of an encoder
    extract   -- cache every layer's hidden states for an SSL model over MNGU0
    probe     -- per-layer ridge probes on the cached features
"""
