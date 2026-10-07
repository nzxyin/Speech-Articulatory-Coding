"""Preprocessing of multi-speaker EMA corpora (USC-TIMIT EMA, USC EMA_5EMO) into audio-aligned utterances.

Modules:
    mview      -- reader for the mview .mat format both corpora use
    corpora    -- file discovery, utterance metadata, sentence segmentation, per-speaker frame corrections
    preprocess -- alignment, dropout handling, resampling to 50 Hz, outputs and normalization statistics
    dataset    -- loader for the preprocessed outputs
"""
