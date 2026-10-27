"""Capacity-matched controls of the speaker-encoder screen (precondition checks).

The baseline pooled layer 6 vector (periodicity-weighted mean, 1024-d) is repeated ``k`` times into a head with the
input width and parameter count of a shortlisted variant: ``cap_rep3`` matches the three-way voiced / unvoiced /
silence split (3072-d input, 3.2M parameters), ``cap_rep2`` matches layer 1 plus layer 6 and the two-way split
(2048-d input, 2.2M parameters). Repetition keeps the function class of the baseline (the first layer can only use the
sum of the copies' weights) while giving the head the same number of weights, the same initialisation scale and the
same weight decay as the variant; it adds no information. Zero padding was rejected because the extra weights would
never receive a gradient and the control would be the baseline again.
"""

from sparc.vocoders.spkscreen.pooling import register
from sparc.vocoders.spkscreen.variants.segment import _SegmentBase


class RepeatedBaseline(_SegmentBase):
    def __init__(self, copies: int):
        super().__init__()
        self.copies = copies
        self._init_head(1024 * copies)

    def features(self, b):
        return self.base_vec(b).repeat(1, self.copies)


register("cap_rep3")(lambda: RepeatedBaseline(3))
register("cap_rep2")(lambda: RepeatedBaseline(2))
