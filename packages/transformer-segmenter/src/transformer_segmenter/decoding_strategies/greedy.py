import torch

from . import SegmenterModel
from ..vocabulary import Vocabulary


class TransformerSegmenterGreedy(SegmenterModel):
    def __init__(self, vocab: Vocabulary):
        super().__init__(vocab)

    def segment_words(self, words: list[str] | torch.Tensor, max_len=50) -> list[list[str]]:
        pass
