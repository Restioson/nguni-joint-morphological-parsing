from abc import ABC, abstractmethod

import torch
from torch import nn

from transformer_segmenter.vocabulary import Vocabulary


class SegmenterModel(ABC, nn.Module):
    def __init__(self, vocab: Vocabulary):
        super().__init__()
        self.vocab = vocab

    @abstractmethod
    def segment_words(self, words: list[str] | torch.Tensor, max_len=50) -> list[list[str]]:
        pass

    def repl(self):
        self.eval()

        with torch.no_grad():
            while True:
                src = input("> ")
                segmentations = self.segment_words(src.split(" "))
                print(" ".join("_".join(segmentation) for segmentation in segmentations))
