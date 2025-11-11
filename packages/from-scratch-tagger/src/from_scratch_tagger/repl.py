from pathlib import Path

import torch

from annotated_corpus_dataset import TaggingDataset
from .common import predict_for_test_set
from .encapsulated_model import load_model, annotate_sentence

if __name__ == "__main__":
    model = load_model("bilstm-words-morpheme-ZU-final.pt")

    portions = TaggingDataset.load_data("ZU", Path("data/processed"), model.split, model.tokenize)
    valid = portions["valid"]

    print(predict_for_test_set(model, valid))

    with torch.no_grad():
        while True:
            morphemes = input("Morphemes separated by _ > ").split("_")
            print(annotate_sentence(model, morphemes))
