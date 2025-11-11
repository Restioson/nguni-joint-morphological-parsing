from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sized, Self, Optional

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from annotated_corpus_dataset import SegmentationDataset, InputTokenizerBuilder, OutputTokenizerBuilder, \
    JointEndToEndParsingDataset, InputTokenizer, ParsingPipelineDataset
from transformer_segmenter.vocabulary import Vocabulary


class TransformerDatasetItem:
    def __init__(self, source: torch.Tensor, target: torch.Tensor):
        self.source = source
        self.target = target

    def to(self, device: torch.device) -> Self:
        return TransformerDatasetItem(self.source.to(device), self.target.to(device))


class TransformerDatasetBatch:
    def __init__(self, source: torch.Tensor, target: torch.Tensor):
        self.source = source
        self.target = target

    def to(self, device: torch.device) -> Self:
        return TransformerDatasetBatch(self.source.to(device), self.target.to(device))


class TransformerDataset(Dataset, Sized, ABC):
    @property
    @abstractmethod
    def vocabulary(self) -> Vocabulary:
        pass

    def __getitem__(self, index) -> TransformerDatasetItem:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    @abstractmethod
    def collate(self, batch):
        pass

    @abstractmethod
    def to(self, device: torch.device) -> Self:
        pass


class TransformerSegmentationDataset(TransformerDataset):
    def __init__(self, base: SegmentationDataset):
        self.base = base
        self.transformed_seqs = [TransformerDatasetItem(item.raw_word, item.morphemes) for item in base.seqs]
        self._vocabulary = Vocabulary(self.base)

    @staticmethod
    def load_data(lang: str, data_dir: Path, input_tokenizer_builder: InputTokenizerBuilder, output_tokenizer_builder: OutputTokenizerBuilder, split, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        portions = SegmentationDataset.load_data(lang, data_dir, input_tokenizer_builder, output_tokenizer_builder, split, device)
        return { name : TransformerSegmentationDataset(dset) for name, dset in portions.items() }

    @property
    def vocabulary(self) -> Vocabulary:
        return self._vocabulary

    def __len__(self):
        return len(self.transformed_seqs)

    def __getitem__(self, index):
        return self.base.seqs[index]

    def collate(self, batch):
        dev = batch[0].raw_word.device
        input_start = torch.tensor([self.vocabulary.input_start_token_ix], device=dev)
        input_end = torch.tensor([self.vocabulary.input_end_token_ix], device=dev)
        output_start = torch.tensor([self.vocabulary.output_start_token_ix], device=dev)
        output_end = torch.tensor([self.vocabulary.output_end_token_ix], device=dev)

        raw_words = pad_sequence([torch.cat((input_start, item.raw_word, input_end)) for item in batch], batch_first=True, padding_value=self.vocabulary.output_pad_ix)
        morphemes = pad_sequence([torch.cat((output_start, item.morphemes, output_end)) for item in batch], batch_first=True, padding_value=self.vocabulary.output_pad_ix)
        return TransformerDatasetBatch(raw_words, morphemes)

    def to(self, device: torch.device) -> Self:
        return TransformerSegmentationDataset(self.base.to(device))


class TransformerJointEndToEndParsingDataset(TransformerDataset):
    def __init__(self, base: JointEndToEndParsingDataset):
        self.base = base
        self.transformed_seqs = [TransformerDatasetItem(item.raw_word, item.morphemes_and_tags) for item in base.seqs]
        self._vocabulary = Vocabulary(self.base)

    @staticmethod
    def load_data(lang: str, data_dir: Path, input_tokenizer_builder: InputTokenizerBuilder, output_tokenizer_builder: OutputTokenizerBuilder, split, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        portions = JointEndToEndParsingDataset.load_data(lang, data_dir, input_tokenizer_builder, output_tokenizer_builder, split, device)
        return {name : TransformerJointEndToEndParsingDataset(dset) for name, dset in portions.items()}

    @property
    def vocabulary(self) -> Vocabulary:
        return self._vocabulary

    def __len__(self):
        return len(self.transformed_seqs)

    def __getitem__(self, index):
        return self.base.seqs[index]

    def collate(self, batch):
        dev = batch[0].raw_word.device
        input_start = torch.tensor([self.vocabulary.input_start_token_ix], device=dev)
        input_end = torch.tensor([self.vocabulary.input_end_token_ix], device=dev)
        output_start = torch.tensor([self.vocabulary.output_start_token_ix], device=dev)
        output_end = torch.tensor([self.vocabulary.output_end_token_ix], device=dev)

        raw_words = pad_sequence([torch.cat((input_start, item.raw_word, input_end)) for item in batch], batch_first=True, padding_value=self.vocabulary.output_pad_ix)
        morphemes_and_tags = pad_sequence([torch.cat((output_start, item.morphemes_and_tags, output_end)) for item in batch], batch_first=True, padding_value=self.vocabulary.output_pad_ix)
        return TransformerDatasetBatch(raw_words, morphemes_and_tags)

    def to(self, device: torch.device) -> Self:
        return TransformerJointEndToEndParsingDataset(self.base.to(device))

