import re
import typing
from pathlib import Path
from typing import Generator, Self

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .segmentation_tokenizers import InputTokenizerBuilder, OutputTokenizerBuilder, InputTokenizer, OutputTokenizer, \
    MORPHEME_SEP_TEXT, WORD_SEP_TEXT

# Constants for some foundational dataset elements (padding, word separators, unknown tokens)
SEQ_PAD_TEXT = "<?pad?>"
SEQ_PAD_IX = 0

UNK_TEXT = "<?unk?>"
UNK_IDX = 2

END_OF_SENTENCE_TEXT = "<END_OF_SENTENCE>"


def tokenize_into_morphemes(morpheme):
    """Tokenise a morpheme into its morphemes... which is just the morpheme itself"""
    return [morpheme]  # The word passed in is already a list of morphemes


def tokenize_into_lower_morphemes(morpheme):
    """Tokenise a word into its morphemes, lowercased (just the morpheme itself lowercased)"""
    return [morpheme.lower()]


def tokenize_into_chars(morpheme):
    """Tokenise a morphemes into its characters"""
    return list(morpheme) if morpheme != WORD_SEP_TEXT else [WORD_SEP_TEXT]


def tokenize_into_lower_chars(morpheme):
    """Tokenise a morphemes into its characters"""
    return [c.lower() for c in morpheme] if morpheme != WORD_SEP_TEXT else [WORD_SEP_TEXT]


def tokenize_into_trigrams_with_sentinels(morpheme):
    """
    Tokenize a morpheme into its trigrams, using sentinels for the beginning and end.
    For instance, "ndi" would be tokenized into <START_MORPHEME, n, d>, <n, d, i>, <d, i, END_MORPHEME>
    """
    morpheme = ["START_MORPHEME"] + list(morpheme) + ["END_MORPHEME"] if morpheme != WORD_SEP_TEXT else [WORD_SEP_TEXT]
    return [(morpheme[i], morpheme[i + 1], morpheme[i + 2]) for i in range(len(morpheme) - 2)]


def tokenize_into_trigrams_no_sentinels(word):
    """
    Tokenize a morpheme into its trigrams, without using sentinels for the beginning and end.
    For instance, "enza" would be tokenized into <e, n, z>, <n, z, a>
    """
    word = list(word) if word != WORD_SEP_TEXT else [WORD_SEP_TEXT]
    if len(word) == 1:
        return [word[0], SEQ_PAD_TEXT, SEQ_PAD_TEXT]
    elif len(word) == 2:
        return [word[0], word[1], SEQ_PAD_TEXT]
    else:
        return [(word[i], word[i + 1], word[i + 2]) for i in range(len(word) - 2)]


def identity(x):
    """Simple identity function (used to map tags when wanting to use the full tagset)"""
    return x


def tags_only_no_classes(tag):
    """Map an NCHLT dataset grammatical tag to just the syntactic tag without noun classes. E.g., Dem14 -> Dem"""
    digits = [(i, c) for i, c in enumerate(tag) if c.isdigit()]
    if len(digits) > 0:
        return tag[:digits[0][0]]
    else:
        return tag


def classes_only_no_tags(tag):
    """Map an NCHLT dataset grammatical tag to just the noun class without syntactic tag. E.g., Dem14 -> 14"""
    digits = [(i, c) for i, c in enumerate(tag) if c.isdigit()]
    if len(digits) > 0:
        return tag[digits[0][0]:]
    else:
        return "NON_CLASS"


def prepare_sequence(seq, to_ix, device=None) -> torch.Tensor:
    """Encode a given sequence as a tensor of indices (from the to_ix dict)"""
    idxs = [to_ix[w] if w in to_ix else UNK_IDX for w in seq]
    return torch.tensor(idxs, device=device)


_DOUBLE_LABEL_PAT = re.compile("Pos[0-9]")


def _normalize_double_labelled_morphemes(morpheme_seq: list, tag_seq: list):
    """Combine double-tagged morphemes into one tag"""
    if len(morpheme_seq) == len(tag_seq):
        return

    pos_tag_ix = [i for i, tag in enumerate(tag_seq) if _DOUBLE_LABEL_PAT.match(tag)][0]
    tag_seq[pos_tag_ix - 1] += tag_seq[pos_tag_ix]
    tag_seq.pop(pos_tag_ix)

    # Can't use more than one ix since it will shift everything post merge - easier to just recurse
    _normalize_double_labelled_morphemes(morpheme_seq, tag_seq)


class DatasetWord:
    def __init__(self, raw: str, morphemes: list[str], tags: list[str]):
        self.raw = raw
        self.morphemes = morphemes
        self.tags = tags

    def __repr__(self):
        return f"DatasetWord(raw={self.raw}, morphemes={self.morphemes}, tags={self.tags})"

class DatasetSentence:
    def __init__(self, words: list[DatasetWord]):
        self.words = words

    def __repr__(self):
        return f"DatasetSentence(words={self.words})"

def split_words(sentences: list[DatasetSentence]) -> list[DatasetSentence]:
    """Split the corpus into words"""
    return [DatasetSentence([word]) for sentence in sentences for word in sentence.words]

def _add_word_separators(sentence: DatasetSentence) -> Generator[DatasetWord, None, None]:
    for word in sentence.words[:-1]:
        yield word
        yield DatasetWord(WORD_SEP_TEXT, [WORD_SEP_TEXT], [WORD_SEP_TEXT])
    yield sentence.words[-1]

def split_sentences(sentences: list[DatasetSentence]) -> list[DatasetSentence]:
    """Split the corpus into sentences with word separators"""
    return [DatasetSentence(list(_add_word_separators(sentence))) for sentence in sentences]

def split_sentences_no_sep(sentences: list[DatasetSentence]) -> list[DatasetSentence]:
    """Split the corpus into sentences without word separators"""
    return sentences


def load_dataset_into_sentences(filename: Path) -> Generator[DatasetSentence, None, None]:
    with open(filename) as f:
        sentence = []
        for line in f.readlines():
            line = line.strip()
            if line == END_OF_SENTENCE_TEXT:
                yield DatasetSentence(sentence)
                sentence = []
                continue

            cols = line.split("\t")
            raw = cols[0]
            morpheme_seq = cols[2].split("_")
            tag_seq = cols[3].split("_")

            # Clean the double-labelled morphemes (by combining them)
            _normalize_double_labelled_morphemes(morpheme_seq, tag_seq)

            assert len(morpheme_seq) == len(tag_seq), f"Wrong len! morphemes: {morpheme_seq}, tags: {tag_seq}"

            sentence.append(DatasetWord(raw, morpheme_seq, tag_seq))

    if len(sentence) != 0:
        yield DatasetSentence(sentence)


def _load_dataset_files(lang: str, data_dir: Path, split_fn) -> dict[str, list[DatasetSentence]]:
    portions = dict()
    for portion in ["train", "dev", "test"]:
        sentences = list(load_dataset_into_sentences(data_dir / portion / f"{lang}_{portion}.tsv"))
        portions[portion] = split_fn(sentences)
    return portions

class EncodedTaggingDatasetItem:
    def __init__(self, morphemes: torch.Tensor, tags: torch.Tensor):
        self.morphemes: torch.Tensor = morphemes
        self.tags: torch.Tensor = tags

    def to(self, device: torch.device) -> Self:
        return EncodedTaggingDatasetItem(self.morphemes.to(device), self.tags.to(device))

class EncodedTaggingDatasetBatch:
    def __init__(self, morphemes: torch.Tensor, tags: torch.Tensor):
        self.morphemes: torch.Tensor = morphemes
        self.tags: torch.Tensor = tags

class TaggingDataset(Dataset):
    """
    `AnnotatedCorpusDataset` represents a loaded and parsed annotated corpus dataset (e.g. Gaustad & Puttkammer 2022).
    It contains all morpheme and tag sequences as well as dictionaries mapping from submorphemes and tags to indices.
    One instance of `AnnotatedCorpusDataset` will either be the training or validation/testing portion.
    """

    def __init__(self, seqs: list, ix_to_tag, tag_to_ix, ix_to_morpheme, morpheme_to_ix, tokenize, split, lang):
        super().__init__()
        self.seqs: list = seqs
        self.num_submorphemes = len(morpheme_to_ix)
        self.num_tags = len(tag_to_ix)
        self.ix_to_tag = ix_to_tag
        self.tag_to_ix = tag_to_ix
        self.ix_to_morpheme = ix_to_morpheme
        self.morpheme_to_ix = morpheme_to_ix
        self.tokenize = tokenize
        self.split = split
        self.lang = lang

    @staticmethod
    def load_data(lang: str, data_dir: Path, split, tokenize=tokenize_into_morphemes,
                  min_submorpheme_frequency=1, map_tag=identity, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        """
        Load the data from the annotated corpus dataset, and return the training and validation portions.

        The major steps are as follows:
        1. Load raw train data
        2. Split into train/valid, OR load test data
        3. Count submorpheme frequencies (this refers to tokenization level - either morpheme or character)
        4. Create submorpheme<-->index and tag<-->index mappings
        5. Replace raw (text) data with indices and transform into tensors
        6. Return train & valid portions of dataset
        """

        # Initialise some index dictionaries used to map submorphemes & tags to indices
        submorpheme_to_ix = {SEQ_PAD_TEXT: SEQ_PAD_IX, WORD_SEP_TEXT: 1, UNK_TEXT: UNK_IDX}  # unk accounts for unseen morphemes
        tag_to_ix = {SEQ_PAD_TEXT: SEQ_PAD_IX, WORD_SEP_TEXT: 1}
        submorpheme_frequencies = dict()

        def insert_tags_into_dicts(tag_sequence, is_train):
            """Insert all tags from the given tag sequence into the tag <-> index dictionaries"""
            for tag in tag_sequence:
                tag = map_tag(tag)
                if tag not in tag_to_ix:
                    if not is_train:
                        print(f"Tag {tag} not found in trainset!")

                    ix = len(tag_to_ix)
                    tag_to_ix[tag] = ix

        # STEP 1: load dataset
        dataset_portions = _load_dataset_files(lang, data_dir, split)
        train = dataset_portions["train"]
        test = dataset_portions["test"]
        dev = dataset_portions["dev"]

        # STEP 2: Count token (=submorpheme) frequencies.
        # We want to replace any token seen less `min_token_frequency` than with `UNK` to improve generalization,
        # so we only include submorphemes which occur more than once
        for sentence in train:
            for word in sentence.words:
                for morpheme in word.morphemes:
                    for submorpheme in tokenize(morpheme):
                        submorpheme_frequencies.setdefault(submorpheme, 0)
                        submorpheme_frequencies[submorpheme] += 1

        # STEP 3: Create tag<-->index and submorpheme<-->index mappings
        for sentence in train:
            for word in sentence.words:
                # Insert submorphemes of morphemes from train set into the embedding indices
                # Replace those with only 1 occurence with UNK though
                for morpheme in word.morphemes:
                    for submorpheme in tokenize(morpheme):
                        if submorpheme_frequencies[submorpheme] > min_submorpheme_frequency:
                            submorpheme_to_ix.setdefault(submorpheme, len(submorpheme_to_ix))

                # Also insert tags into embedding indices
                insert_tags_into_dicts(word.tags, True)

        unseen_morphemes = set()  # We also track any morphemes present in dev/test but not in train
        for sentence in test + dev:
            # We skip inserting morphemes from the test/dev set into the embedding indices, because it is realistic
            # that there may be unseen morphemes in the final data, of course

            # However, we _do_ insert tags, since we know all the tags upfront. Technically we should just have
            # a predefined list, but that's annoying to do

            for word in sentence.words:
                insert_tags_into_dicts(word.tags, False)

                for morpheme in word.morphemes:
                    for submorpheme in tokenize(morpheme):
                        if submorpheme not in submorpheme_to_ix:
                            unseen_morphemes.add(submorpheme)

        print(f"{(len(unseen_morphemes) / len(submorpheme_to_ix)) * 100.0}% submorphemes not found in train!")

        # STEP 4: Transform to tensors
        ix_to_tag = {v : k for k, v in tag_to_ix.items()}
        ix_to_submorpheme = {v : k for k, v in submorpheme_to_ix.items()}

        dataset_portions = {
            portion : TaggingDataset(
                list(TaggingDataset.encode_dataset(sentences, tokenize, map_tag, tag_to_ix, submorpheme_to_ix, device)),
                ix_to_tag, tag_to_ix, ix_to_submorpheme, submorpheme_to_ix, tokenize, split, lang
            )
            for portion, sentences in dataset_portions.items()
        }

        print("train, valid, test len:", len(dataset_portions["train"]), len(dataset_portions["dev"]), len(dataset_portions["test"]))

        return dataset_portions

    @staticmethod
    def encode_dataset(sentences: list[DatasetSentence], tokenize, map_tag, tag_to_ix, submorpheme_to_ix, device)\
            -> Generator[EncodedTaggingDatasetItem, None, None]:
        """Encode the dataset into tensors by converting all submorphemes/tags to indices and padding them"""

        for sentence in sentences:
            morphemes = [morpheme for word in sentence.words for morpheme in word.morphemes]
            morphemes = [tokenize(morpheme) for morpheme in morphemes]
            morphemes = [prepare_sequence(morpheme, submorpheme_to_ix, device) for morpheme in morphemes]
            morphemes = pad_sequence(morphemes, padding_value=SEQ_PAD_IX, batch_first=True)

            tags = [tag for word in sentence.words for tag in word.tags]
            tags = [map_tag(tag) for tag in tags]
            tags = prepare_sequence(tags, tag_to_ix, device)

            yield EncodedTaggingDatasetItem(morphemes, tags)

    def to(self, device):
        """Return a copy of the dataset on a specific device"""
        return TaggingDataset([item.to(device) for item in self.seqs], self.ix_to_tag, self.tag_to_ix, self.ix_to_morpheme, self.morpheme_to_ix, self.tokenize, self.split, self.lang)

    def __getitem__(self, item):
        return self.seqs[item]

    def __len__(self):
        return len(self.seqs)

    def __add__(self, other: Self):
        assert (self.ix_to_tag == other.ix_to_tag
                and self.tag_to_ix == other.tag_to_ix
                and self.ix_to_morpheme == other.ix_to_morpheme
                and self.morpheme_to_ix == other.morpheme_to_ix
                and self.tokenize == other.tokenize
                and self.split == other.split
                and self.lang == other.lang)

        return TaggingDataset(
            self.seqs + other.seqs,
            self.ix_to_tag, self.tag_to_ix, self.ix_to_morpheme, self.morpheme_to_ix, self.tokenize, self.split, self.lang
        )

    @staticmethod
    def collate_batch(batch):
        """Collate sequences by padding them - used to adapt `AnnotatedCorpusDataset` to torch's `DataLoader`"""

        # Check if the morphemes are divided into submorphemes
        # If so, we need to pad every item in the batch to the same length
        if batch[0].morphemes.dim() == 2:
            longest_submorphemes = max(item.morphemes.size(dim=1) for item in batch)
            for (i, item) in enumerate(batch):
                pad = nn.ConstantPad1d((0, longest_submorphemes - item.morphemes.size(dim=1)), SEQ_PAD_IX)
                batch[i] = EncodedTaggingDatasetItem(pad(item.morphemes), item.tags)

        morphemes = pad_sequence([item.morphemes for item in batch], batch_first=True, padding_value=SEQ_PAD_IX)
        expected_tags = pad_sequence([item.tags for item in batch], batch_first=True, padding_value=SEQ_PAD_IX)
        return EncodedTaggingDatasetBatch(morphemes, expected_tags)


class EncodedSegmentationDatasetItem:
    def __init__(self, raw_word: torch.Tensor, morphemes: torch.Tensor):
        self.raw_word: torch.Tensor = raw_word
        self.morphemes: torch.Tensor = morphemes

    def to(self, device: torch.device) -> Self:
        return EncodedSegmentationDatasetItem(self.raw_word.to(device), self.morphemes.to(device))

    def __repr__(self):
        return f"EncodedSegmentationDatasetItem({self.raw_word}, {self.morphemes})"

class EncodedSegmentationDatasetBatch:
    def __init__(self, raw_word: torch.Tensor, morphemes: torch.Tensor):
        self.raw_words: torch.Tensor = raw_word
        self.morphemes: torch.Tensor = morphemes


class SegmentationDataset(Dataset):
    """
    `AnnotatedCorpusDataset` represents a loaded and parsed annotated corpus dataset (e.g. Gaustad & Puttkammer 2022).
    It contains all morpheme and tag sequences as well as dictionaries mapping from submorphemes and tags to indices.
    One instance of `AnnotatedCorpusDataset` will either be the training or validation/testing portion.
    """

    def __init__(self, seqs: list, input_tokenizer, output_tokenizer, input_subword_to_ix, output_subword_to_ix, lang):
        super().__init__()
        self.seqs: list = seqs
        self.num_input_subwords = len(input_subword_to_ix)
        self.num_output_subwords = len(output_subword_to_ix)

        self.input_tokenizer = input_tokenizer
        self.input_subword_to_ix = input_subword_to_ix
        self.ix_to_input_subword = {v:k for k, v in input_subword_to_ix.items()}

        self.output_tokenizer = output_tokenizer
        self.output_subword_to_ix = output_subword_to_ix
        self.ix_to_output_subword = {v:k for k, v in output_subword_to_ix.items()}

        self.lang = lang

    @staticmethod
    def load_data(
            lang: str,
            data_dir: Path,
            input_tokenizer_builder: InputTokenizerBuilder,
            output_tokenizer_builder: OutputTokenizerBuilder,
            split,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ) -> dict[str, 'SegmentationDataset']:
        """
        Load the data from the annotated corpus dataset, and return the training, validation, and test portions.

        The major steps are as follows:
        1. Load raw train data
        2. Split into train/valid/test
        3. Build tokenizer
        4. Create submorpheme<-->index and tag<-->index mappings
        5. Replace raw (text) data with indices and transform into tensors
        6. Return all portions of dataset
        """

        # Initialise some index dictionaries used to map submorphemes & tags to indices
        input_subword_to_ix = {SEQ_PAD_TEXT: SEQ_PAD_IX, WORD_SEP_TEXT: 1, UNK_TEXT: UNK_IDX}  # unk accounts for unseen morphemes
        output_subword_to_ix = {SEQ_PAD_TEXT: SEQ_PAD_IX, WORD_SEP_TEXT: 1, MORPHEME_SEP_TEXT: 2}

        # STEP 1: load dataset
        dataset_portions = _load_dataset_files(lang, data_dir, split)
        train = dataset_portions["train"]

        for sentence in train:
            for word in sentence.words:
                input_tokenizer_builder.add_word(word.raw)
                output_tokenizer_builder.add_word(word.morphemes)

        # STEP 3: build tokenizer
        input_tokenizer = input_tokenizer_builder.build()
        output_tokenizer = output_tokenizer_builder.build()

        # STEP 4: Create tag<-->index and subword<-->index mappings
        for sentence in train:
            for word in sentence.words:
                for in_subword in input_tokenizer.tokenize(word.raw):
                    input_subword_to_ix.setdefault(in_subword, len(input_subword_to_ix))

                for out_subword in output_tokenizer.tokenize(word.morphemes):
                    output_subword_to_ix.setdefault(out_subword, len(output_subword_to_ix))

        # STEP 5: Transform to tensors
        dataset_portions = {
            portion : SegmentationDataset(
                list(SegmentationDataset.encode_dataset(sentences, input_tokenizer, output_tokenizer, input_subword_to_ix, output_subword_to_ix, device)),
                input_tokenizer,
                output_tokenizer,
                input_subword_to_ix,
                output_subword_to_ix,
                lang
            )
            for portion, sentences in dataset_portions.items()
        }

        print("train, valid, test len:", len(dataset_portions["train"]), len(dataset_portions["dev"]), len(dataset_portions["test"]))

        return dataset_portions

    @staticmethod
    def encode_dataset(
            sentences: list[DatasetSentence],
            input_tokenizer: InputTokenizer,
            output_tokenizer: OutputTokenizer,
            input_subword_to_ix: dict[str, int],
            output_subword_to_ix: dict[str, int],
            device
    ) -> Generator[EncodedSegmentationDatasetItem, None, None]:
        """Encode the dataset into tensors by converting all input subwords/output subwords to indices and padding them"""

        for sentence in sentences:
            input_words = [word.raw for word in sentence.words]
            input_words = [subword for word in input_words for subword in input_tokenizer.tokenize(word)]
            input_words = prepare_sequence(input_words, input_subword_to_ix, device)

            morphemes = [token for word in sentence.words for token in output_tokenizer.tokenize(word.morphemes)]
            morphemes = prepare_sequence(morphemes, output_subword_to_ix, device)

            yield EncodedSegmentationDatasetItem(input_words, morphemes)

    def to(self, device: torch.device) -> Self:
        """Return a copy of the dataset on a specific device"""
        return SegmentationDataset([item.to(device) for item in self.seqs], self.input_tokenizer, self.output_tokenizer, self.input_subword_to_ix, self.output_subword_to_ix, self.lang)

    def __getitem__(self, item):
        return self.seqs[item]

    def __len__(self):
        return len(self.seqs)

    def __add__(self, other: Self):
        assert (self.ix_to_input_subword == other.ix_to_input_subword
                and self.ix_to_output_subword == other.ix_to_output_subword
                and self.input_subword_to_ix == other.input_subword_to_ix
                and self.output_subword_to_ix == other.output_subword_to_ix
                and self.input_tokenizer == other.input_tokenizer
                and self.output_tokenizer == other.output_tokenizer
                and self.lang == other.lang)

        return SegmentationDataset(
            self.seqs + other.seqs,
            self.input_tokenizer, self.output_tokenizer, self.input_subword_to_ix, self.output_subword_to_ix, self.lang
        )

    __iter__: typing.Callable[..., typing.Iterator[EncodedSegmentationDatasetItem]]

class EncodedParsingDatasetItem(EncodedSegmentationDatasetItem):
    def __init__(self, raw_word: torch.Tensor, morphemes: torch.Tensor, tags: torch.Tensor):
        super().__init__(raw_word, morphemes)
        self.tags: torch.Tensor = tags

    def to(self, device: torch.device) -> Self:
        return EncodedParsingDatasetItem(self.raw_word.to(device), self.morphemes.to(device), self.tags.to(device))

class EncodedParsingDatasetBatch(EncodedSegmentationDatasetBatch):
    def __init__(self, raw_word: torch.Tensor, morphemes: torch.Tensor, tags: torch.Tensor):
        super().__init__(raw_word, morphemes)
        self.tags: torch.Tensor = tags

class ParsingPipelineDataset(SegmentationDataset):
    """
    `AnnotatedCorpusDataset` represents a loaded and parsed annotated corpus dataset (e.g. Gaustad & Puttkammer 2022).
    It contains all morpheme and tag sequences as well as dictionaries mapping from submorphemes and tags to indices.
    One instance of `AnnotatedCorpusDataset` will either be the training or validation/testing portion.
    """

    def __init__(self, seqs: list, input_tokenizer, output_tokenizer, input_subword_to_ix, output_subword_to_ix, tag_to_ix, lang):
        super().__init__(seqs, input_tokenizer, output_tokenizer, input_subword_to_ix, output_subword_to_ix, lang)
        self.tag_to_ix = tag_to_ix
        self.ix_to_tag = {v:k for k, v in tag_to_ix.items()}

    @staticmethod
    def load_data(
            lang: str,
            data_dir: Path,
            input_tokenizer_builder: InputTokenizerBuilder,
            output_tokenizer_builder: OutputTokenizerBuilder,
            split,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ) -> dict[str, 'ParsingPipelineDataset']:
        """
        Load the data from the annotated corpus dataset, and return the training, validation, and test portions.

        The major steps are as follows:
        1. Load raw train data
        2. Split into train/valid/test
        3. Build tokenizer
        4. Create submorpheme<-->index and tag<-->index mappings
        5. Replace raw (text) data with indices and transform into tensors
        6. Return all portions of dataset
        """

        # Initialise some index dictionaries used to map submorphemes & tags to indices
        input_subword_to_ix = {SEQ_PAD_TEXT: SEQ_PAD_IX, WORD_SEP_TEXT: 1, UNK_TEXT: UNK_IDX}  # unk accounts for unseen morphemes
        output_subword_to_ix = {SEQ_PAD_TEXT: SEQ_PAD_IX, WORD_SEP_TEXT: 1, MORPHEME_SEP_TEXT: 2}
        tag_to_ix = {SEQ_PAD_TEXT: SEQ_PAD_IX, WORD_SEP_TEXT: 1}

        # STEP 1: load dataset
        dataset_portions = _load_dataset_files(lang, data_dir, split)
        train, test, valid = dataset_portions["train"], dataset_portions["test"], dataset_portions["dev"]

        for sentence in train:
            for word in sentence.words:
                input_tokenizer_builder.add_word(word.raw)
                output_tokenizer_builder.add_word(word.morphemes)

        # STEP 3: build tokenizer
        input_tokenizer = input_tokenizer_builder.build()
        output_tokenizer = output_tokenizer_builder.build()

        # STEP 4: Create tag<-->index and subword<-->index mappings
        for sentence in train:
            for word in sentence.words:
                for in_subword in input_tokenizer.tokenize(word.raw):
                    input_subword_to_ix.setdefault(in_subword, len(input_subword_to_ix))

                for out_subword in output_tokenizer.tokenize(word.morphemes):
                    output_subword_to_ix.setdefault(out_subword, len(output_subword_to_ix))

                for tag in word.tags:
                    tag_to_ix.setdefault(tag, len(tag_to_ix))

        # Still include tags from test + valid as these are knowable ahead of time
        for sentence in test + valid:
            for word in sentence.words:
                for tag in word.tags:
                    tag_to_ix.setdefault(tag, len(tag_to_ix))

        # STEP 5: Transform to tensors
        dataset_portions = {
            portion : ParsingPipelineDataset(
                list(ParsingPipelineDataset.encode_dataset(sentences, input_tokenizer, output_tokenizer, input_subword_to_ix, output_subword_to_ix, tag_to_ix, device)),
                input_tokenizer,
                output_tokenizer,
                input_subword_to_ix,
                output_subword_to_ix,
                tag_to_ix,
                lang
            )
            for portion, sentences in dataset_portions.items()
        }

        print("train, valid, test len:", len(dataset_portions["train"]), len(dataset_portions["dev"]), len(dataset_portions["test"]))

        return dataset_portions

    # This is a static method
    # noinspection PyMethodOverriding
    @staticmethod
    def encode_dataset(
            sentences: list[DatasetSentence],
            input_tokenizer: InputTokenizer,
            output_tokenizer: OutputTokenizer,
            input_subword_to_ix: dict[str, int],
            output_subword_to_ix: dict[str, int],
            tag_to_ix: dict[str, int],
            device
    ) -> Generator[EncodedParsingDatasetItem, None, None]:
        """Encode the dataset into tensors by converting all input subwords/output subwords to indices and padding them"""

        for sentence in sentences:
            input_words = [word.raw for word in sentence.words]
            input_words = [subword for word in input_words for subword in input_tokenizer.tokenize(word)]
            input_words = prepare_sequence(input_words, input_subword_to_ix, device)

            morphemes = [token for word in sentence.words for token in output_tokenizer.tokenize(word.morphemes)]
            morphemes = prepare_sequence(morphemes, output_subword_to_ix, device)

            tags = [tag for word in sentence.words for tag in word.tags]
            tags = prepare_sequence(tags, tag_to_ix, device)

            yield EncodedParsingDatasetItem(input_words, morphemes, tags)


    def to(self, device: torch.device) -> Self:
        """Make a copy of dataset on a specific device"""
        return ParsingPipelineDataset([item.to(device) for item in self.seqs], self.input_tokenizer, self.output_tokenizer, self.input_subword_to_ix, self.output_subword_to_ix, self.tag_to_ix, self.lang)

    def __getitem__(self, item):
        return self.seqs[item]

    def __len__(self):
        return len(self.seqs)

    def __add__(self, other: Self):
        assert (self.ix_to_input_subword == other.ix_to_input_subword
                and self.ix_to_output_subword == other.ix_to_output_subword
                and self.input_subword_to_ix == other.input_subword_to_ix
                and self.output_subword_to_ix == other.output_subword_to_ix
                and self.tag_to_ix == other.tag_to_ix
                and self.ix_to_tag == other.ix_to_tag
                and self.input_tokenizer == other.input_tokenizer
                and self.output_tokenizer == other.output_tokenizer
                and self.lang == other.lang)

        return ParsingPipelineDataset(
            self.seqs + other.seqs,
            self.input_tokenizer, self.output_tokenizer, self.input_subword_to_ix, self.output_subword_to_ix,
            self.tag_to_ix, self.lang
        )

    __iter__: typing.Callable[..., typing.Iterator[EncodedParsingDatasetItem]]


class EncodedJointParsingDatasetItem(EncodedParsingDatasetItem):
    def __init__(self, raw_word: torch.Tensor, morphemes_and_tags: torch.Tensor, first_tag_ix: int):
        # We override it with our own property
        # noinspection PyTypeChecker
        super().__init__(raw_word, None, None)
        self.morphemes_and_tags = morphemes_and_tags
        self.first_tag_ix = first_tag_ix

    @staticmethod
    def from_base(base: EncodedParsingDatasetItem, ix_to_output_token, tag_offset: int, first_tag_ix: int, morpheme_sep_ix: int, word_sep_ix: int) -> 'EncodedParsingDatasetItem':
        tags, morphemes = base.tags.tolist(), base.morphemes.tolist()

        morphemes_and_tags_list = []

        for token in morphemes:
            if token in [morpheme_sep_ix, word_sep_ix]:
                tag = tags.pop(0) + tag_offset
                morphemes_and_tags_list.append(tag)

                if token == word_sep_ix:
                    tags.pop(0) # Next tag is word sep - we don't need it to appear both as a morpheme and as a tag

            morphemes_and_tags_list.append(token)

        # Add tag for last morpheme
        morphemes_and_tags_list.append(tags.pop(0) + tag_offset)
        assert len(tags) == 0, f"tags left! {'-'.join(ix_to_output_token[t + tag_offset] for t in tags)}"

        morphemes_and_tags = torch.tensor(morphemes_and_tags_list, device=base.morphemes.device)
        return EncodedJointParsingDatasetItem(base.raw_word, morphemes_and_tags, first_tag_ix)

    def to(self, device: torch.device):
        return EncodedJointParsingDatasetItem(self.raw_word.to(device), self.morphemes_and_tags.to(device), self.first_tag_ix)

    @property
    def tags(self):
        return self.morphemes_and_tags[self.morphemes_and_tags >= self.first_tag_ix]

    @tags.setter
    def tags(self, tags):  # Empty for base class to "set"
        pass

    @property
    def morphemes(self):
        return self.morphemes_and_tags[self.morphemes_and_tags < self.first_tag_ix]

    @morphemes.setter
    def morphemes(self, morphemes):
        pass


class EncodedJointParsingDatasetBatch(EncodedParsingDatasetBatch):
    def __init__(self, raw_word: torch.Tensor, morphemes_and_tags: torch.Tensor, first_tag_ix: int):
        # We override it with our own property
        # noinspection PyTypeChecker
        super().__init__(raw_word, None, None)
        self.morphemes_and_tags = morphemes_and_tags
        self.first_tag_ix = first_tag_ix

    def to(self, device: torch.device) -> Self:
        return EncodedJointParsingDatasetBatch(self.raw_words.to(device), self.morphemes_and_tags.to(device), self.first_tag_ix)

    @property
    def tags(self):
        return pad_sequence(
            [seq[seq >= self.first_tag_ix] for seq in self.morphemes_and_tags.unbind(dim=0)],
            batch_first=True, padding_value=SEQ_PAD_IX
        )

    @property
    def morphemes(self):
        return pad_sequence(
            [seq[seq < self.first_tag_ix] for seq in self.morphemes_and_tags.unbind(dim=0)],
            batch_first=True, padding_value=SEQ_PAD_IX
        )


class JointEndToEndParsingDataset(ParsingPipelineDataset):
    """
    `AnnotatedCorpusDataset` represents a loaded and parsed annotated corpus dataset (e.g. Gaustad & Puttkammer 2022).
    It contains all morpheme and tag sequences as well as dictionaries mapping from submorphemes and tags to indices.
    One instance of `AnnotatedCorpusDataset` will either be the training or validation/testing portion.
    """

    def __init__(self, seqs: list, input_tokenizer, output_tokenizer, input_subword_to_ix, output_token_to_ix, first_tag_ix, lang):
        super().__init__(seqs, input_tokenizer, output_tokenizer, input_subword_to_ix, output_token_to_ix, output_token_to_ix, lang)

        self.output_token_to_ix = output_token_to_ix
        self.ix_to_output_token = {v:k for k, v in self.output_token_to_ix.items()}

        self.ix_to_tag = {ix : tag for tag, ix in self.output_token_to_ix.items() if ix >= first_tag_ix}
        self.tag_to_ix = {v:k for k, v in self.ix_to_tag.items()}
        self.first_tag_ix = first_tag_ix


    @staticmethod
    def from_parsing_pipeline_dataset(base: ParsingPipelineDataset):
        output_token_to_ix = dict(base.output_subword_to_ix)
        first_tag_ix = len(output_token_to_ix)

        tag_offset = first_tag_ix
        # Remap tags
        for tag in base.tag_to_ix:
            if tag in base.output_subword_to_ix:
                tag_offset -= 1
                continue
            output_token_to_ix[f"[{tag}]"] = len(output_token_to_ix)

        morpheme_sep_ix = output_token_to_ix[MORPHEME_SEP_TEXT]
        word_sep_ix = output_token_to_ix[WORD_SEP_TEXT]
        seqs = [EncodedJointParsingDatasetItem.from_base(item, {v:k for k, v in output_token_to_ix.items()}, tag_offset, first_tag_ix, morpheme_sep_ix, word_sep_ix) for item in base.seqs]
        return JointEndToEndParsingDataset(seqs, base.input_tokenizer, base.output_tokenizer, base.input_subword_to_ix, output_token_to_ix, first_tag_ix, base.lang)

    def to(self, device: torch.device) -> Self:
        """Make a copy of dataset on a specific device"""
        return JointEndToEndParsingDataset([it.to(device) for it in self.seqs], self.input_tokenizer, self.output_tokenizer, self.input_subword_to_ix, self.output_token_to_ix, self.first_tag_ix, self.lang)


    @staticmethod
    def load_data(
            lang: str,
            data_dir: Path,
            input_tokenizer_builder: InputTokenizerBuilder,
            output_tokenizer_builder: OutputTokenizerBuilder,
            split,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ) -> dict[str, 'JointEndToEndParsingDataset']:
        portions = ParsingPipelineDataset.load_data(lang, data_dir, input_tokenizer_builder, output_tokenizer_builder, split, device)
        portions = { name : JointEndToEndParsingDataset.from_parsing_pipeline_dataset(dset) for name, dset in portions.items() }
        return portions

    def __add__(self, other: Self):
        assert (self.ix_to_input_subword == other.ix_to_input_subword
                and self.ix_to_output_subword == other.ix_to_output_subword
                and self.input_subword_to_ix == other.input_subword_to_ix
                and self.output_subword_to_ix == other.output_subword_to_ix
                and self.tag_to_ix == other.tag_to_ix
                and self.ix_to_tag == other.ix_to_tag
                and self.input_tokenizer == other.input_tokenizer
                and self.output_tokenizer == other.output_tokenizer
                and self.lang == other.lang)

        return JointEndToEndParsingDataset(
            self.seqs + other.seqs,
            self.input_tokenizer, self.output_tokenizer, self.input_subword_to_ix, self.output_subword_to_ix,
            self.tag_to_ix, self.lang
        )

    __iter__: typing.Callable[..., typing.Iterator[EncodedJointParsingDatasetItem]]


