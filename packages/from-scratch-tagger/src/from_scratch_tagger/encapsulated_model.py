import pprint
import sys
from typing import Self

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from annotated_corpus_dataset import TaggingDataset, WORD_SEP_TEXT, UNK_IDX, SEQ_PAD_IX, DatasetSentence, DatasetWord


class EncapsulatedModel(nn.Module):
    """
    An EncapsulatedModel wraps a BiLSTMTagger or BiLSTMCrfTagger to accept segmented morphemes, thus performing
    any necessary data prep / mapping.
    """

    def __init__(
            self,
            name,
            model: nn.Module,
            dataset: TaggingDataset,
            device,
    ):
        super(EncapsulatedModel, self).__init__()
        self.name = name
        self.lang = dataset.lang
        self.is_surface = vars(dataset).get("is_surface") or False
        self.ix_to_tag = dataset.ix_to_tag
        self.submorpheme_to_ix = dataset.morpheme_to_ix
        self.model = model
        self.tokenize = dataset.tokenize
        self.split = dataset.split
        self.device: torch.device = device

    def to_(self, device: torch.device) -> Self:
        super(EncapsulatedModel, self).to(device)
        self.model = self.model.to(device)
        self.device = device

    def eval(self) -> Self:
        self.model.eval()

    def __getstate__(self):
        state = dict(self.__dict__)
        del state["device"]
        return state

    def __setstate__(self, state):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return super(EncapsulatedModel, self).__setstate__(state)

    def forward(self, segmented_sentences):
        """
        Morphologically analyse the given sentences. The input format is expected to be a `list[list[list[str]]]`. Each
        element of the list is a sentence, which is a list of words, which is a list of its morphemes.

        Returns a `list[list[list[str]]]` (same format as above but the str is replaced by the tag)
        """

        batches = []
        for segmented_sentence in segmented_sentences:
            # Convert the list-of-lists format to a separator format
            sentence = []
            for word in segmented_sentence:
                sentence.append(DatasetWord("", word, []))
                sentence.append(DatasetWord(WORD_SEP_TEXT, [WORD_SEP_TEXT], [WORD_SEP_TEXT]))

            batches.append(DatasetSentence(sentence[:-1]))  # Discard the last separator

        # Encode all sentences and batch together
        batch_sentences_encoded = []
        batches_morphemes = []
        for batch in batches:
            # split functions expect an entire dataset (list) of sentences - hence we put the batch (one sentence)
            # in its own list
            for sentence in self.split([batch]):
                sentence_encoded = []
                sentence_morphemes = []

                # If we are at sentence-level, we have already put in word separator words, so we don't need to treat
                # them specially here
                for word in sentence.words:
                    for morpheme in word.morphemes:
                        morpheme_encoded = []
                        for submorpheme in self.tokenize(morpheme):
                            morpheme_encoded.append(self.submorpheme_to_ix[submorpheme] if submorpheme in self.submorpheme_to_ix else UNK_IDX)
                        sentence_encoded.append(torch.tensor(morpheme_encoded, device=self.device))
                        sentence_morphemes.append(morpheme)

                batches_morphemes.append(sentence_morphemes)
                batch_sentences_encoded.append(pad_sequence(sentence_encoded, padding_value=SEQ_PAD_IX, batch_first=True))


        # Run inference
        batches_encoded = pad_sequence(batch_sentences_encoded, padding_value=SEQ_PAD_IX, batch_first=True)
        batches_tagged = self.model.forward_tags_only(batches_encoded)

        tags_per_batch = []
        # Decode all sentences and re-batch together
        for sentence, morphemes in zip(torch.unbind(batches_tagged), batches_morphemes):
            sentence_tags = []
            word_tags = []
            for tag_ix, morpheme_text in zip(sentence.tolist(), morphemes):
                if morpheme_text == WORD_SEP_TEXT:
                    sentence_tags.append(word_tags)
                    word_tags = []
                    continue
                word_tags.append(self.ix_to_tag[tag_ix])

            if word_tags:
                sentence_tags.append(word_tags)

            tags_per_batch.append(sentence_tags)

        return tags_per_batch


def _with_sys_modules(func, **modules):
    old_modules = dict()
    for module_name, module in modules.items():
        if module_name in sys.modules:
            old_modules[module_name] = sys.modules[module_name]
        sys.modules[module_name] = module

    ret = func()

    for module_name, module in modules.items():
        if module_name in old_modules:
            sys.modules[module_name] = old_modules[module_name]

    return ret


def load_model(path, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    from . import lstm, bilstm_crf, common  # Has to be here to prevent circular importing
    import annotated_corpus_dataset

    model: EncapsulatedModel = _with_sys_modules(
        lambda: torch.load(path, map_location=device, weights_only=False),
        encapsulated_model=sys.modules[__name__],
        lstm=lstm,
        bilstm_crf=bilstm_crf,
        dataset=annotated_corpus_dataset,
        common=common,
    )
    model.eval()
    model.to_(device)
    return model


def annotate_sentence(model, words):
    model.eval()
    with torch.no_grad():
        annotated_words = [list(zip(word, tags)) for word, tags in zip(words, model.forward(words))]
        return ["-".join(f"{morpheme}[{tag}]" for morpheme, tag in word) for word in annotated_words]


def predict_tags_for_word(model, morphemes):
    model.eval()
    with torch.no_grad():
        # Return the first sentence's first word (we have one word per sentence, which there is also one of, here)
        return model.forward([[morphemes]])[0][0]


def predict_tags_for_words_batched(model, words):
    """Predict the tags for a list of _unrelated_ words (i.e. batch, not together in a sentence)"""
    model.eval()
    with torch.no_grad():
        # Wrap each word in a list so it is, itself, treated as a sentence
        tags = model.forward([[word] for word in words])

        # Return the first word of each sentence (we have one word per sentence here)
        return [sentence[0] for sentence in tags]
