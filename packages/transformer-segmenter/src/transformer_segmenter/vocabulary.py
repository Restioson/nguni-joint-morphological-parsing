import torch
from torch.nn.utils.rnn import pad_sequence

import annotated_corpus_dataset
from annotated_corpus_dataset import SegmentationDataset, WORD_SEP_TEXT


class Vocabulary: # TODO shouldn't really rely on SegmentationDataset like this ideally...
    def __init__(self, dataset: SegmentationDataset):
        self.input_tokenizer = dataset.input_tokenizer

        self.input_start_token_ix = len(dataset.input_subword_to_ix)
        self.input_end_token_ix = len(dataset.input_subword_to_ix) + 1
        self.output_start_token_ix = len(dataset.output_subword_to_ix)
        self.output_end_token_ix = len(dataset.output_subword_to_ix) + 1

        self.output_subword_to_ix = dataset.output_subword_to_ix
        self.ix_to_output_subword = dataset.ix_to_output_subword

        self.input_subword_to_ix = dataset.input_subword_to_ix
        self.ix_to_input_subword = dataset.ix_to_input_subword

        self.output_pad_ix = self.input_pad_ix = annotated_corpus_dataset.SEQ_PAD_IX
        self.output_morpheme_sep_ix = self.output_subword_to_ix[annotated_corpus_dataset.MORPHEME_SEP_TEXT]
        self.output_word_sep_ix = self.output_subword_to_ix[annotated_corpus_dataset.WORD_SEP_TEXT]
        self.input_word_sep_ix = self.input_subword_to_ix.get(annotated_corpus_dataset.WORD_SEP_TEXT)

    # TODO explain input/output is still morphemes not raw
    def encode_sentence(self, sentence: list[str], device) -> torch.Tensor: # TODO import normalize
        src_tokens = [tok for word in sentence for tok in [self.input_tokenizer.tokenize(word), WORD_SEP_TEXT]]
        src_tokens = src_tokens[:-1] # Remove last WORD_SEP_TEXT
        src_indexes = [self.input_subword_to_ix[token] for token in src_tokens]
        src_indexes = [self.input_start_token_ix] + src_indexes + [self.input_end_token_ix]
        return torch.tensor(src_indexes, device=device).unsqueeze(0)

    def encode_sentences_batched(self, sentences: list[list[str]], device) -> torch.Tensor:
        return pad_sequence([self.encode_sentence(sentence, device) for sentence in sentences], batch_first=True, padding_value=self.input_pad_ix)

    def decode_batched_output_sentences(self, sentences: torch.Tensor) -> list[list[list[str]]]:
        return [self.decode_output_sentence(sentence) for sentence in sentences.unbind(0)]

    def decode_output_sentence(self, sentence: list[int] | torch.Tensor) -> list[list[str]]:
        """Decode the output into a list of morphemes"""
        return self._decode_sentence_inner(sentence, self.ix_to_output_subword, self.output_start_token_ix, self.output_end_token_ix, self.output_morpheme_sep_ix, self.output_word_sep_ix)

    def decode_batched_input_sentences(self, words: torch.Tensor) -> list[list[list[str]]]:
        return [self.decode_input_sentence(word) for word in words.unbind(0)]

    # TODO fix up rest
    def decode_input_sentence(self, word: list[int] | torch.Tensor) -> list[list[str]]:
        """Decode the output into a list of morphemes"""
        return self._decode_sentence_inner(word, self.ix_to_input_subword, self.input_start_token_ix, self.input_end_token_ix, None, self.input_word_sep_ix)


    @staticmethod
    def _decode_sentence_inner(word: list[int] | torch.Tensor, ix_to_subword, start_ix, end_ix, sep_ix, word_sep_ix) -> list[list[str]]:
        """Decode the indices into a list of morphemes"""
        if isinstance(word, torch.Tensor):
            word = word.tolist()

        out = [[""]]

        for ix in word:
            if ix == end_ix:
                return out
            elif ix == start_ix:
                continue
            elif ix == sep_ix:
                out[-1].append("")
            elif ix == word_sep_ix:
                out.append([""])
            else:
                out[-1][-1] += ix_to_subword[ix]

        return out

    def debug_batched_input_words(self, words: torch.Tensor) -> list[str]:
        return [self.debug_input_word(word) for word in words.unbind(0)]

    def debug_input_word(self, word: list[int] | torch.Tensor) -> str:
        return self._debug_word_inner(word, self.ix_to_input_subword, self.input_start_token_ix, self.input_end_token_ix, None)[0]

    def debug_batched_output_words(self, words: torch.Tensor) -> list[str]:
        return [self.debug_output_word(word) for word in words.unbind(0)]

    def debug_output_word(self, word: list[int] | torch.Tensor) -> str:
        return self._debug_word_inner(word, self.ix_to_output_subword, self.output_start_token_ix, self.output_end_token_ix, self.output_morpheme_sep_ix)

    @staticmethod
    def _debug_word_inner(word: list[int] | torch.Tensor, ix_to_subword, start_ix, end_ix, sep_ix) -> str:
        out = ""

        if isinstance(word, torch.Tensor):
            word = word.tolist()

        for ix in word:
            if ix == end_ix:
                return out + "<eos>"
            elif ix == start_ix:
                out += "<sos>"
                continue
            elif ix == sep_ix:
                out += "_"
            else:
                out += ix_to_subword[ix]

        return out
