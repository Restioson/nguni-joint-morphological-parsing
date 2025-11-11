from abc import ABC, abstractmethod

MORPHEME_SEP_TEXT = "<?morpheme_sep?>"
WORD_SEP_TEXT = "<?word_sep?>"

class InputTokenizer(ABC):
    @abstractmethod
    def tokenize(self, word: str) -> list[str]:
        """Tokenize the given word using the tokenizer"""

class InputTokenizerBuilder(ABC):
    @abstractmethod
    def add_word(self, word: str):
        """Add the word to the tokenizer"""
        pass

    @abstractmethod
    def build(self) -> InputTokenizer:
        """Build the tokenizer"""
        pass

class OutputTokenizer(ABC):
    @abstractmethod
    def tokenize(self, morphemes: list[str]) -> list[str]:
        """Tokenize the given word using the tokenizer"""

class OutputTokenizerBuilder(ABC):
    @abstractmethod
    def add_word(self, morphemes: list[str]):
        """Add the word to the tokenizer"""
        pass

    @abstractmethod
    def build(self) -> OutputTokenizer:
        """Build the tokenizer"""
        pass

class ReplaceWithUnkTokenizerBuilder(InputTokenizerBuilder):
    def __init__(self, tokenizer_builder: InputTokenizerBuilder, min_token_frequency: int, unk):
        self.builder = tokenizer_builder
        self.unk = unk
        self.words = []
        self.min_token_frequency = min_token_frequency

    def add_word(self, word: str):
        self.builder.add_word(word)
        self.words.append(word)

    def build(self) -> InputTokenizer:
        tokenizer = self.builder.build()
        token_freqs = dict()

        for word in self.words:
            for token in tokenizer.tokenize(word):
                token_freqs.setdefault(token, 0)
                token_freqs[token] += 1

        replace_with_unk = {token for token, freq in token_freqs.items() if freq < self.min_token_frequency}
        return ReplaceWithUnkTokenizer(tokenizer, replace_with_unk, self.unk)

class ReplaceWithUnkTokenizer(InputTokenizer):
    def __init__(self, tokenizer: InputTokenizer, replace_with_unk: set, unk):
        self.tokenizer = tokenizer
        self.replace_with_unk = replace_with_unk
        self.unk = unk

    def tokenize(self, word: str):
        return [(token if token not in self.replace_with_unk else self.unk) for token in self.tokenizer.tokenize(word)]


class CharacterInputTokenizerBuilder(InputTokenizerBuilder):
    def build(self) -> InputTokenizer:
        return CharacterInputTokenizer()

    def add_word(self, word: str):
        pass

class CharacterInputTokenizer(InputTokenizer):
    """Tokenize a word into characters"""
    def tokenize(self, word: str):
        return list(word) if word != WORD_SEP_TEXT else [WORD_SEP_TEXT]


class CharacterOutputTokenizerBuilder(OutputTokenizerBuilder):
    def build(self) -> OutputTokenizer:
        return CharacterOutputTokenizer()

    def add_word(self, word: str):
        pass

class CharacterOutputTokenizer(OutputTokenizer):
    """Tokenize a word into characters"""
    def tokenize(self, morphemes: list[str]) -> list[str]:
        if morphemes == [WORD_SEP_TEXT]:
            return [WORD_SEP_TEXT]

        out = []
        for morpheme in morphemes:
            out.extend(morpheme)
            out.append(MORPHEME_SEP_TEXT)
        return out[:-1]
