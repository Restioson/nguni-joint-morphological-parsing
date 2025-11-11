import random
import re
import os
import shutil
import sys
from pathlib import Path
from typing import Generator

from .dataset import END_OF_SENTENCE_TEXT


def read_lines_raw(file_path: Path, skip_first=False) -> list:
    """Reads the lines of a file and returns them as a list."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
        return lines if not skip_first else lines[1:]

class ParsedLine:
    def __init__(self, line: str):
        line = line.rstrip()
        self.raw, self.parsed = line.split()[:2]
        self.morphemes, self.tags = split_tags(self.parsed)

    def __str__(self):
        return '\t'.join([self.raw, self.parsed,'_'.join(self.morphemes), '_'.join(self.tags)])

    def __repr__(self):
        return self.__str__()


def split_tags(text: str) -> tuple[list[str], list[str]]:
    """Split a word into its canonical segmentation and morpheme tags."""
    split = [morpheme for morpheme in re.split(r'\[[a-zA-Z-_0-9|]*?]-?', text) if morpheme != ""]
    return split, re.findall(r'\[([a-zA-Z-_0-9|]*?)]', text)

def write_sentences(file_path: Path, sentences: list[list[ParsedLine]]) -> None:
    """Writes a list of lines to a file."""
    with open(file_path, 'w') as f:
        f.write('\n'.join((str(sentence) for sentence in flatten_with_separator(sentences, END_OF_SENTENCE_TEXT))))

def split_sentences(lines: list[ParsedLine]) -> Generator[list[ParsedLine], None, None]:
    """Split the dataset into sentences of contiguous words (each word is a ParsedLine)."""
    sentence = []
    for line in lines:
        sentence.append(line)
        if line.morphemes in [["."], ["!"], ["?"]]:
            yield sentence
            sentence = []

    if len(sentence) != 0:
        yield sentence


def flatten_with_separator(list_of_lists: list[list], separator) -> list:
    return [item for inner_list in list_of_lists for item in inner_list + [separator]]

def main(raw_data_path: Path, processed_data_path: Path, seed=1):
    random.seed(seed)

    # assumes that the directory structure is as follows:
    # raw_sadilar/  (= raw_data_path)
    #   TEST/
    #     SADII.{lang}.*
    #   TRAIN/
    #     SADII.{lang}.*
    dataset_split = ['TEST', 'TRAIN']
    out_name_format = "{0}_{1}.tsv"

    for output_dir in ['test', 'train', 'dev']:
        shutil.rmtree(processed_data_path / output_dir, ignore_errors=True)
        os.makedirs(processed_data_path / output_dir)

    # outputs the files in the following format:
    # word{tab}canonical_segmentation{tab}morphological_parse
    for dataset_split in dataset_split:
        for in_file in os.listdir(raw_data_path / dataset_split):
            # Skip the English files
            if 'EN' in in_file or not in_file.endswith('.txt'):
                continue

            in_file = raw_data_path / dataset_split / in_file

            # Get the language of the file out of the SADII.{lang}.Morph_Lemma_POS... format
            lang = in_file.name.split('.')[1].lower()

            # Read the lines of the file and remove the lines with the <LINE#> tag
            lines = read_lines_raw(in_file)
            lines = [ParsedLine(line) for line in lines if '<LINE#' not in line]
            sentences = list(split_sentences(lines))

            if dataset_split == 'TEST':
                # Write the formatted lines to a new file
                out_file = out_name_format.format(lang, "test")
                write_sentences(processed_data_path / "test" / out_file, sentences)
            else:
                # Split training set into 90% train / 10% dev
                random.shuffle(sentences)

                train = sentences[len(sentences) // 10:]
                dev = sentences[:len(sentences) // 10]

                train_file = out_name_format.format(lang, "train")
                dev_file = out_name_format.format(lang, "dev")

                write_sentences(processed_data_path / "train" / train_file, train)
                write_sentences(processed_data_path / "dev" / dev_file, dev)


if __name__ == '__main__':
    main(Path(sys.argv[1]), Path(sys.argv[2]))
