import math
import pprint
from pathlib import Path
from typing import Callable, Any

import torch
import tqdm
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, BatchSampler, SequentialSampler

from annotated_corpus_dataset import ParsingPipelineDataset, SEQ_PAD_IX, \
    EncodedParsingDatasetBatch, split_words, EncodedJointParsingDatasetBatch
from annotated_corpus_dataset.segmentation_tokenizers import CharacterInputTokenizerBuilder, \
    CharacterOutputTokenizerBuilder
from evaluation_utils.aligned_set_multiset import eval_model_aligned_multiset
from from_scratch_tagger.encapsulated_model import load_model, predict_tags_for_words_batched
from transformer_segmenter.dataset import TransformerDataset, TransformerSegmentationDataset, \
    TransformerJointEndToEndParsingDataset, TransformerDatasetBatch
from transformer_segmenter.decoding_strategies import SegmenterModel
from ..model import Seq2Seq


class TransformerSegmenterBeamSearch(SegmenterModel):
    def __init__(self, config, train_dataset: TransformerDataset, device,
                 combine_probabilities: Callable[[list[float]], float] = lambda probs: sum(probs) / len(probs),
                 max_len=50, best_k=5):
        super().__init__(train_dataset.vocabulary)
        self.best_k = best_k
        self.max_len = max_len
        self.device: torch.device = device
        self.train_dataset = train_dataset
        self.combine_probabilities = combine_probabilities
        self.model = Seq2Seq.from_config(self.vocab, config, device)

    # TODO max_len
    # TODO sentences
    def segment_words(self, words: Tensor, max_len=50) -> list[list[str]]:
        segmentations = self.beam_search_sentences_batched_with_probs(words)
        return [sentence[0][1][0] for sentence in segmentations]

    def beam_search_words_batched_without_probs(self, words: Tensor) -> list[list[str]]:
        segmentations = self.beam_search_sentences_batched_with_probs(words)
        return [[seg[0] for _score, seg in sentence] for sentence in segmentations]

    def beam_search_sentences_batched_with_probs(self, sentences: list[list[str]] | Tensor) -> list[list[tuple[float, list[list[str]]]]]:
        if isinstance(sentences, list):
            src_tensor = self.vocab.encode_sentences_batched(sentences, self.device)
        else:
            src_tensor = sentences

        batch_size = src_tensor.size(dim=0)

        src_mask = self.model.make_src_mask(src_tensor)

        with torch.no_grad():
            src_encoded = self.model.encoder(src_tensor, src_mask)

        final_segmentations = [[] for _ in range(batch_size)]

        in_progress_branches = [[([1.0], [self.vocab.output_start_token_ix])] for _ in range(batch_size)]

        while any(len(word_branches) != 0 for word_branches in in_progress_branches):
            # Predict the `k` best next tokens for each branch under consideration
            # We will be left with `k` new branches for each existing branch
            new_branches = self._generate_next_tokens(in_progress_branches, src_encoded, src_mask)

            # Take top `best_k` branches and use those as either the final segmentations or new in-progress branches
            all_branches = [
                sorted(
                    new_branches[i] + final_segmentations[i],
                    reverse=True,
                    key=lambda probs_and_branch: self.combine_probabilities(probs_and_branch[0])
                )[:self.best_k]
                    for i in range(batch_size)
            ]

            final_segmentations = [[] for _ in range(batch_size)]
            in_progress_branches = [[] for _ in range(batch_size)]
            for i in range(batch_size):
                for (probs, branch) in all_branches[i]:
                    if len(branch) == self.max_len or branch[-1] == self.vocab.output_end_token_ix:
                        final_segmentations[i].append((probs, branch))
                    else:
                        in_progress_branches[i].append((probs, branch))

        # Decode & return sorted in descending order of probability
        return [
            sorted(
                ((self.combine_probabilities(probs), self.vocab.decode_output_sentence(target_indices)) for probs, target_indices in word),
                key=lambda prob_and_branch: prob_and_branch[0],
                reverse=True,
            )
            for word in final_segmentations
        ]

    def _generate_next_tokens(
            self,
            in_progress_branches: list[list[tuple[list[float], list[int]]]],
            srcs_encoded: list[Tensor],
            srcs_masks: list[Tensor]
    ) -> list[list[tuple[list[float], list[int]]]]:
        target_tensor_list, src_encoded_list, src_mask_list = [], [], []

        for branches, encoded, mask in zip(in_progress_branches, srcs_encoded, srcs_masks):
            for _prob, branch in branches:
                target_tensor_list.append(torch.tensor(branch, device=self.device))
                src_encoded_list.append(encoded)
                src_mask_list.append(mask)

        target_tensor = pad_sequence(
            [torch.tensor(branch, device=self.device) for word in in_progress_branches for _prob, branch in word],
            batch_first=True,
            padding_value=self.vocab.output_pad_ix,
        )

        src_encoded = pad_sequence(src_encoded_list, batch_first=True, padding_value=self.vocab.input_pad_ix)
        src_mask = pad_sequence(src_mask_list, batch_first=True, padding_value=False)
        target_mask = self.model.make_trg_mask(target_tensor)

        with torch.no_grad():
            output, attention = self.model.decoder(target_tensor, src_encoded, target_mask, src_mask)

        # Softmax before topk so that probabilities aren't inflated (since softmax ensures sum(p) = 1)
        probabilities = torch.nn.functional.softmax(output, dim=2)[:, -1, :]

        # This returns the best `self.best_k` tokens for _each_ branch: Each item of top_k.indices/top_k.values will
        # contain `k` indices/values, which are for the best `k` tokens per branch.
        top_k = torch.topk(probabilities, k=self.best_k, dim=1)

        new_branches = [[] for _ in range(len(srcs_encoded))]
        input_word_idx, branch_idx = 0, 0

        idx_in_batch = 0
        while idx_in_batch < probabilities.size(dim=0):
            # We have finished all branches for this input word, but have more to do - move on to the next one
            if branch_idx == len(in_progress_branches[input_word_idx]):
                input_word_idx += 1
                branch_idx = 0

                # Skip this word as it is empty (completely finished)
                if len(in_progress_branches[input_word_idx]) == 0:
                    input_word_idx += 1
                    continue

            best_tokens, best_tokens_probs = top_k.indices[idx_in_batch], top_k.values[idx_in_batch]
            prev_probs, prev_tokens = in_progress_branches[input_word_idx][branch_idx]

            for token, prob in zip(best_tokens, best_tokens_probs):
                new_branches[input_word_idx].append((prev_probs + [prob.item()], prev_tokens + [token.item()]))

            # Move to next branch of this input word
            branch_idx += 1
            idx_in_batch += 1

        for i, (old, new) in enumerate(zip(in_progress_branches, new_branches)):
            assert len(old) * self.best_k == len(new), f"i {i}, old {len(old)}, new {len(new)}"

        return new_branches

    def _convert_indices_to_morphemes(self, target_indices):
        joined = "".join([self.vocab.ix_to_output_subword[i] for i in target_indices[1:] if i != self.vocab.output_end_token_ix])
        joined = joined.split("-")
        return joined

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))


def beam_search_pipeline():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    portions = ParsingPipelineDataset.load_data(
        "ZU",
        Path("data/processed"),
        CharacterInputTokenizerBuilder(),
        CharacterOutputTokenizerBuilder(),
        split_words,
        device,
    )

    cfg = {
        'hidden_dim_per_head': 20, 'layers': 5, 'heads': 4, 'encoder_pf_dim': 1260,
        'decoder_pf_head': 1918, 'encoder_dropout': 0.0, 'decoder_dropout': 0.0, 'lr': 0.0008,
        'gradient_clip': 1.0, 'batch_size': 64, 'valid_batch_size': 512, 'max_epochs': 150
    }

    train_dataset, valid_dataset = portions["train"], portions["dev"]
    model = TransformerSegmenterBeamSearch(
        cfg,
        TransformerSegmentationDataset(train_dataset),
        device,
        combine_probabilities=lambda probs: math.prod(probs)
    )

    print(len(train_dataset.input_subword_to_ix))

    vocab = model.vocab
    def collate(batch: list[Any]) -> Any:
        dev = batch[0].raw_word.device
        input_start = torch.tensor([vocab.input_start_token_ix], device=dev)
        input_end = torch.tensor([vocab.input_end_token_ix], device=dev)
        output_start = torch.tensor([vocab.output_start_token_ix], device=dev)
        output_end = torch.tensor([vocab.output_end_token_ix], device=dev)

        raw_words = pad_sequence([torch.cat((input_start, item.raw_word, input_end)) for item in batch],
                                 batch_first=True, padding_value=vocab.output_pad_ix)
        morphemes = pad_sequence([torch.cat((output_start, item.morphemes, output_end)) for item in batch],
                                 batch_first=True, padding_value=vocab.output_pad_ix)
        tags = pad_sequence([item.tags for item in batch], batch_first=True, padding_value=SEQ_PAD_IX)
        return EncodedParsingDatasetBatch(raw_words, morphemes, tags)

    # Anything beyond this and we do not experience a speedup on RTX4060 mobile
    valid_batch_size = 64
    valid_loader = DataLoader(
        valid_dataset,
        batch_sampler=BatchSampler(SequentialSampler(valid_dataset), valid_batch_size, False),
        collate_fn=collate,
    )

    model.load('best_seg.pt')

    tagger = load_model(
        "bilstm-words-morpheme-ZU-final.pt",
        device=device
    )

    greedy_acc = 0

    model.eval()
    tagger.eval()
    with torch.no_grad():
        for best_k in [1, 2, 3, 4, 5, 10]:

            print(f"\n\nk = {best_k}")
            model.best_k = best_k

            tags_correct = 0
            seg_correct = 0
            analysis_correct = 0
            gold_tag_hr = 0
            incorrect = dict()
            total = 0

            for i, batch in enumerate(tqdm.tqdm(valid_loader)):
                batch: EncodedParsingDatasetBatch = batch
                batch_size = batch.morphemes.size(dim=0)

                # TODO sentence
                raw_words = [sentence[0] for sentence in model.vocab.decode_batched_input_sentences(batch.raw_words)]
                true_morphemes = [sentence[0] for sentence in model.vocab.decode_batched_output_sentences(batch.morphemes)]
                true_tags = [[valid_dataset.ix_to_tag[ix] for ix in item.tolist() if ix != SEQ_PAD_IX] for item in batch.tags.unbind(0)]

                # Beam-search all segmentations for this batch
                pred_morphemes = model.beam_search_words_batched_without_probs(batch.raw_words)

                # Flatten all possible segmentations for each word
                # We go from a list of words, which is a list of possible segs, which is a list of morphemes
                # down to just a list of words, which is a list of morphemes (we flatten dim 1)
                pred_morphemes_flat = [seg for word in pred_morphemes for seg in word]

                # Predict all tags for the resulting segmentations _and_ the gold standard segmentation (in one batch)
                batched_tags = predict_tags_for_words_batched(
                    tagger,
                    pred_morphemes_flat + true_morphemes
                )

                pred_analyses = [[] for _ in range(batch_size)]
                pred_tags = [[] for _ in range(batch_size)]

                input_word_idx, branch_idx = 0, 0
                for idx_in_batch in range(len(batched_tags)):
                    tags = batched_tags[idx_in_batch]
                    segmentation = pred_morphemes[input_word_idx][branch_idx]

                    pred_analyses[input_word_idx].append((segmentation, tags))
                    pred_tags[input_word_idx].append(tags)

                    branch_idx += 1

                    # We have finished all branches for this input word, but have more to do - move on to the next one
                    if branch_idx == len(pred_morphemes[input_word_idx]):
                        input_word_idx += 1
                        branch_idx = 0

                    # Done - everything else is the true morphemes
                    if input_word_idx >= batch_size:
                        break

                true_morphemes_pred_tags = batched_tags[-len(true_morphemes):]
                for word in range(batch_size):
                    total += 1
                    if true_tags[word] == true_morphemes_pred_tags[word]:
                        gold_tag_hr += 1

                    # print(true_morphemes[word], true_tags[word], pred_analyses[word])
                    if (true_morphemes[word], true_tags[word]) in pred_analyses[word]:
                        analysis_correct += 1
                    else:
                        if true_morphemes[word] in pred_morphemes[word]:
                            seg_correct += 1

                        if true_tags[word] in pred_tags[word]:
                            tags_correct += 1

                        incorrect[i * batch_size + word] = {"word": raw_words[word][0], "true": (true_morphemes[word], true_tags[word]), "tried": pred_analyses[word]}

            # incorrect_with_punc = [i for i in incorrect.values() if not i["word"].isalpha()]
            # print(f"{len(incorrect_with_punc)} incorrect examples with numbers or punctuation out of"
            #       f" {len(incorrect)} errors total")
            # print(f"If these errors were fixed, then analysis lattice coverage would be "
            #       f"{(analysis_correct + len(incorrect_with_punc)) / len(valid_dataset) * 100:.2f}")
            print(f"Gold morphemes tag hitrate: {gold_tag_hr/len(valid_dataset) *100:.2f}")
            print(
                f"{(seg_correct + analysis_correct) / len(valid_dataset) * 100:.2f}% segmentation lattice coverage (seg only)"
            )
            print(f"{(tags_correct + analysis_correct) / len(valid_dataset) * 100:.2f}% tag lattice coverage (tag only)")
            print(f"{analysis_correct / len(valid_dataset) * 100:.2f}% analysis lattice coverage (seg + parse)")

            if best_k == 1:
                greedy_acc = analysis_correct / len(valid_dataset) * 100

            print(f"Analysis lattice HR@k=1 on the greedy best choice of segmentation: {greedy_acc:.2f}%")

def beam_search_joint():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    portions = TransformerJointEndToEndParsingDataset.load_data(
        "zu",
        Path("data/processed"),
        CharacterInputTokenizerBuilder(),
        CharacterOutputTokenizerBuilder(),
        split_words,
        device,
    )

    cfg = {'hidden_dim_per_head': 45, 'layers': 3, 'heads': 6, 'encoder_pf_dim': 1277, 'decoder_pf_head': 1953, 'encoder_dropout': 0.20901048079735057, 'decoder_dropout': 0.23025335953340642, 'lr': 0.0008, 'gradient_clip': 2.680819135276812, 'batch_size': 64, 'valid_batch_size': 512, 'max_epochs': 150}

    train_dataset, valid_dataset = portions["train"], portions["dev"]
    model = TransformerSegmenterBeamSearch(
        cfg,
        train_dataset,
        device,
        combine_probabilities=lambda probs: math.prod(probs)
    )

    print(len(train_dataset.vocabulary.input_subword_to_ix))


    # Anything beyond this and we do not experience a speedup on RTX4060 mobile
    valid_batch_size = 64
    valid_loader = DataLoader(
        valid_dataset,
        batch_sampler=BatchSampler(SequentialSampler(valid_dataset), valid_batch_size, False),
        collate_fn=train_dataset.collate,
    )

    model.load('best-so-far-parse.pt')

    greedy_acc = 0

    model.eval()
    with torch.no_grad():
        for best_k in [1, 2, 3, 4, 5, 10]:
            print(f"\n\nk = {best_k}")
            model.best_k = best_k

            tags_correct = 0
            seg_correct = 0
            analysis_correct = 0
            gold_tag_hr = 0
            incorrect = dict()
            total = 0

            for i, batch in enumerate(tqdm.tqdm(valid_loader)):
                batch: TransformerDatasetBatch = batch
                batch_size = batch.target.size(dim=0)

                # TODO sentence
                raw_words = [sentence[0] for sentence in model.vocab.decode_batched_input_sentences(batch.source)]
                true_analysis = [sentence[0] for sentence in model.vocab.decode_batched_output_sentences(batch.target)]

                # Beam-search all analyses for this batch
                pred_analyses = model.beam_search_words_batched_without_probs(batch.source)

                all_true_tags, all_pred_tags = [], []

                for word in range(batch_size):
                    true_morphemes = [morpheme.split("[")[0] for morpheme in true_analysis[word]]
                    true_tags = [morpheme.split("[")[-1].removesuffix("]") for morpheme in true_analysis[word]]

                    pred_morphemes = [[morpheme.split("[")[0] for morpheme in analysis] for analysis in pred_analyses[word]]
                    pred_tags = [[morpheme.split("[")[-1].removesuffix("]") for morpheme in analysis] for analysis in
                                 pred_analyses[word]]

                    total += 1

                    if best_k == 1:
                        all_true_tags.append(true_tags)
                        all_pred_tags.extend(pred_tags)

                    if true_analysis[word] in pred_analyses[word]:
                        analysis_correct += 1
                    else:
                        if true_morphemes in pred_morphemes:
                            seg_correct += 1

                        if true_tags in pred_tags:
                            tags_correct += 1

                        incorrect[i * batch_size + word] = {"word": raw_words[word][0], "true": (true_analysis[word]), "tried": pred_analyses[word]}

            if best_k == 1:
                micro, macro = eval_model_aligned_multiset(all_true_tags, all_pred_tags)
                print(f"Tag F1 micro: {micro*100:.2f}. F1 macro: {macro*100:.2f}")


            # incorrect_with_punc = [i for i in incorrect.values() if not i["word"].isalpha()]
            # print(f"{len(incorrect_with_punc)} incorrect examples with numbers or punctuation out of"
            #       f" {len(incorrect)} errors total")
            # print(f"If these errors were fixed, then analysis lattice coverage would be "
            #       f"{(analysis_correct + len(incorrect_with_punc)) / len(valid_dataset) * 100:.2f}")
            print(f"Gold morphemes tag hitrate: {gold_tag_hr/len(valid_dataset) *100:.2f}")
            print(
                f"{(seg_correct + analysis_correct) / len(valid_dataset) * 100:.2f}% segmentation lattice coverage (seg only)"
            )
            print(f"{(tags_correct + analysis_correct) / len(valid_dataset) * 100:.2f}% tag lattice coverage (tag only)")
            print(f"{analysis_correct / len(valid_dataset) * 100:.2f}% analysis lattice coverage (seg + parse)")

            if best_k == 1:
                greedy_acc = analysis_correct / len(valid_dataset) * 100

            print(f"Analysis lattice HR@k=1 on the greedy best choice of segmentation: {greedy_acc:.2f}%")


if __name__ == "__main__":
    # beam_search_pipeline()
    beam_search_joint()
