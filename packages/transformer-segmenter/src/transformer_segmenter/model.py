import csv
import glob
import math
import os
import pickle
import re
import shutil
import tempfile
import time
from enum import Enum
from pathlib import Path
from typing import Optional

import ray
import torch
import torch.nn as nn
import tqdm
from ray.tune import Checkpoint, get_checkpoint, CheckpointConfig
from ray import tune
from ray.tune.execution.tune_controller import TuneController
from ray.tune.experiment import Trial
from ray.tune.schedulers import FIFOScheduler
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.sample import Integer
from ray.tune.stopper import FunctionStopper
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, BatchSampler, RandomSampler, SequentialSampler

from annotated_corpus_dataset import split_words
from annotated_corpus_dataset.segmentation_tokenizers import CharacterInputTokenizerBuilder, \
    CharacterOutputTokenizerBuilder
from evaluation_utils.aligned_set_multiset import eval_model_aligned_multiset, eval_model_aligned_set
from evaluation_utils.maximal_alignment import align_seqs
from transformer_segmenter.dataset import TransformerSegmentationDataset, TransformerDatasetBatch, TransformerDataset, \
    TransformerJointEndToEndParsingDataset
from transformer_segmenter.vocabulary import Vocabulary

MAX_LENGTH = 4192

# TODO https://stackoverflow.com/a/77445896
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = MAX_LENGTH):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        return self.pe[:x.size(0)]

class Encoder(nn.Module):
    """
    Encoder block made up of the encoder layers: positional embedding, multi-head attention, feed foward and dropout
    """

    def __init__(self,
                 input_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 device,
                 max_length=MAX_LENGTH):
        super().__init__()

        self.device: torch.device = device

        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim) # TODO
        # self.pos_embedding = PositionalEncoding(hid_dim, max_length)

        self.layers = nn.ModuleList([EncoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.tensor([hid_dim], device=device))

    def forward(self, src, src_mask):
        batch_size = src.shape[0]
        src_len = src.shape[1]

        pos = torch.arange(0, src_len, device=self.device).unsqueeze(0).repeat(batch_size, 1)

        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))

        for layer in self.layers:
            src = layer(src, src_mask)

        return src


class EncoderLayer(nn.Module):
    """Encoder layer to make up encoder block, specifies the hidden dimension, number of heads etc..."""

    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 device):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        _src, _ = self.self_attention(src, src, src, src_mask)

        src = self.self_attn_layer_norm(src + self.dropout(_src))

        _src = self.positionwise_feedforward(src)

        src = self.ff_layer_norm(src + self.dropout(_src))

        return src


class MultiHeadAttentionLayer(nn.Module):
    """Multi-Head Attention block, fits into the encoder and decoder"""

    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.tensor([self.head_dim], device=device))

    def forward(self, query, key, value, mask: torch.Tensor = None):
        batch_size = query.shape[0]

        q = self.fc_q(query)
        k = self.fc_k(key)
        v = self.fc_v(value)

        q = q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        energy = torch.matmul(q, k.permute(0, 1, 3, 2)) / self.scale

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim=-1)

        x = torch.matmul(self.dropout(attention), v)

        x = x.permute(0, 2, 1, 3).contiguous()

        x = x.view(batch_size, -1, self.hid_dim)

        x = self.fc_o(x)

        return x, attention


class PositionwiseFeedforwardLayer(nn.Module):
    """Feed Forward network to fit into encoder and decoder blocks"""

    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(torch.relu(self.fc_1(x)))

        x = self.fc_2(x)

        return x


class Decoder(nn.Module):
    """
    Decoder block made up of the decoder layers: positional embedding, (masked) multi-head attention, feed foward
    and dropout
    """

    def __init__(self,
                 output_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 device,
                 max_length=MAX_LENGTH):
        super().__init__()

        self.device: torch.device = device

        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)  # TODO
        # self.pos_embedding = PositionalEncoding(hid_dim, max_length)

        self.layers = nn.ModuleList([DecoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)])

        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.tensor([hid_dim], device=device))

    def forward(self, trg: torch.Tensor, enc_src, trg_mask, src_mask):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        pos = torch.arange(0, trg_len, device=self.device).unsqueeze(0).repeat(batch_size, 1)
        tok_embedding = self.tok_embedding(trg)
        pos_embedding = self.pos_embedding(pos)
        trg = self.dropout((tok_embedding * self.scale) + pos_embedding)

        attention = None
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)

        output = self.fc_out(trg)

        return output, attention


class DecoderLayer(nn.Module):
    """Decoder layer to make up decoder block, specifies the hidden dimension, number of heads etc.."""

    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 device):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
        # TODO pointer gen all happens in the DECODER - leave the encoder completely unchanged
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))

        # TODO no output vocab anymore really

        # TODO compute on the attention in ptrgen
        _trg = self.positionwise_feedforward(trg)
        trg = self.ff_layer_norm(trg + self.dropout(_trg))
        return trg, attention



class Seq2Seq(nn.Module):
    """Entire model tied together with the encoder and decoder"""

    def __init__(self, encoder, decoder, vocab: Vocabulary, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vocab = vocab
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.device: torch.device = device

    DEFAULT_CONFIG = {
        "hidden_dim": 256,
        "encoder_layers": 3,
        "decoder_layers": 3,
        "encoder_heads": 8,
        "decoder_heads": 8,
        "encoder_pf_dim": 512,
        "decoder_pf_head": 512,
        "encoder_dropout": 0.1,
        "decoder_dropout": 0.1,
        "lr": 0.0005,
        "max_epochs": 150,
        "batch_size": 64,
        "valid_batch_size": 512,
        "gradient_clip": 1,
    }

    @staticmethod
    def from_config(vocab: Vocabulary, config=None, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        config = config or Seq2Seq.DEFAULT_CONFIG

        vocab = vocab

        input_dim = len(vocab.input_subword_to_ix) + 2
        output_dim = len(vocab.output_subword_to_ix) + 2

        enc_layers = config.get("encoder_layers") or config["layers"]
        dec_layers = config.get("decoder_layers") or config["layers"]

        enc_heads = config.get("encoder_heads") or config["heads"]
        dec_heads = config.get("decoder_heads") or config["heads"]

        enc_hidden_dim = config.get("hidden_dim") or config["hidden_dim_per_head"] * enc_heads
        dec_hidden_dim = config.get("hidden_dim") or config["hidden_dim_per_head"] * dec_heads

        enc = Encoder(input_dim,
                      enc_hidden_dim,
                      enc_layers,
                      enc_heads,
                      config["encoder_pf_dim"],
                      config["encoder_dropout"],
                      device)

        dec = Decoder(output_dim,
                      dec_hidden_dim,
                      dec_layers,
                      dec_heads,
                      config["decoder_pf_head"],
                      config["decoder_dropout"],
                      device)

        return Seq2Seq(enc, dec, vocab, device)

    def make_src_mask(self, src: torch.Tensor):
        src_mask = (src != self.vocab.input_pad_ix).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg: torch.Tensor):
        trg_pad_mask = (trg != self.vocab.output_pad_ix).unsqueeze(1).unsqueeze(2)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask

    def forward(self, src, trg):
        """
        src = the source word
        trg = current predicted tokens for the segmentation
        """
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)

        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output, attention

    def segment_sentences(self, sentences: list[list[str]] | torch.Tensor, max_len=50) -> list[list[list[str]]]:
        assert not self.training and not torch.is_grad_enabled()

        if isinstance(sentences, torch.Tensor):
            src_tensor = sentences
        elif isinstance(sentences, list):
            src_tensor = self.vocab.encode_sentences_batched(sentences, self.device)
        else:
            raise RuntimeError("Invalid type; expected list[list[str]] or tensor (shape B x W)")

        batch_size = src_tensor.size(dim=0)

        src_mask = self.make_src_mask(src_tensor)

        enc_src = self.encoder(src_tensor, src_mask)

        trg_tensor = torch.stack([torch.tensor([self.vocab.output_start_token_ix], device=self.device)] * batch_size, dim=0)

        # Track which words are still busy generating, so we can exit early if needed
        end_of_sequence = torch.tensor(self.vocab.output_end_token_ix)
        outputs_still_generating = torch.tensor([True for _ in sentences], device=self.device)

        for i in range(max_len):
            trg_mask = self.make_trg_mask(trg_tensor)

            output, attention = self.decoder(trg_tensor, enc_src, trg_mask, src_mask)

            pred_tokens = output.argmax(2)[:, -1]
            trg_tensor = torch.cat((trg_tensor, pred_tokens.reshape((pred_tokens.shape[0], 1))), dim=1)

            # Do (outputs_still_generating & !finished_this_iteration) in place to update which words are still generating
            finished_this_iteration = torch.not_equal(pred_tokens, end_of_sequence)
            outputs_still_generating.logical_and_(finished_this_iteration)

            # Check if all words are done generating
            if torch.all(torch.logical_not(outputs_still_generating)):
                break

        return self.vocab.decode_batched_output_sentences(trg_tensor)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_model(model, iterator, criterion):
    """Function to evaluate the model inbetween each epoch"""

    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.source
            trg = batch.target

            output, _ = model(src, trg[:, :-1])
            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)
            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


class SequenceAlignment(Enum):
    MAXIMAL_EQUALITY = 1
    ALIGNED_SET = 2
    ALIGNED_MULTISET = 3
    PADDING_END = 4

def f1_scores(valid, model, alignment, quiet=False):
    pad = '__<PAD>__'

    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        all_targets, all_preds = [], []
        for batch in tqdm.tqdm(valid, disable=quiet):
            batch: TransformerDatasetBatch = batch
            predictions = model.segment_sentences(batch.source)
            targets = model.vocab.decode_batched_output_sentences(batch.target)

            for prediction, target in zip(predictions, targets):
                prediction = [tok for pred in prediction for tok in pred]
                target = [tok for targ in target for tok in targ]

                # TODO a bit brittle
                prediction = [tok.removesuffix("]") for pred in prediction for tok in pred.split("[")]
                target = [tok.removesuffix("]") for pred in target for tok in pred.split("[")]

                assert pad not in prediction and pad not in target

                total += 1
                if target == prediction:
                    correct += 1

                if alignment == SequenceAlignment.MAXIMAL_EQUALITY:
                    prediction, target = align_seqs(prediction, target)
                elif alignment == SequenceAlignment.PADDING_END:
                    desired_len = max(len(prediction), len(target))
                    prediction += [pad] * (desired_len - len(prediction))
                    target += [pad] * (desired_len - len(target))
                elif alignment in [SequenceAlignment.ALIGNED_SET, SequenceAlignment.ALIGNED_MULTISET]:
                    prediction, target = [prediction], [target]
                else:
                    raise ValueError("Invalid sequence alignment", alignment)

                all_targets.extend(target)
                all_preds.extend(prediction)

        if alignment in [SequenceAlignment.PADDING_END, SequenceAlignment.MAXIMAL_EQUALITY]:
            micro = f1_score(all_targets, all_preds, zero_division=0.0, average='micro')
            macro = f1_score(all_targets, all_preds, zero_division=0.0, average='macro')
        elif alignment == SequenceAlignment.ALIGNED_SET:
            micro, macro = eval_model_aligned_set(all_targets, all_preds)
        elif alignment == SequenceAlignment.ALIGNED_MULTISET:
            micro, macro = eval_model_aligned_multiset(all_targets, all_preds)
        else:
            raise ValueError("Invalid sequence alignment", alignment)

        hr_at_1 = correct / total  # Hit-rate at 1: % of _perfect_ segmentations with one try
    return micro, macro, hr_at_1


def train_epoch(model, iterator: DataLoader, optimizer, criterion, clip, quiet=False):
    """Function to train the model"""
    model.train()

    epoch_loss = 0

    for i, batch in enumerate(tqdm.tqdm(iterator, disable=quiet)):
        # Torch doesn't properly propagate the generic from T to the underlying iterator type, so cast is needed
        batch: TransformerDatasetBatch = batch
        src = batch.source
        trg = batch.target

        optimizer.zero_grad()
        output, _ = model(src, trg[:, :-1])

        # Flatten output + target batches for loss function
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)

        trg = trg[:, 1:].contiguous().view(-1)

        loss = criterion(output, trg)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


def train(model, cfg, name: str, train_dataset: TransformerDataset, valid_dataset: TransformerDataset, device: torch.device, evaluation_alignment: SequenceAlignment=SequenceAlignment.PADDING_END, max_params=100_000_000, max_loss=10, use_ray=False):
    model.apply(initialize_weights)

    train_dataset = train_dataset.to(device)
    valid_dataset = valid_dataset.to(device)
    model = model.to(device)
    vocab = train_dataset.vocabulary

    # Function to compare model size with hyper-parameter changes
    print(f"Model: {name}")
    print(f'The model has {count_parameters(model):,} trainable parameters')

    # Exit early - we can't train this locally
    if count_parameters(model) > max_params and use_ray:
        print("Exiting early... model too big!")
        checkpoint_data = {
            "epoch": 0,
            "loss": max_loss,
        }

        with tempfile.TemporaryDirectory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "wb") as fp:
                pickle.dump(checkpoint_data, fp)

            checkpoint = Checkpoint.from_directory(checkpoint_dir)
            ray.tune.report(
                {"loss": max_loss},
                checkpoint=checkpoint,
            )

    # Specify learning rate and optimisation function
    lr = cfg["lr"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg["weight_decay"])
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.output_pad_ix)
    n_epochs = cfg["max_epochs"]
    gradient_clip = cfg["gradient_clip"]
    batch_size = cfg["batch_size"]
    valid_batch_size = cfg["valid_batch_size"]
    start_epoch = 0

    best_valid_loss = float('inf')
    best_valid_loss_epoch = 0


    if use_ray:
        checkpoint = get_checkpoint()
        if checkpoint:
            with checkpoint.as_directory() as checkpoint_dir:
                data_path = Path(checkpoint_dir) / "data.pkl"
                with open(data_path, "rb") as fp:
                    checkpoint_state = pickle.load(fp)
                start_epoch = checkpoint_state["epoch"]
                best_valid_loss = checkpoint_state["best_valid_loss"]
                best_valid_loss_epoch = checkpoint_state["best_epoch"]
                model.load_state_dict(checkpoint_state["model_state_dict"])
                optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=BatchSampler(RandomSampler(train_dataset), batch_size, False),
        collate_fn=train_dataset.collate,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_sampler=BatchSampler(SequentialSampler(valid_dataset), valid_batch_size, False),
        collate_fn=valid_dataset.collate,
    )

    # Training loop for N_Epochs
    for epoch in tqdm.tqdm(range(start_epoch, n_epochs)):
        start_time = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, gradient_clip, quiet=use_ray)
        valid_loss = evaluate_model(model, valid_loader, criterion)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_valid_loss_epoch = epoch

            if not use_ray:
                torch.save(model.state_dict(), f'{name}.pt')

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

        # if epoch % 10 == 0 or True: # TODO
        #     micro, macro, hr_at_1 = f1_scores(valid_loader, model, evaluation_alignment, quiet=use_ray)
        #     print(f'Micro F1: {micro:.6f}. Macro F1: {macro:.6f}. HR@1: {hr_at_1 * 100:.6f}%')

        if use_ray:
            checkpoint_data = {
                "epoch": epoch,
                "loss": valid_loss,
                "best_epoch": best_valid_loss_epoch,
                "best_valid_loss": best_valid_loss,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }

            with tempfile.TemporaryDirectory() as checkpoint_dir:
                data_path = Path(checkpoint_dir) / "data.pkl"
                with open(data_path, "wb") as fp:
                    pickle.dump(checkpoint_data, fp)

                checkpoint = Checkpoint.from_directory(checkpoint_dir)
                ray.tune.report(
                    {"loss": best_valid_loss},
                    checkpoint=checkpoint,
                )


class BadTrialDeleterScheduler(FIFOScheduler):
    def __init__(self, checkpoint_dir, keep_trials=3):
        super().__init__()
        self.finished_trial_performances: list[tuple[Trial, float]] = list()
        self.mode = None
        self.keep_trials = keep_trials
        self.checkpoint_dir = checkpoint_dir

    def set_search_properties(self, metric: Optional[str], mode: Optional[str], **spec):
        # For some reason, not set in the base class (TrialScheduler)
        if mode:
            self.mode = mode

        return super().set_search_properties(metric, mode)

    # TODO does not remove the right ones
    def on_trial_add(self, tune_controller: "TuneController", trial: Trial):
        # Handle resuming
        if trial.is_finished() and trial.last_result:
            self.on_trial_complete(tune_controller, trial, trial.last_result)

    def on_trial_complete(
        self, tune_controller: TuneController, trial: Trial, result: dict
    ):
        trials = self.finished_trial_performances
        trials.append((trial, result[self.metric]))
        trials = sorted(trials, reverse=self.mode == "max", key=lambda trial_and_metric: trial_and_metric[1])
        self.finished_trial_performances, delete = trials[:self.keep_trials], trials[self.keep_trials:]

        for trial, _metric in delete:
            for checkpoint_dir in glob.glob(str(Path(trial.storage.trial_fs_path) / "checkpoint_*")):
                shutil.rmtree(checkpoint_dir)

        return super().on_trial_complete(tune_controller, trial, result)

def tune_model(model_for_config, search_space, name: str, fixed_cfg, train_set: TransformerDataset,
               valid_set: TransformerDataset, cpus=4, hrs=11, max_loss=10):
    """Tune the given model with Ray"""

    ray.init(num_cpus=cpus)

    algo = BayesOptSearch(metric="loss", mode="min")
    algo = ConcurrencyLimiter(algo, max_concurrent=cpus)

    # Move the trainset & validset into shared memory (they are very large)
    train_set, valid_set = ray.put(train_set), ray.put(valid_set)

    int_params = []
    search_space_to_float = dict()

    for param, space in search_space.items():
        if isinstance(space, Integer):
            search_space_to_float[param] = tune.uniform(space.lower, space.upper)
            int_params.append(param)
        else:
            search_space_to_float[param] = space

    def do_train(conf):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        conf = {param : (val if param not in int_params else round(val)) for param, val in conf.items()}
        return train(model_for_config(conf, device), conf, name, ray.get(train_set), ray.get(valid_set), device, use_ray=True, max_loss=max_loss)

    checkpoint_dir = Path(os.environ["TUNING_CHECKPOINT_DIR"]).resolve()

    # Do the hyperparameter tuning
    # TODO custom TrialScheduler to delete bad trials?
    result = tune.run(
        do_train,
        metric="loss",
        name=name,
        resume=True,
        mode="min",
        resources_per_trial={"gpu": 1.0 / cpus} if torch.cuda.is_available() else None,
        config=search_space_to_float,
        num_samples=100,
        time_budget_s=hrs * 60 * 60,
        search_alg=algo,
        checkpoint_config=CheckpointConfig(
            num_to_keep=3,
            checkpoint_score_attribute="loss",
            checkpoint_score_order="min",
        ),
        scheduler=BadTrialDeleterScheduler(checkpoint_dir),
        storage_path=str(checkpoint_dir),
        trial_dirname_creator=lambda trial: trial.trial_id,
        stop=FunctionStopper(lambda trial_id, results: results["loss"] >= max_loss)
    )

    valid_set = ray.get(valid_set)
    valid_loader = DataLoader(
        valid_set,
        batch_sampler=BatchSampler(SequentialSampler(valid_set), fixed_cfg["valid_batch_size"], False),
        collate_fn=valid_set.collate,
    )

    # Print out the epoch with best macro & micro F1 scores
    for metric in ["loss"]:
        best_trial = result.get_best_trial(metric, "min", "all")
        print(f"Best trial by {metric}:")
        print(f" config: {best_trial.config}")
        print(f" val loss: {best_trial.last_result['loss']}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        conf = {param: (val if param not in int_params else round(val)) for param, val in best_trial.config.items()}
        best_model = model_for_config(conf, device)
        best_model = best_model.to(device)
        best_checkpoint = result.get_best_checkpoint(trial=best_trial, metric=metric, mode="min")

        with best_checkpoint.as_directory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "rb") as fp:
                best_checkpoint_data = pickle.load(fp)

            print(f" val loss epoch: {best_checkpoint_data['best_epoch']}")

            best_model.load_state_dict(best_checkpoint_data["model_state_dict"])
            f1_micro, f1_macro, hr_at_1 = f1_scores(valid_loader, best_model, SequenceAlignment.ALIGNED_MULTISET)
            print(f" {name}: Micro F1: {f1_micro}. Macro f1: {f1_macro}. Hit-rate @ 1: {hr_at_1 * 100:.2f}%")
            # print(
            #     f" {name}: Best macro f1: {best_checkpoint_data['best_macro']} at epoch "
            #     f"{best_checkpoint_data['best_epoch']}")


# def segment_and_tag_unseen():
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     seed = 1
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True
#
#     d = dataset.Data()
#     train_data, train_iter, valid_data, valid_iter, test_data, test_iter, src_field, target_field = d.get_iterators()
#
#     # Specify the input, hidden, ouput dimensions. Encoder, Decoder heads and dropout
#     input_dim = len(src_field.vocab)
#     output_dim = len(target_field.vocab)
#     hidden_dim = 256
#     encoder_layers = 3
#     decoder_layers = 3
#     encoder_heads = 8
#     decoder_heads = 8
#     encoder_pf_dim = 512
#     decoder_pf_head = 512
#     encoder_dropout = 0.1
#     decoder_dropout = 0.1
#
#     enc = Encoder(input_dim,
#                   hidden_dim,
#                   encoder_layers,
#                   encoder_heads,
#                   encoder_pf_dim,
#                   encoder_dropout,
#                   device)
#
#     dec = Decoder(output_dim,
#                   hidden_dim,
#                   decoder_layers,
#                   decoder_heads,
#                   decoder_pf_head,
#                   decoder_dropout,
#                   device)
#
#     SRC_PAD_IDX = src_field.vocab.stoi[src_field.pad_token]
#     TRG_PAD_IDX = target_field.vocab.stoi[target_field.pad_token]
#
#     # Initialise model
#     model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)
#
#     # # Load best saved model for evaluation
#     model.load_state_dict(
#         torch.load(
#             'segment_new_zu_no_validset.pt',
#             map_location=torch.device('cpu'),
#             weights_only=True
#         )
#     )
#
#     tagger = load_model(
#         "/home/restioson/PycharmProjects/MORPH_PARSE/bilstm-no-testset-words-morpheme-ZU.pt"
#     )
#
#     unseen = []
#     with open("Corpus_Zulu.txt", "rb") as f:
#         for line_no, line in enumerate(f.read().decode("utf-8", errors="ignore").split("\n")):
#             unseen.extend((line_no + 1, word) for word in line.split())
#
#     with open("corpus_analysed.tsv", "w") as f:
#         writer = csv.writer(f, delimiter="\t")
#         writer.writerow(
#             ["Raw word", "Line in input", "Morphological analysis", "Morphemes", "Morphological tags"]
#         )
#
#         for (line_no, word) in tqdm.tqdm(unseen):
#             morphemes = evaluation.segment_one(src_field, target_field, model, device, word)[0]
#             tags = predict_tags_for_word(tagger, morphemes)
#             analysis = "-".join([f"{morpheme}[{tag}]" for morpheme, tag in zip(morphemes, tags)])
#             writer.writerow([word, str(line_no), analysis, "-".join(morphemes), "-".join(tags)])


def find_class_5_or_11():
    with open("corpus_analysed.tsv", "r") as in_f:
        reader = csv.reader(in_f, delimiter="\t")
        next(reader)

        by_line = []
        cur_line = 0
        for record in reader:
            if (line_no := record[1]) != cur_line:
                by_line.append([])
                cur_line = line_no
            by_line[-1].append(record)

        with open("corpus_filtered_class_11_or_5.tsv", "w") as out_f:
            writer = csv.writer(out_f, delimiter="\t")
            for line in by_line:
                #print(any("11" in (analysis := word[4]) for word in line))
                # print(any(re.match(r"[a-zA-Z]5]", word[4]) for word in line))
                if (any(re.search(r"[a-zA-Z]5(-|$)", word[4]) for word in line)):
                    print("HI")
                #print('='*100)

                if any("11" in word[4] or re.search(r"[a-zA-Z]5(-|$)", word[4]) for word in line):
                    writer.writerow([
                        " ".join(word[0] for word in line),  # Raw words
                        line[0][1],  # Line no
                        " ".join(word[2] for word in line),  # Analysis
                        " ".join(word[3] for word in line),  # Morphemes
                        " ".join(word[4] for word in line),  # Tags
                    ])

def do_train_seg():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    portions = TransformerSegmentationDataset.load_data(
        "zu",
        Path("data/processed"),
        CharacterInputTokenizerBuilder(),
        CharacterOutputTokenizerBuilder(),
        split_words,
        device=device,
    )

    cfg = {
        'hidden_dim_per_head': 20, 'layers': 5, 'heads': 4, 'encoder_pf_dim': 1260,
        'decoder_pf_head': 1918, 'encoder_dropout': 0.0, 'decoder_dropout': 0.0, 'lr': 0.0008,
        'gradient_clip': 1.0, 'batch_size': 32, 'valid_batch_size': 32, 'max_epochs': 150
    }


    train_dataset, valid_dataset = portions["train"], portions["dev"]
    model = Seq2Seq.from_config(train_dataset.vocabulary, cfg, device=device)
    train(model, cfg, "segmenter-zu", train_dataset, valid_dataset, device, SequenceAlignment.ALIGNED_MULTISET)

# TODO too big what to do? - trim size? reduce some things?
    # Trial do_train_0caf1444 started with configuration:
    # ╭────────────────────────────────────────────╮
    # │ Trial do_train_0caf1444 config             │
    # ├────────────────────────────────────────────┤
    # │ batch_size                              64 │
    # │ decoder_dropout                     0.0919 │
    # │ decoder_pf_head                       2019 │
    # │ encoder_dropout                     0.0289 │
    # │ encoder_pf_dim                         805 │
    # │ gradient_clip                      5.25259 │
    # │ heads                                    7 │
    # │ hidden_dim_per_head                    268 │
    # │ layers                                   1 │
    # │ lr                                 0.00078 │
    # │ max_epochs                              30 │
    # │ valid_batch_size                       512 │
    # ╰────────────────────────────────────────────╯


def do_train_parse():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    portions = TransformerJointEndToEndParsingDataset.load_data(
        "ZU",
        Path("data/processed"),
        CharacterInputTokenizerBuilder(),
        CharacterOutputTokenizerBuilder(),
        split_words,
        device=device,
    )

    cfg = {
        **Seq2Seq.DEFAULT_CONFIG,
        # "lr": 0.001,
        # "hidden_dim": 1024,
        # "encoder_heads": 16,
        # "decoder_heads": 16,
        "batch_size": 1,  # Can't go above 4 on 8gb of vram
        "valid_batch_size": 32,
    }

    cfg = {'hidden_dim_per_head': 45, 'layers': 3, 'heads': 6, 'encoder_pf_dim': 1277, 'decoder_pf_head': 1953, 'encoder_dropout': 0.20901048079735057, 'decoder_dropout': 0.23025335953340642, 'lr': 0.0008, 'gradient_clip': 2.680819135276812, 'batch_size': 64, 'valid_batch_size': 512, 'max_epochs': 150}

    # best per bayesopt
    # cfg = {
    #     'hidden_dim_per_head': 56, 'layers': 4, 'heads': 6,
    #  'encoder_pf_dim': 1277, 'decoder_pf_head': 1953, 'encoder_dropout': 0.292797576724562,
    #  'decoder_dropout': 0.149816047538945, 'lr': 0.00048129089438282387, 'gradient_clip': 2.4041677639819286,
    #  'batch_size': 64, 'valid_batch_size': 256, 'max_epochs': 150}


    # TODO where is te tag in the input rel. to the output?
    # TODO segment only at a sentence level?
    # TODO Seq2seq LSTM + att for the joint end-to-end
    # TODO beam search the word-level
    # TODO predict for the last word only token? for each - maybe later
    # TODO use PLM embeddings later - even for segmenter and morphparse taggers (way to encode sent. level context)
    # TODO split the tags? NPrePre14 -> NPrePre, 14
    # ANS : BAYESOPT TODO grid search or bayesopt?

    # TODO ASAP (before moving on to the ptr gen) : hyperparam tuning
    # TODO then ptr gen
        # TODO do exactly as the paper with the masking being one possible tag per seg
        #      but not the embeddings
        # TODO embeddings: each char and tag and sep is its own token

    train_dataset, valid_dataset = portions["train"], portions["dev"]
    model = Seq2Seq.from_config(train_dataset.vocabulary, cfg, device=device)
    train(model, cfg, "best-so-far-parse", train_dataset, valid_dataset, device, SequenceAlignment.ALIGNED_MULTISET)

def do_tune_parse():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    portions = TransformerJointEndToEndParsingDataset.load_data(
        "zu",
        Path("data/processed"),
        CharacterInputTokenizerBuilder(),
        CharacterOutputTokenizerBuilder(),
        split_words,
        device=device,
    )

    # TODO scheduler with patience?

    # TODO _try_ a grid search
    fixed_cfg = {"batch_size": 32, "valid_batch_size": 512, "max_epochs": 1}
    search_space = {
        "hidden_dim_per_head": tune.qrandint(16, 256),
        "layers": tune.qrandint(1, 6),
        "heads": tune.qrandint(4, 16),
        "encoder_pf_dim": tune.qrandint(128, 2048),
        "decoder_pf_head": tune.qrandint(128, 2048),
        "encoder_dropout": tune.uniform(0.0, 0.4),
        "decoder_dropout": tune.uniform(0.0, 0.4),
        "lr": tune.loguniform(1e-6, 1e-3),
        "gradient_clip": tune.uniform(1, 100),
        "weight_decay": tune.uniform(0, 1),
        **fixed_cfg
    }

    train_dataset, valid_dataset = portions["train"], portions["dev"]
    vocab = train_dataset.vocabulary
    # cfg = {
    #     "hidden_dim_per_head": 32, # TODO?
    #     "layers": 4,
    #     "heads": 8, # TODO?
    #     "encoder_pf_dim": 512,  # TODO do these just need to be equal?
    #     "decoder_pf_head": 512,
    #     "encoder_dropout": 0.1, # TODO?
    #     "decoder_dropout": 0.1, # TODO?
    #     "lr": 0.0005, # TODO
    #     "gradient_clip": 2.40417,
    #     **fixed_cfg,
    # }

    cfg = {
        "hidden_dim_per_head": 44,
        "layers": 4,
        "heads": 5,
        "encoder_pf_dim": 1277,
        "decoder_pf_head": 1953,
        "encoder_dropout": 0.29,
        "decoder_dropout": 0.14,
        "lr": 6e-4,
        "gradient_clip": 2.40417,
        **fixed_cfg,
    }

    other = {
        "hidden_dim": 256,  # 220 hidden
        "encoder_layers": 3, # 4
        "decoder_layers": 3,
        "encoder_heads": 8, # 5
        "decoder_heads": 8,
        "encoder_pf_dim": 512,  # 1277
        "decoder_pf_head": 512,  #1953
        "encoder_dropout": 0.1,  # 0.29
        "decoder_dropout": 0.1,  # 0.14
        "lr": 0.0005,   # 0.00120263
        "max_epochs": 150,
        "batch_size": 64,
        "valid_batch_size": 512,
        "gradient_clip": 1,  # 2
    }

    # cfg = other

    # train(Seq2Seq.from_config(vocab, cfg, device), cfg,"testing_seg", train_dataset, valid_dataset, device)
    tune_model(lambda conf, dev: Seq2Seq.from_config(vocab, conf, dev), search_space, "parse_bayesopt2", fixed_cfg, train_dataset, valid_dataset, cpus=int(os.environ.get("RAY_CPUS") or 1), hrs=24)


if __name__ == "__main__":
    # do_train_parse()
    # do_train_seg()
    do_tune_parse()


# Best bayes on tune parse, 30 epoch
# Current best trial: 0613fc6e with loss=0.14888812676072122 and params={'hidden_dim_per_head': 45.18525423955908, 'layers': 2.761549392895106, 'heads': 6.145521235586304, 'encoder_pf_dim': 1277.1666873192985, 'decoder_pf_head': 1953.0961800896232, 'encoder_dropout': 0.20901048079735057, 'decoder_dropout': 0.23025335953340642, 'lr': 0.0008, 'gradient_clip': 2.680819135276812, 'batch_size': 64, 'valid_batch_size': 512, 'max_epochs': 30}
# TODO Why
"""
Trial status: 100 TERMINATED
Current time: 2025-11-12 19:28:30. Total running time: 3hr 26min 28s
Logical resource usage: 0/5 CPUs, 0.25/1 GPUs (0.0/1.0 accelerator_type:L40S)
Current best trial: 21653d2b with loss=0.14585136100649834 and params={'hidden_dim_per_head': np.float64(49.03129023154627), 'layers': np.float64(2.7792713402042297), 'heads': np.float64(4.709749791504686), 'encoder_pf_dim': np.float64(1279.7199593249554), 'decoder_pf_head': np.float64(1953.9742734310378), 'encoder_dropout': np.float64(0.0), 'decoder_dropout': np.float64(0.0), 'lr': np.float64(0.0008), 'gradient_clip': np.float64(1.0), 'batch_size': 64, 'valid_batch_size': 512, 'max_epochs': 30}
╭─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Trial name          status         hidden_dim_per_head     layers      heads     encoder_pf_dim     decoder_pf_head     encoder_dropout     decoder_dropout            lr     gradient_clip     iter     total time (s)        loss │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ do_train_d3594872   TERMINATED                 48.0298    2.07263    4.75616           1278.97             1951.7            0.368171            0            0.0008                2.44136       31           998.481     0.197828 │
│ do_train_cbd71cfe   TERMINATED                 46.9892    1.10065    5.30589           1278.08             1951.47           0.0860662           0            0.0008                1.80872       31           702.477     0.160734 │
│ do_train_d31cd821   TERMINATED                 48.1222    2.011      6.39805           1279.47             1951.7            0                   0            0.0008                1.98509       31           997.715     0.153767 │
│ do_train_0d5b11f5   TERMINATED                 47.9949    2.15532    4.58454           1278.99             1951.61           0.324619            0            0.0008                2.40224       31           995.848     0.178822 │
│ do_train_103c837c   TERMINATED                 47.033     1.33034    4.75503           1278.23             1952.15           0.298787            0            0.0008                3.64438       30           620.347     0.170799 │
│ do_train_c015caee   TERMINATED                 46.2133    2.74427    7.20992           1278.03             1953.15           0.246683            0            0.0008                1.07416       30          1287.07      0.230818 │
│ do_train_6184f750   TERMINATED                 46.9074    1.74477    5.28285           1278.11             1952.72           0.0542376           0            0.0008                4.33394       30           907.732     0.149664 │
│ do_train_91572a5f   TERMINATED                 46.7377    1.99554    5.07171           1278                1952.76           0                   0            0.0008                4.32283       30           891.687     0.155354 │
│ do_train_a05b6bac   TERMINATED                 46.1644    2.43892    5.21179           1278.78             1953.71           0.168104            0            0.0008                2.09635       30           907.769     0.156631 │
│ do_train_34c26445   TERMINATED                 46.1432    2.54272    5.29956           1278.65             1953.71           0.205316            0            0.0008                2.16262       30          1209.29      0.164987 │
│ do_train_24451fac   TERMINATED                 45.9562    2.79325    5.07879           1278.4              1953.77           0.4                 0            0.0008                2.32122       30          1205         0.201523 │
│ do_train_51b6aaed   TERMINATED                 46.1883    2.49942    5.28175           1278.74             1953.66           0                   0            0.0008                2.15635       30           866.419     0.155962 │
│ do_train_6b7ad74a   TERMINATED                 45.8504    3.41791    6.12527           1278.28             1951.89           0.00960962          0            0.0008                2.90459       30          1231.76      0.153842 │
│ do_train_bba63e58   TERMINATED                 45.8984    3.43015    6.12665           1278.21             1951.98           0                   0            0.0008                2.88774       30          1187.88      0.153874 │
│ do_train_28bb783d   TERMINATED                 45.8257    3.38458    6.10805           1278.06             1952.09           0.391064            0            0.0008                2.842         31          1319.14      0.223406 │
│ do_train_f153e232   TERMINATED                 45.7695    3.41904    6.09779           1278                1952.17           0.4                 0            0.0008                2.81671       31          1319.57      0.222495 │
│ do_train_f6128e64   TERMINATED                 47.0431    3.47949    6.03739           1277.29             1953.21           0.165466            0            0.0008                2.57258       31          1294.25      0.158744 │
│ do_train_0613fc6e   TERMINATED                 45.1853    2.76155    6.14552           1277.17             1953.1            0.20901             0.230253     0.0008                2.68082       31          1307.11      0.148888 │
│ do_train_464fc570   TERMINATED                 46.2737    4.52397    5.80612           1277.37             1953.32           0.224754            0.0361812    1.01308e-06           2.33868       30          1952.13      0.823515 │
│ do_train_d4110dfa   TERMINATED                 46.8842    4.34764    5.9929            1277.48             1953.43           0.268274            0.0971871    1e-06                 2.5251        30          1584.6       1.01706  │
│ do_train_3642af3b   TERMINATED                 44.7598    2.65691    6.277             1277.11             1953.36           0                   0.4          1e-06                 2.77849       30          1171.31      1.81332  │
│ do_train_f560751a   TERMINATED                 44.8392    4.96376    6.96665           1278.37             1954.33           0.4                 0.4          0.0008                3.4971         3           249.802    10.0811   │
│ do_train_f4fa63d6   TERMINATED                 45.1596    4.13972    6.21977           1277.59             1953.54           0.229365            0            0.000475704           2.752         30          1490.76      0.170109 │
│ do_train_004c5b00   TERMINATED                 45.1501    4.14759    6.21093           1277.59             1953.54           0.230784            0.0031924    0.00028663            2.74316       30          1546.28      0.174379 │
│ do_train_2fc86297   TERMINATED                 45.145     4.15086    6.20744           1277.59             1953.54           0.231432            0.00470483   0.0002936             2.73967       30          1542.26      0.172241 │
│ do_train_f513247e   TERMINATED                 45.1304    4.16448    6.19291           1277.58             1953.53           0.234107            0.0109745    0.000310293           2.72515       30          1550.4       0.172238 │
│ do_train_4f23de13   TERMINATED                 31.639     5          9.47686            969.147             315.985          0.266366            0.111499     0.000224688           1.71844       31          2008.19      0.168271 │
│ do_train_78d8a3b6   TERMINATED                 48.9694    4.46274    5.86997           1277.42             1953.37           0.290831            0.147849     1.95711e-06           2.4022        31          1656.54      0.789328 │
│ do_train_a6b2e178   TERMINATED                104.227     2.50259    7.08566           1725.6               168.422          0.260648            0.170144     0.000240376           5.49036       31          1493.24      0.150005 │
│ do_train_f8cb7b95   TERMINATED                 44.2372    4.99997    5.29971           1277.14             1953.09           0.397459            0.397307     0.000799971           1.83195       27          1802.94     10.211    │
│ do_train_229c0a77   TERMINATED                511.601     1.40562   10.6549            1710.11             2045.84           0.261918            0.253379     0.000584778           4.40904        1            57.0417   10        │
│ do_train_7a4d14a2   TERMINATED                512         1         16                 1746.22             2048              0                   0            0.0008                1              1            68.0309   10        │
│ do_train_5a102e35   TERMINATED                 16         1          4                 2038.24             2048              0.4                 0            6.14252e-05          10             30           597.86      0.22072  │
│ do_train_dd8ac28a   TERMINATED                 16         1          4                 2038.27             2048              0.4                 0            0.000679124          10             30           587.038     0.163144 │
│ do_train_dea6d445   TERMINATED                 16         1          4                 2038.27             2048              0.4                 0            0.000634833          10             30           573.359     0.166076 │
│ do_train_a6d630ed   TERMINATED                193.799     1.46348    4.88854           1527.69             1693.69           0.282743            0.00220885   0.00069062            7.94143       30           906.49      0.555586 │
│ do_train_b1f3a3d0   TERMINATED                208.784     2.0854     7.90396            504.287            1898              0.035397            0.23916      0.000663161           1.40705       30          1995.02      3.30821  │
│ do_train_752dd297   TERMINATED                 33.0567    4.63728    9.94212            973.093             315.53           0.273693            0.121846     0.000207765           2.09834       30          2016.88      0.16816  │
│ do_train_fe53c08e   TERMINATED                332.229     4.54885   12.7553             725.086             763.324          0.0254233           0.249319     0.0003783             3.92665        1           187.756    10        │
│ do_train_5e264f89   TERMINATED                 44.8095    4.4647     5.87193           1277.42             1953.37           0.292798            0.149816     0.000481291           2.40417       30          1727.59      0.149322 │
│ do_train_44f7a0c1   TERMINATED                505.496     4.08898    4.89461            398.575             667.394          0.217078            0.142701     0.000159774           8.21977        1           132.29     10        │
│ do_train_f17f4f40   TERMINATED                106.969     2.21697    6.1819            1726.29              167.522          0.387964            0.283229     0.00042028            2.91105       30          1190.53      0.171123 │
│ do_train_e6e8a80f   TERMINATED                400.466     4.758     15.635             1177.68              726.485          0.208027            0.265009     0.000715967           2.66369        1           107.161    10        │
│ do_train_72b86861   TERMINATED                486.647     4.86253    4.78062           1294.49             1265.44           0.0185802           0.205694     0.000646909           2.53472        1            58.0683   10        │
│ do_train_742c9d7f   TERMINATED                242.211     4.1407     8.39634            395.828             687.16           0.244741            0.172778     0.000160539           3.6293         1            49.2481   10        │
│ do_train_342c845f   TERMINATED                 48.3406    2.36622    6.2299            1279.82             1951.95           0.147201            0            0.0008                2.36212       30           926.042     0.157784 │
│ do_train_e7f1346e   TERMINATED                 48.5762    1.0226     5.59624           1277.69             1953.88           0.373923            0            0.0008                1.84163       30           629.608     0.183618 │
│ do_train_d93e268e   TERMINATED                 48.2512    1.38554    4.21324           1277.19             1953.85           0                   0            0.0008                1.71566       30           614.572     0.16196  │
│ do_train_ef34a002   TERMINATED                 47.426     1.67201    7.90797           1277.17             1951.31           0                   0            0.0008                3.02225       30           934.585     0.17405  │
│ do_train_21653d2b   TERMINATED                 49.0313    2.77927    4.70975           1279.72             1953.97           0                   0            0.0008                1             30          1191.69      0.145851 │
│ do_train_2a9b6e9f   TERMINATED                 49.0446    2.75563    4.56474           1279.7              1953.99           0                   0            0.0008                1             30          1189.64      0.149047 │
│ do_train_07022c54   TERMINATED                 49.0764    2.70601    4.43179           1279.73             1954.06           0                   0            0.0008                1             30          1206.83      0.156944 │
│ do_train_5b15d4b9   TERMINATED                 48.9381    1.72704    5.47431           1275.89             1951.76           0                   0            0.0008                3.13024       30           879.08      0.152478 │
│ do_train_32a8b49a   TERMINATED                 48.9304    1.75369    5.45059           1275.89             1951.65           0                   0            0.0008                3.15604       30           900.118     0.152277 │
│ do_train_49a47caa   TERMINATED                 47.9498    1.31995    5.66823           1275.27             1952.24           0.4                 0            0.0008                3.24064       30           626.06      0.176052 │
│ do_train_d1d549ea   TERMINATED                 49.2447    2.19355    4.16327           1279                1954.6            0                   0            0.0008                3.9409        30           897.222     0.157699 │
│ do_train_4b0afa0e   TERMINATED                 50.1465    2.23965    4.30109           1278.24             1953.72           0                   0            0.0008                3.95726       30           903.137     0.15158  │
│ do_train_2031aff6   TERMINATED                 49.9423    2.27273    4.23566           1278.44             1954.05           0                   0            0.0008                4.02295       30           901.1       0.151328 │
│ do_train_e2bc5deb   TERMINATED                 49.8257    2.28799    4.12726           1278.68             1954.21           0                   0            0.0008                4.05024       30           901.153     0.152952 │
│ do_train_56557d81   TERMINATED                 48.8301    2.47518    4                 1275.94             1954.08           0.4                 0            0.0008                4.80207       30           924.029     0.177844 │
│ do_train_8da7fda1   TERMINATED                 49.8985    1.89161    7.67177           1278.26             1953.35           0                   0            0.0008                4.04603       30           946.831     0.186997 │
│ do_train_3f5e067c   TERMINATED                 49.976     1.86267    7.7319            1278.28             1953.36           0                   0            0.0008                4.05798       30           957.473     0.212484 │
│ do_train_6165d791   TERMINATED                 50.2462    1.85144    7.80651           1278.34             1953.04           0                   0            0.0008                3.97826       30           944.758     0.182254 │
│ do_train_cb45e629   TERMINATED                 49.3025    2.67511    6.73618           1276.78             1952.17           0.4                 0            0.0008                5.75019       30          1293.75      0.294866 │
│ do_train_0091f5a6   TERMINATED                 49.0742    2.86173    6.37282           1276.72             1951.83           0.4                 0            0.0008                5.9371        30          1249.16      0.233146 │
│ do_train_f5283481   TERMINATED                 49.1739    2.55825    6.20618           1276.65             1951.71           0.4                 0            0.0008                6.2164        30          1237.8       0.238371 │
│ do_train_bd602940   TERMINATED                 49.2487    2.59071    6.14294           1276.78             1951.62           0.4                 0            0.0008                6.23695       30          1249.74      0.282623 │
│ do_train_8f9ffe9e   TERMINATED                 48.0217    2.71534    6.18484           1278.59             1949.46           0                   0            0.0008                4.60973       30          1229.07      0.163169 │
│ do_train_598e61b6   TERMINATED                 47.9166    2.62087    6.23068           1278.69             1949.32           0                   0            0.0008                4.50733       30          1213.45      0.147699 │
│ do_train_759b9112   TERMINATED                 47.8725    2.4303     6.18955           1278.71             1949.27           0                   0            0.0008                4.58364       30           900.615     0.145902 │
│ do_train_579daf24   TERMINATED                 48.9076    1          4                 1278.08             1953.72           0                   0            0.0008                7.19843       30           615.555     0.158773 │
│ do_train_ce868b19   TERMINATED                 49.8117    1          5.67463           1280.59             1952.44           0                   0            0.0008                5.65327       30           628.294     0.164612 │
│ do_train_e5d85966   TERMINATED                 49.6401    1          5.5764            1280.45             1951.87           0                   0            0.0008                5.61316       30           629.452     0.163389 │
│ do_train_dc86dc50   TERMINATED                 49.6372    1          5.56989           1280.5              1951.86           0                   0            0.0008                5.62584       30           630.231     0.162808 │
│ do_train_95d658bb   TERMINATED                 49.738     1          5.63917           1280.65             1952.14           0                   0            0.0008                5.63793       30           631.351     0.158037 │
│ do_train_d472d155   TERMINATED                 52.0134    1          5.54475           1278.37             1954.23           0                   0            0.0008                6.56758       30           634.377     0.160548 │
│ do_train_e99f574c   TERMINATED                 51.7099    1          5.29941           1276.64             1954.54           0                   0            0.0008                6.31698       30           629.981     0.157403 │
│ do_train_a5912244   TERMINATED                 51.6142    1          5.24302           1276.32             1954.66           0                   0            0.0008                6.28879       30           627.759     0.157602 │
│ do_train_58bfd306   TERMINATED                 51.7066    1          5.21513           1276.47             1954.76           0.4                 0            0.0008                6.33176       30           637.595     0.178987 │
│ do_train_85f45c9a   TERMINATED                 49.6878    1          6.00076           1276.28             1956.03           0.4                 0            0.0008                6.34731       30           641.605     0.187974 │
│ do_train_28e62d70   TERMINATED                 49.404     1          5.61504           1277.88             1956.83           0.4                 0            0.0008                6.32885       30           640.917     0.188429 │
│ do_train_893a71d2   TERMINATED                 49.499     1          5.44088           1277.93             1956.96           0.4                 0            0.0008                6.34163       30           630.856     0.179612 │
│ do_train_ceb323d0   TERMINATED                 49.2844    1          5.72403           1277.81             1956.87           0                   0            0.0008                6.3842        30           629.08      0.157645 │
│ do_train_b67dc3a9   TERMINATED                 50.4312    1          4                 1279.12             1957.07           0                   0            0.0008                6.50158       30           577.824     0.161112 │
│ do_train_be650bd3   TERMINATED                 50.4894    3.88142    4.66316           1277.58             1955.21           0                   0            0.0008                7.53082       30          1504.7       2.57755  │
│ do_train_f06b2329   TERMINATED                 50.4732    3.87173    4.6671            1277.54             1955.26           0                   0            0.0008                7.59155       30          1497.69      3.0149   │
│ do_train_8395ab41   TERMINATED                 50.4731    3.87169    4.66714           1277.54             1955.26           0                   0            0.0008                7.59153       30          1496.33      2.86985  │
│ do_train_fdfd1559   TERMINATED                 48.0831    1          8.84386           1280.6              1950.8            0                   0            0.0008                4.73915       30           631.353     0.18372  │
│ do_train_2fddef53   TERMINATED                 50.5137    3.83988    4.54852           1277.64             1955.36           0                   0            0.0008                7.57067       30          1480.55      3.62523  │
│ do_train_3ccaacba   TERMINATED                 47.9039    1          4                 1281.58             1955.37           0                   0            0.0008                6.54845       30           608.711     0.162256 │
│ do_train_50c49020   TERMINATED                 56.411     1.30705   15.8236             543.734            1315.13           0.372699            0.0482776    0.000409996           8.01303       30           792.962     0.185928 │
│ do_train_1984a304   TERMINATED                468.208     4.96866   15.2181            1672.67             1181.95           0.0276394           0.266502     0.000680163           1.13623        1           126.746    10        │
│ do_train_4a571ad7   TERMINATED                343.932     2.38009   10.6187             699.11             1525.98           0.343208            0.326897     0.000180438           8.7065         1            75.1526   10        │
│ do_train_c2c75101   TERMINATED                100.199     4.05334    7.17703            191.135            1018.62           0.216652            0.0274199    0.000716983           5.35298       30          1653.48      6.86121  │
│ do_train_071d7f16   TERMINATED                 43.2157    3.72725   12.5144             480.46              676.711          0.324731            0.273669     0.000447269           2.83607       30          1413.58      0.279205 │
│ do_train_5b048a4c   TERMINATED                425.097     1.44157    6.89119            552.743            1437.9            0.299779            0.353106     0.000657592           4.46621        1            70.3493   10        │
│ do_train_05e5def6   TERMINATED                377.917     2.58008    6.11014           1806.77             1363.65           0.194811            0.0383053    0.000783434           7.3527         1            73.8102   10        │
│ do_train_683eb1da   TERMINATED                135.857     4.23202   14.1948            1491.8              1418.27           0.346764            0.232292     0.000413192           2.02516        1            73.0314   10        │
│ do_train_13380ede   TERMINATED                224.568     2.17509   14.2337             320.035             356.776          0.066112            0.34636      0.000384145           8.04942        1            73.3506   10        │
│ do_train_e0c62cbf   TERMINATED                 39.6879    3.3144     9.9833            1482.82             2001.62           0.174767            0.260067     3.58231e-06           5.22356       30           921.208     0.650595 │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

Best trial by loss:
 config: {'hidden_dim_per_head': np.float64(49.07636920947619), 'layers': np.float64(2.706008393259113), 'heads': np.float64(4.431793909441377), 'encoder_pf_dim': np.float64(1279.732353194125), 'decoder_pf_head': np.float64(1954.0621745348171), 'encoder_dropout': np.float64(0.0), 'decoder_dropout': np.float64(0.0), 'lr': np.float64(0.0008), 'gradient_clip': np.float64(1.0), 'batch_size': 64, 'valid_batch_size': 512, 'max_epochs': 30}
 val loss: 0.1569441005587578
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/home/mrqcae001/nguni-joint-morphological-parsing/packages/transformer-segmenter/src/transformer_segmenter/model.py", line 1079, in <module>
    do_tune_parse()
  File "/home/mrqcae001/nguni-joint-morphological-parsing/packages/transformer-segmenter/src/transformer_segmenter/model.py", line 1073, in do_tune_parse
    tune_model(lambda conf, dev: Seq2Seq.from_config(vocab, conf, dev), search_space, "parse_bayesopt", fixed_cfg, train_dataset, valid_dataset, cpus=int(os.environ.get("RAY_CPUS") or 1), hrs=24)
  File "/home/mrqcae001/nguni-joint-morphological-parsing/packages/transformer-segmenter/src/transformer_segmenter/model.py", line 779, in tune_model
    with open(data_path, "rb") as fp:
         ^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/scratch/mrqcae001/checkpoints/parse_bayesopt/07022c54/checkpoint_000014/data.pkl'

"""