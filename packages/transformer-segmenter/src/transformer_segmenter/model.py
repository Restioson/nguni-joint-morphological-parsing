import csv
import math
import os
import pickle
import re
import tempfile
import time
from enum import Enum
from pathlib import Path

import ray
import torch
import torch.nn as nn
import tqdm
from ray.tune import Checkpoint, get_checkpoint, CheckpointConfig
from ray import tune
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
            ray.train.report(
                {"loss": max_loss},
                checkpoint=checkpoint,
            )

    # Specify learning rate and optimisation function
    lr = cfg["lr"]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.output_pad_ix)
    n_epochs = cfg["max_epochs"]
    gradient_clip = cfg["gradient_clip"]
    batch_size = cfg["batch_size"]
    valid_batch_size = cfg["valid_batch_size"]
    start_epoch = 0

    if use_ray:
        checkpoint = get_checkpoint()
        if checkpoint:
            with checkpoint.as_directory() as checkpoint_dir:
                data_path = Path(checkpoint_dir) / "data.pkl"
                with open(data_path, "rb") as fp:
                    checkpoint_state = pickle.load(fp)
                start_epoch = checkpoint_state["epoch"]
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

    best_valid_loss = float('inf')
    best_valid_loss_epoch = 0

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
                ray.train.report(
                    {"loss": valid_loss},
                    checkpoint=checkpoint,
                )

def tune_model(model_for_config, search_space, name: str, fixed_cfg, train_set: TransformerDataset,
               valid_set: TransformerDataset, cpus=4, hrs=11, max_loss=10):
    """Tune the given model with Ray"""

    ray.init(num_cpus=cpus)

    algo = BayesOptSearch(space=search_space, metric="loss", mode="min")
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

    # Do the hyperparameter tuning
    result = tune.run(
        do_train,
        metric="loss",
        name=name,
        resume=True,
        mode="min",
        resources_per_trial={"gpu": 1.0 / cpus} if torch.cuda.is_available() else None,
        # config=search_space,
        num_samples=100,
        time_budget_s=hrs * 60 * 60,
        search_alg=algo,
        checkpoint_config=CheckpointConfig(
            num_to_keep=3,
            checkpoint_score_attribute="loss",
            checkpoint_score_order="min",
        ),
        storage_path=str(Path(os.environ["TUNING_CHECKPOINT_DIR"]).resolve()),
        trial_dirname_creator=lambda trial: trial.trial_id,
        # scheduler=scheduler,
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
        # print(f" macro f1 {best_trial.last_result['f1_macro']}")
        # print(f" micro {best_trial.last_result['f1_micro']}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        conf = {param: (val if param not in int_params else round(val)) for param, val in best_trial.config.items()}
        best_model = model_for_config(conf, device)
        best_model = best_model.to(device)

        best_checkpoint = result.get_best_checkpoint(trial=best_trial, metric=metric, mode="max")
        with best_checkpoint.as_directory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "rb") as fp:
                best_checkpoint_data = pickle.load(fp)

            best_model.load_state_dict(best_checkpoint_data["model_state_dict"])
            f1_micro, f1_macro, hr_at_1 = f1_scores(valid_loader, best_model, SequenceAlignment.ALIGNED_MULTISET)
            print(f" {name}: Micro F1: {f1_micro}. Macro f1: {f1_macro}. Hit-rate @ 1: {hr_at_1 * 100:.2f}%")
            print(
                f" {name}: Best macro f1: {best_checkpoint_data['best_macro']} at epoch "
                f"{best_checkpoint_data['best_epoch']}")


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

    cfg = {**Seq2Seq.DEFAULT_CONFIG, "decoder_pf_head": 2019, "encoder_pf_dim": 805, "heads": 7, "hidden_dim_per_head": 268, "layers": 1}

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
    train(model, cfg, "hyperopt-big", train_dataset, valid_dataset, device, SequenceAlignment.ALIGNED_MULTISET)

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

    # TODO _try_ a grid search
    fixed_cfg = {"batch_size": 64, "valid_batch_size": 512, "max_epochs": 30}
    search_space = {
        "hidden_dim_per_head": tune.qrandint(16, 512),
        "layers": tune.qrandint(1, 5),
        "heads": tune.qrandint(4, 16),
        "encoder_pf_dim": tune.qrandint(128, 2048),
        "decoder_pf_head": tune.qrandint(128, 2048),
        "encoder_dropout": tune.uniform(0.0, 0.4),
        "decoder_dropout": tune.uniform(0.0, 0.4),
        "lr": tune.uniform(1e-6, 8e-4),
        "gradient_clip": tune.uniform(1, 10),
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
    tune_model(lambda conf, dev: Seq2Seq.from_config(vocab, conf, dev), search_space, "parse_bayesopt", fixed_cfg, train_dataset, valid_dataset, cpus=os.environ["RAY_CPUS"] or 1, hrs=24)


if __name__ == "__main__":
    # do_train_parse()
    # do_train_seg()
    do_tune_parse()