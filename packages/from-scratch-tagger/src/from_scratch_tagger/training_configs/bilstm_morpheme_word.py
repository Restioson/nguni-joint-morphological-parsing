import datetime
import sys
from pathlib import Path

from ray import tune
from ray.util.client import ray

from ..common import TaggingDataset, tune_model, train_all, \
    EmbedSingletonFeature, tokenize_into_morphemes, split_words
from ..lstm import BiLSTMTagger

# Some configurable aspects of the model - the model itself, context level, and submorpheme tokenisation
model = (
    "bilstm",
    lambda train_set, embed, config, dev: BiLSTMTagger(embed, config, train_set, dev)
)
splits = (split_words, "words", 20)
feature_level = (
    "morpheme",
    {
        "embed_target_embed": tune.grid_search([256, 512, 1024][::-1]),
    },
    tokenize_into_morphemes,
    lambda config, dset, dev: EmbedSingletonFeature(dset, config["embed_target_embed"], dev)
)

split, split_name, epochs = splits
model_name, mk_model = model
(feature_name, _, extract_features, embed_features) = feature_level

name = f"split-{split_name}feature-{feature_level}_model-{model_name}"

data_dir = Path(sys.argv[1])


def fine_tune():
    """Tune the model to select best hyperparameters"""

    print(f"Tuning {split_name}-level, {feature_name}-feature {model_name} for ZU")
    cfg = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "weight_decay": tune.loguniform(1e-10, 1e-5),
        "hidden_dim": tune.grid_search([256, 512, 1024][::-1]),
        "dropout": tune.choice([0.1, 0.2]),
        "batch_size": tune.choice([4]),
        "epochs": tune.choice([epochs]),
        "gradient_clip": tune.choice([0.5, 1, 2, 4, float("inf")])
    }

    train, valid = TaggingDataset.load_data("ZU", data_dir, split=split, tokenize=extract_features)
    tune_model(model, cfg, feature_level, name, epochs, train, valid)


def final_train():
    """Train & save the model with a given config"""

    cfg = {
        'lr': 0.0002948382869797967,
        'weight_decay': 0,
        'hidden_dim': 256,
        'dropout': 0.2,
        'batch_size': 1,
        'epochs': {
            "NR": 25,
            "SS": 25,
            "XH": 22,
            "ZU": 17,
        },
        'gradient_clip': 2,
        'embed_target_embed': 128,
    }

    train_all(data_dir, model, splits, feature_level, cfg, langs=["ZU"], add_valid_to_training_set=False)


final_train()

print("Done at", datetime.datetime.now())
ray.shutdown()
