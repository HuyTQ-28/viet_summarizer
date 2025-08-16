import torch
from pathlib import Path

def get_config():
    config = {
        # Data & Tokenizer
        "train_csv": "data/processed/train.csv",
        "val_csv": "data/processed/val.csv",
        "test_csv": None,
        "tokenizer_file": "tokenizers/spm_vietnamese_model.model",

        # Sequence lengths
        "max_len_text": 1024,
        "max_len_summary": 128,

        # DataLoader
        "batch_size": 6,
        "val_batch_size": 12,
        "num_workers": 8,
        
        # Truncate strategies (Dataset)
        "truncate_strategy_text": "head_tail",
        "truncate_strategy_summary": "head",

        # Model
        "d_model": 512,
        "num_layers": 6,
        "num_heads": 8,
        "d_ff": 2048,
        "dropout": 0.1,

        # Training
        "epochs": 20,
        "lr": 3e-4,
        "weight_decay": 1e-2,
        "max_grad_norm": 1.0,
        "label_smoothing": 0.1,
        "label_smoothing_dropoff_epoch": 15,

        # Memory/throughput helpers
        "use_amp": True,
        "num_accumulation_steps": 8,

        # Checkpoint & Logging
        "save_dir": "checkpoints",
        "model_basename": "transformer_summarizer_",
        "preload": None,
        "log_interval": 50,


        # Device
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

    Path(config["save_dir"]).mkdir(parents=True, exist_ok=True)
    return config