import torch
from pathlib import Path

def get_config(base_path="/"):
    data_root = Path(base_path) / "data"
    tokenizer_root = Path(base_path) / "tokenizers"
    save_dir_root = Path(base_path) / "checkpoints"

    config = {
        # Data & Tokenizer
        "train_csv": str(data_root / "train.csv"),
        "val_csv": str(data_root / "validation.csv"),
        "test_csv": None,
        "tokenizer_file": str(tokenizer_root / "spm_vietnamese_model.model"),

        # Sequence lengths
        "max_len_text": 1024,
        "max_len_summary": 128,

        # DataLoader
        "batch_size": 24,
        "val_batch_size": 32,
        "num_workers": 12,
        
        # Truncate strategies (Dataset)
        "truncate_strategy_text": "head_tail",
        "truncate_strategy_summary": "head",

        # Model
        "d_model": 512,
        "num_layers": 6,
        "num_heads": 8,
        "d_ff": 2048,
        "dropout": 0.2,

        # Training
        "epochs": 20,
        "lr": 2e-4,
        "weight_decay": 3e-2,
        "max_grad_norm": 1.0,
        "label_smoothing": 0.1,
        "label_smoothing_dropoff_epoch": -1,

        # Memory/throughput helpers
        "use_amp": True,
        "num_accumulation_steps": 6,

        # Checkpoint & Logging
        "save_dir": str(save_dir_root),
        "model_basename": "transformer_summarizer_",
        "preload": None,
        "log_interval": 50,


        # Device
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

    Path(config["save_dir"]).mkdir(parents=True, exist_ok=True)
    return config