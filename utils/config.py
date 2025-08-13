import torch
from pathlib import Path

def get_config():
    config = {
        # Data & Tokenizer
        "train_csv": "data/processed/train_small.csv",
        "val_csv": "data/processed/train_small.csv",
        "test_csv": None,
        "tokenizer_file": "tokenizers/spm_vietnamese_model.model",

        # Sequence lengths
        # Với dữ liệu dài (~1085 tokens), có thể tăng max_len_text, cân nhắc batch_size/AMP
        "max_len_text": 512,
        "max_len_summary": 128,

        # DataLoader
        "batch_size": 4,
        "val_batch_size": 4,
        "num_workers": 0,
        
        # Truncate strategies (Dataset)
        # head | tail | head_tail
        "truncate_strategy_text": "head_tail",
        "truncate_strategy_summary": "head",

        # Model
        "d_model": 512,
        "num_layers": 6,
        "num_heads": 8,
        "d_ff": 2048,
        "dropout": 0.1,

        # Training
        "epochs": 3,
        "lr": 3e-4,
        "weight_decay": 1e-2,
        "max_grad_norm": 1.0,

        # Memory/throughput helpers
        "use_amp": True,
        # Tăng num_accumulation_steps để mô phỏng batch lớn khi seq_len tăng
        "num_accumulation_steps": 1,

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