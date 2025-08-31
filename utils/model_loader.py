from pathlib import Path
import torch
import sentencepiece as spm

from utils.config import get_config
from model.transformer import build_transformer

def load_model_resources(base_dir: str):
    """
    Tải tokenizer và model
    """
    config = get_config(base_dir)
    device = config['device']
    
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(config["tokenizer_file"])

    model = build_transformer(
        src_vocab_size=tokenizer.get_piece_size(),
        tgt_vocab_size=tokenizer.get_piece_size(),
        src_seq_len=config["max_len_text"],
        tgt_seq_len=config["max_len_summary"],
        d_model=config["d_model"],
        N=config["num_layers"],
        h=config["num_heads"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
    ).to(device)
    
    model_path = Path(config["save_dir"]) / "transformer_summarizer_best.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}. Please train the model first by running `train.py`.")
        
    state = torch.load(model_path, map_location=device)
    
    state_dict = state['model_state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            # Key from torch.compile
            new_state_dict[k[len('_orig_mod.'):]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)

    model.eval()
    return model, tokenizer, config, device
