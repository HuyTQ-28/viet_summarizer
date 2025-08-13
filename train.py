import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from utils.config import get_config
from utils.data_loader import load_data_from_csv, get_data_loaders
from utils.data_loader import clear_causal_mask_cache
from evaluate import evaluate, compute_loss

from model.transformer import build_transformer

import sentencepiece as spm


def load_tokenizer(spm_model_file: str):
    """
    Tải tokenizer từ file spm model
    Args:
        spm_model_file: str, đường dẫn đến file spm model
    Returns:
        sp: SentencePieceProcessor, tokenizer
    Raises:
        FileNotFoundError: nếu file spm model không tồn tại
        ValueError: nếu tokenizer thiếu BOS/EOS/PAD
    """
    sp = spm.SentencePieceProcessor()
    if not os.path.exists(spm_model_file):
        raise FileNotFoundError(f"Không tìm thấy tokenizer model: {spm_model_file}")
    sp.load(spm_model_file)
    # Kiểm tra các token đặc biệt tồn tại
    for name, fn in {"bos_id": sp.bos_id, "eos_id": sp.eos_id, "pad_id": sp.pad_id}.items():
        if fn() < 0:
            raise ValueError(f"Tokenizer thiếu {name}. Hãy huấn luyện tokenizer với BOS/EOS/PAD hoặc chỉnh lại dataset.")
    return sp

def train_one_epoch(model, optimizer, dataloader, device, pad_id, log_interval=50, writer=None, epoch=0, max_grad_norm: float = 1.0,
                    use_amp: bool = False, num_accumulation_steps: int = 1):
    """
    Huấn luyện mô hình trong một epoch
    Args:
        model: TransformerModel
        optimizer: Optimizer
        dataloader: DataLoader
        device: Device
        pad_id: int
        log_interval: int
        writer: SummaryWriter
        epoch: int
        max_grad_norm: float
    """
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    running_loss = 0.0
    count = 0
    start = time.time()

    for step, batch in enumerate(dataloader):
        src = batch['encoder_input'].to(device)
        tgt_in = batch['decoder_input'].to(device)
        labels = batch['label'].to(device)
        src_mask = batch['encoder_mask'].to(device)
        tgt_mask = batch['decoder_mask'].to(device)

        if (step % num_accumulation_steps) == 0:
            optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            enc_out = model.encode(src, src_mask)
            dec_out = model.decode(enc_out, src_mask, tgt_in, tgt_mask)
            logits = model.project(dec_out)
            loss = compute_loss(logits, labels, pad_id)

        loss_to_backprop = loss / max(1, num_accumulation_steps)
        scaler.scale(loss_to_backprop).backward()

        # Step when accumulation boundary reached
        if ((step + 1) % num_accumulation_steps) == 0:
            # Gradient clipping to stabilize training
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

        running_loss += loss.item()
        count += 1

        if (step + 1) % log_interval == 0:
            avg_loss = running_loss / count
            if writer:
                global_step = epoch * len(dataloader) + step
                writer.add_scalar('train/loss', avg_loss, global_step)
            elapsed = time.time() - start
            print(f"Epoch {epoch+1} Step {step+1}/{len(dataloader)} - loss: {avg_loss:.4f} - {elapsed:.1f}s")
            running_loss = 0.0
            count = 0
            start = time.time()

def main():
    config = get_config()

    device = torch.device(config["device"])

    # Tokenizer
    tokenizer = load_tokenizer(config["tokenizer_file"])
    pad_id = int(tokenizer.pad_id())

    # Data
    train_df, val_df, _ = load_data_from_csv(
        config["train_csv"],
        config.get("val_csv"),
        config.get("test_csv")
    )
    loaders = get_data_loaders(train_df, tokenizer, config, val_df=val_df)
    train_loader = loaders['train']
    val_loader = loaders.get('val')

    # Model
    vocab_size = int(tokenizer.get_piece_size())
    model = build_transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        src_seq_len=config["max_len_text"],
        tgt_seq_len=config["max_len_summary"],
        d_model=config["d_model"],
        N=config["num_layers"],
        h=config["num_heads"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
    ).to(device)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]) 

    # Logging
    os.makedirs(config["save_dir"], exist_ok=True)
    writer = None
    try:
        writer = SummaryWriter(log_dir=os.path.join(config["save_dir"], "runs"))
    except Exception:
        pass

    best_val = float('inf')

    for epoch in range(config["epochs"]):
        clear_causal_mask_cache()
        train_one_epoch(
            model, optimizer, train_loader, device, pad_id,
            log_interval=config.get("log_interval", 50), writer=writer, epoch=epoch,
            max_grad_norm=float(config.get("max_grad_norm", 1.0)),
            use_amp=bool(config.get("use_amp", True)),
            num_accumulation_steps=int(config.get("num_accumulation_steps", 1)),
        )

        if val_loader is not None:
            val_loss = evaluate(model, val_loader, device, pad_id)
            print(f"Epoch {epoch+1} validation loss: {val_loss:.4f}")
            if writer: 
                writer.add_scalar('val/loss', val_loss, epoch)
            if val_loss < best_val:
                best_val = val_loss
                ckpt_path = os.path.join(config["save_dir"], f"best.pt")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': config,
                    'val_loss': val_loss,
                    'epoch': epoch,
                }, ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}")

    if writer:
        writer.close()


if __name__ == "__main__":
    main()