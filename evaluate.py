import torch.nn as nn
import torch

def compute_loss(logits: torch.Tensor, labels: torch.Tensor, pad_id: int) -> torch.Tensor:
    return nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=pad_id
    )

@torch.no_grad()
def evaluate(model, dataloader, device, pad_id):
    model.eval()
    total_loss, total_count = 0.0, 0
    for batch in dataloader:
        src = batch['encoder_input'].to(device)
        tgt_in = batch['decoder_input'].to(device)
        labels = batch['label'].to(device)
        src_mask = batch['encoder_mask'].to(device)
        tgt_mask = batch['decoder_mask'].to(device)

        enc_out = model.encode(src, src_mask)
        dec_out = model.decode(enc_out, src_mask, tgt_in, tgt_mask)
        logits = model.project(dec_out)
        loss = compute_loss(logits, labels, pad_id)
        total_loss += loss.item()
        total_count += 1
    return total_loss / max(total_count, 1)