import torch.nn as nn
import torch
import torch.nn.functional as F

class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing: float = 0.0, pad_id: int = -100, reduction='mean'):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.pad_id = pad_id
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch_size * seq_len, vocab_size)
            target: (batch_size * seq_len)
        """
        target = target.reshape(-1)
        logits = logits.reshape(-1, logits.size(-1))

        vocab_size = logits.size(-1)
        
        true_dist = torch.full_like(logits, self.smoothing / (vocab_size - 2))
        true_dist.scatter_(1, target.unsqueeze(1), 1 - self.smoothing)
        
        log_probs = F.log_softmax(logits, dim=-1)
        loss = F.kl_div(log_probs, true_dist, reduction='none').sum(dim=-1)

        non_pad_mask = (target != self.pad_id)
        loss = loss.where(non_pad_mask, 0.0)

        if self.reduction == 'mean':
            return loss.sum() / non_pad_mask.sum()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

@torch.no_grad()
def evaluate(model, dataloader, device, loss_fn):
    model.eval()
    total_loss, total_count = 0.0, 0
    pad_id = loss_fn.pad_id
    for batch in dataloader:
        src = batch['encoder_input'].to(device)
        tgt_in = batch['decoder_input'].to(device)
        labels = batch['label'].to(device)
        src_mask = batch['encoder_mask'].to(device)
        tgt_mask = batch['decoder_mask'].to(device)

        enc_out = model.encode(src, src_mask)
        dec_out = model.decode(enc_out, src_mask, tgt_in, tgt_mask)
        logits = model.project(dec_out)
        loss = loss_fn(logits, labels)
        total_loss += loss.item()
        total_count += 1
    return total_loss / max(total_count, 1)