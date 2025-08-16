import torch.nn as nn
from .attention import MultiHeadAttentionBlock
from .layers import FeedForwardBlock, ResidualConnection
    
class EncoderBlock(nn.Module):

    def __init__(self, d_model: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module):

    def __init__(self, d_model: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)