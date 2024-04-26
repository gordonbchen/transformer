import torch
import torch.nn as nn
from torch.nn import functional as F

from hyperparams import HyperParms as HP


class Transformer(nn.Module):
    """Transformer language model."""

    def __init__(self, vocab_size: int, embed_size: int, block_size: int, n_heads: int, n_blocks: int, dropout: float) -> None:
        super().__init__()
        self.block_size = block_size

        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(self.block_size, embed_size)

        self.attention_blocks = nn.Sequential(*(
            AttentionBlock(n_heads, embed_size, block_size, dropout)
            for i in range(n_blocks)
        ))

        self.layer_norm = nn.LayerNorm(embed_size)

        self.model_head = nn.Linear(embed_size, vocab_size)
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor = None) -> tuple[torch.Tensor, None|torch.Tensor]:
        B, T = inputs.shape  # inputs and targets are (B, T).

        token_embeddings = self.token_embedding(inputs)  # (B, T, embed_size).
        position_embeddings = self.position_embedding(torch.arange(T, device=HP.DEVICE))  # (T, embed_size).
        embeddings = token_embeddings + position_embeddings

        z = self.attention_blocks(embeddings)
        z = self.layer_norm(z)
        logits = self.model_head(z)  # (B, T, vocab_size).
        
        loss = None
        
        # Calc cross-entropy loss.
        if targets != None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)

            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, tokens: torch.Tensor, n_tokens: int) -> torch.Tensor:
        for i in range(n_tokens):
            logits, _ = self(tokens[:, -self.block_size:])  # (B, T, C)
            logits = logits[:, -1, :]  # only use last pred col. (B, C)
            
            probs = F.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            
            tokens = torch.cat((tokens, next_tokens), dim=-1)  # append. (B, T+1)
        
        return tokens


class AttentionBlock(nn.Module):
    """A multi-head attention block."""

    def __init__(self, n_heads: int, embed_size: int, block_size: int, dropout: float) -> None:
        super().__init__()
        head_size = embed_size // n_heads
        self.mha = MultiHeadAttention(n_heads, head_size, embed_size, block_size, dropout)
        self.ffwd = FeedForward(embed_size, dropout)

        self.layer_norm1 = nn.LayerNorm(embed_size)
        self.layer_norm2 = nn.LayerNorm(embed_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        z = self.layer_norm1(inputs)
        z = z + self.mha(z)

        z = self.layer_norm2(z)
        z = z + self.ffwd(z)
        return z


class MultiHeadAttention(nn.Module):
    """Multi-head attention."""

    def __init__(self, n_heads: int, head_size: int, embed_size: int, block_size: int, dropout: float) -> None:
        super().__init__()
        self.heads = nn.ModuleList([
            Head(embed_size, head_size, block_size, dropout)
            for i in range(n_heads)
        ])
        self.linear = nn.Linear(embed_size, embed_size, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        z = torch.cat([h(inputs) for h in self.heads], dim=-1)  # concat over channel dim.
        z = self.linear(z)
        z = self.dropout(z)
        return z


class Head(nn.Module):
    """A single self-attention head."""
    
    def __init__(self, embed_size: int, head_size: int, block_size: int, dropout: float) -> None:
        super().__init__()
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)

        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

        self.head_size = head_size

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Calculate keys and queries from token embeddings.
        k = self.key(inputs)
        q = self.query(inputs)

        # Calculate affinities from k and q.
        wei = k @ q.transpose(-2, -1)
        wei = wei * (self.head_size ** -0.5)  # scale by 1/sqrt(head_size).

        T = wei.shape[-1]
        wei = wei.masked_fill((self.tril == 0)[:T, :T], float("-inf"))  # mask future.
        
        wei = F.softmax(wei, dim=-1)  # softmax smoothing.

        wei = self.dropout(wei)

        # Take weighted mean of values.
        v = self.value(inputs)
        out = wei @ v
        return out


class FeedForward(nn.Module):
    """A simple MLP feed-forward module."""

    def __init__(self, embed_size: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4, bias=True),
            nn.ReLU(),
            nn.Linear(embed_size * 4, embed_size, bias=True),
            nn.Dropout(dropout)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)
