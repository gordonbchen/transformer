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
        self.position_embedding = SinPositionalEncoding(block_size, embed_size)

        self.attention_blocks = nn.Sequential(*(
            AttentionBlock(n_heads, embed_size, block_size, dropout)
            for i in range(n_blocks)
        ))

        self.layer_norm = nn.LayerNorm(embed_size)

        self.model_head = nn.Linear(embed_size, vocab_size)
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor = None) -> tuple[torch.Tensor, None|torch.Tensor]:
        B, T = inputs.shape  # inputs and targets are (B, T).

        token_embeddings = self.token_embedding(inputs)  # (B, T, embed_size).
        embeddings = self.position_embedding(token_embeddings)  # Add positional embeddings.

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
        self.mha = MultiHeadAttention(n_heads, embed_size, block_size, dropout)
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

    def __init__(self, n_heads: int, embed_size: int, block_size: int, dropout: float) -> None:
        super().__init__()
        
        assert embed_size % n_heads == 0, f"embed_size {embed_size} must be divisible by n_heads {n_heads}."
        self.head_size = embed_size // n_heads

        self.n_heads = n_heads

        self.k_net = nn.Linear(embed_size, embed_size, bias=False)
        self.q_net = nn.Linear(embed_size, embed_size, bias=False)
        self.v_net = nn.Linear(embed_size, embed_size, bias=False)

        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.wei_dropout = nn.Dropout(dropout)

        self.linear = nn.Linear(embed_size, embed_size, bias=True)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        K = self.k_net(inputs)
        Q = self.q_net(inputs)
        V = self.v_net(inputs)

        K, Q, V = (self.split_heads(x) for x in (K, Q, V))
        
        z = self.scaled_dot_product_attention(K, Q, V)

        z = self.combine_heads(z)

        z = self.linear(z)
        z = self.output_dropout(z)
        return z
    
    def scaled_dot_product_attention(self, K, Q, V):
        wei = (K @ Q.transpose(-2, -1))  # Calculate affinities.
        wei = wei * (self.head_size ** -0.5)  # Scale by 1/sqrt(head_size).
        
        T = wei.shape[-1]  # Only mask to time length (for short seqence generation).
        wei = wei.masked_fill((self.tril == 0)[:T, :T], float("-inf"))  # Mask future.

        wei = F.softmax(wei, dim=-1)  # Softmax smoothing.
        wei = self.wei_dropout(wei)

        out = wei @ V  # Take weighted mean of values.
        return out

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        return x.view(B, T, self.n_heads, self.head_size).transpose(1, 2)
    
    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, n_heads, T, head_size = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, int(n_heads * head_size))
    

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


class SinPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, max_seq_length: int, embed_size: int) -> None:
        super().__init__()

        pos_encoding = torch.zeros(max_seq_length, embed_size, dtype=torch.float32)

        pos_arange = torch.arange(max_seq_length, dtype=torch.float32).unsqueeze(1)
        dim_arange = torch.arange(embed_size // 2, dtype=torch.float32)

        div_factor = 10_000 ** ((2 * dim_arange) / embed_size)
        pos_encoding[:, ::2] = torch.sin(pos_arange / div_factor)
        pos_encoding[:, 1::2] = torch.cos(pos_arange / div_factor)

        self.register_buffer("pos_encoding", pos_encoding.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_encoding[:, :x.shape[-2]]
