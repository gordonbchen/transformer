import torch
import torch.nn as nn
from torch.nn import functional as F


class BigramModel(nn.Module):
    """Bigram language model."""
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, vocab_size)
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor = None) -> tuple[torch.Tensor, None|torch.Tensor]:
        # inputs and targets are (B, T).
        logits = self.token_embedding(inputs)  # (B, T, C).
        
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
            logits, _ = self(tokens)  # (B, T, C)
            logits = logits[:, -1, :]  # only use last pred col. (B, C)
            
            probs = F.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            
            tokens = torch.cat((tokens, next_tokens), dim=-1)  # append. (B, T+1)
        
        return tokens
