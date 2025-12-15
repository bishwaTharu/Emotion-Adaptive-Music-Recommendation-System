import torch
import torch.nn as nn

class WideDeepVA(nn.Module):
    def __init__(self, transformer: nn.Module, wide_dim: int = 7):
        super().__init__()
        self.transformer = transformer
        self.embedding_dim = transformer.config.hidden_size

        self.deep_mlp = nn.Sequential(
            nn.Linear(self.embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.wide_mlp = nn.Sequential(
            nn.Linear(wide_dim, 32),
            nn.ReLU(),
        )

        self.regressor = nn.Linear(256 + 32, 2)

    def mean_pooling(self, output: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        token_embeddings = output[0]
        mask = mask.unsqueeze(-1).float()
        return (token_embeddings * mask).sum(1) / mask.sum(1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, wide: torch.Tensor) -> torch.Tensor:
        output = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        sent_emb = self.mean_pooling(output, attention_mask)
        deep_out = self.deep_mlp(sent_emb)
        wide_out = self.wide_mlp(wide)

        fused = torch.cat([deep_out, wide_out], dim=1)
        return self.regressor(fused)
