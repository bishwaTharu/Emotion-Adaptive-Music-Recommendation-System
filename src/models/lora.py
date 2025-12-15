import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Module, r: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        in_dim = base_layer.in_features
        out_dim = base_layer.out_features

        self.A = nn.Parameter(torch.randn(r, in_dim) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_dim, r))

        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scaling


def apply_lora_to_transformer(model: nn.Module, r: int = 8, alpha: int = 16):
    """
    Inject LoRA into query, key & value projections
    """
    for module in model.modules():
        if hasattr(module, "query"):
            module.query = LoRALinear(module.query, r, alpha)
        if hasattr(module, "key"):
            module.key = LoRALinear(module.key, r, alpha)
        if hasattr(module, "value"):
            module.value = LoRALinear(module.value, r, alpha)
