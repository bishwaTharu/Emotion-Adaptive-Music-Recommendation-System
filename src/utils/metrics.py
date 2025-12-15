import torch

def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true, dim=0)) ** 2)
    return 1.0 - ss_res / ss_tot


def pearson_corr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x - x.mean(dim=0)
    y = y - y.mean(dim=0)

    num = torch.sum(x * y)
    den = torch.sqrt(torch.sum(x ** 2)) * torch.sqrt(torch.sum(y ** 2))
    return num / (den + 1e-8)
