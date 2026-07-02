import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Feedforward network cho 45-class multi-label classification.
    Input: flat feature vector (252-dim allele heights).
    Output: logits 45-dim (sigmoid ở ngoài khi inference).
    """

    def __init__(
        self,
        in_features: int = 252,
        hidden_dims: list[int] = None,
        n_classes: int = 45,
        dropout: float = 0.3,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]

        layers = []
        prev = in_features
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, n_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
