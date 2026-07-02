import torch
import torch.nn as nn


class LociCNN(nn.Module):
    """
    1D CNN cho 45-class multi-label classification.

    Input: (batch, n_loci, max_alleles) — ma trận allele heights theo locus.
    Mỗi locus là một "channel"; conv chạy dọc theo chiều allele bins,
    học pattern tổ hợp allele trong và giữa các locus.

    Architecture:
        (B, n_loci, max_alleles)
        → Conv1d(n_loci, 64, k=3) + BN + ReLU
        → Conv1d(64, 128, k=3) + BN + ReLU
        → AdaptiveMaxPool1d(1)          # (B, 128)
        → FC 128 → 256 → 45
    """

    def __init__(
        self,
        n_loci: int = 24,
        max_alleles: int = 33,
        channels: list[int] = None,
        kernel_size: int = 3,
        n_classes: int = 45,
        dropout: float = 0.3,
    ):
        super().__init__()
        if channels is None:
            channels = [64, 128]

        conv_layers = []
        in_ch = n_loci
        for out_ch in channels:
            conv_layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size,
                          padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch

        self.conv = nn.Sequential(*conv_layers)
        self.pool = nn.AdaptiveMaxPool1d(1)

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_ch, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_loci, max_alleles)
        out = self.conv(x)          # (B, 128, max_alleles)
        out = self.pool(out)        # (B, 128, 1)
        out = out.squeeze(-1)       # (B, 128)
        return self.head(out)       # (B, 45)
