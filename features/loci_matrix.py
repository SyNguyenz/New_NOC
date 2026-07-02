"""
Transform flat feature vector (252-dim) thành ma trận (n_loci x max_alleles)
để làm input cho 1D CNN.

Feature names có dạng LOCUS_ALLELE, ví dụ:
  D3S1358_15, vWA_16, SE33_19.2, AMEL_X, Yindel_1
"""

from __future__ import annotations
import json
import re
from pathlib import Path

import numpy as np


# Map allele string → float bin (để sort nhất quán)
def _allele_to_float(allele: str) -> float:
    if allele == "X":
        return -2.0
    if allele == "Y":
        return -1.0
    try:
        return float(allele)
    except ValueError:
        return 0.0


class LociMatrix:
    """
    Biến flat feature vector → tensor (n_loci, max_alleles).

    Mỗi locus chiếm 1 hàng; các allele bin được sort theo giá trị allele
    và zero-padded đến max_alleles.
    """

    def __init__(self, feature_cols: list[str]):
        self.feature_cols = feature_cols
        self._build_mapping(feature_cols)

    def _build_mapping(self, feature_cols: list[str]):
        from collections import defaultdict
        locus_to_feats: dict[str, list[tuple[float, int]]] = defaultdict(list)

        for idx, col in enumerate(feature_cols):
            # Split trên dấu _ cuối cùng → (locus, allele)
            last_underscore = col.rfind("_")
            if last_underscore == -1:
                continue
            locus  = col[:last_underscore]
            allele = col[last_underscore + 1:]
            locus_to_feats[locus].append((_allele_to_float(allele), idx))

        # Sort loci theo tên (reproducible order)
        self.loci = sorted(locus_to_feats.keys())
        self.n_loci = len(self.loci)

        # Sort allele bins trong mỗi locus theo giá trị allele
        self._locus_feat_indices: list[list[int]] = []
        for locus in self.loci:
            sorted_feats = sorted(locus_to_feats[locus], key=lambda t: t[0])
            self._locus_feat_indices.append([idx for _, idx in sorted_feats])

        self.max_alleles = max(len(v) for v in self._locus_feat_indices)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n_loci, self.max_alleles)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: (N, n_features)
        Returns:
            (N, n_loci, max_alleles)  float32
        """
        N = X.shape[0]
        out = np.zeros((N, self.n_loci, self.max_alleles), dtype=np.float32)
        for i, feat_indices in enumerate(self._locus_feat_indices):
            n = len(feat_indices)
            out[:, i, :n] = X[:, feat_indices]
        return out

    def save(self, path: str | Path):
        data = {
            "loci": self.loci,
            "locus_feat_indices": self._locus_feat_indices,
            "max_alleles": self.max_alleles,
            "feature_cols": self.feature_cols,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str | Path) -> "LociMatrix":
        with open(path) as f:
            data = json.load(f)
        obj = cls.__new__(cls)
        obj.feature_cols = data["feature_cols"]
        obj.loci = data["loci"]
        obj.n_loci = len(obj.loci)
        obj._locus_feat_indices = data["locus_feat_indices"]
        obj.max_alleles = data["max_alleles"]
        return obj


if __name__ == "__main__":
    meta = json.load(open(Path(__file__).parents[1] / "data" / "meta.json"))
    lm = LociMatrix(meta["feature_cols"])
    print(f"n_loci     : {lm.n_loci}")
    print(f"max_alleles: {lm.max_alleles}")
    print(f"matrix shape per sample: {lm.shape}")
    print("\nLoci and their allele counts:")
    for locus, idxs in zip(lm.loci, lm._locus_feat_indices):
        print(f"  {locus:<12} {len(idxs):>3} alleles")
