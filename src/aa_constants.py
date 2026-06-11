"""Amino-acid lookup tables: BLOSUM62, Grantham distance, physicochemical scales.

Self-contained (no external dependency). Non-standard codes and `*` (stop)
return NaN from the matrix lookups; callers flag those rows separately.
"""
from __future__ import annotations

import numpy as np

# Canonical BLOSUM62 ordering.
_B62_ORDER = list("ARNDCQEGHILKMFPSTWYV")
_B62_ROWS = [
    [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0],
    [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3],
    [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3],
    [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3],
    [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],
    [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2],
    [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2],
    [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3],
    [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3],
    [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3],
    [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1],
    [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2],
    [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1],
    [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1],
    [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2],
    [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2],
    [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0],
    [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3],
    [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1],
    [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4],
]
BLOSUM62 = {
    a: {b: _B62_ROWS[i][j] for j, b in enumerate(_B62_ORDER)}
    for i, a in enumerate(_B62_ORDER)
}

# Grantham (1974) properties: composition (c), polarity (p), volume (v).
# Distance is computed from these so we avoid embedding the 190-pair table.
_GRANTHAM_PROPS = {
    "S": (1.42, 9.2, 32), "R": (0.65, 10.5, 124), "L": (0.0, 4.9, 111),
    "P": (0.39, 8.0, 32.5), "T": (0.71, 8.6, 61), "A": (0.0, 8.1, 31),
    "V": (0.0, 5.9, 84), "G": (0.74, 9.0, 3), "I": (0.0, 5.2, 111),
    "F": (0.0, 5.2, 132), "Y": (0.20, 6.2, 136), "C": (2.75, 5.5, 55),
    "H": (0.58, 10.4, 96), "Q": (0.89, 10.5, 85), "N": (1.33, 11.6, 56),
    "K": (0.33, 11.3, 119), "D": (1.38, 13.0, 54), "E": (0.92, 12.3, 83),
    "M": (0.0, 5.7, 105), "W": (0.13, 5.4, 170),
}
_GRANTHAM_ALPHA, _GRANTHAM_BETA, _GRANTHAM_GAMMA = 1.833, 0.1018, 0.000399
_GRANTHAM_RHO = 50.723  # scaling so values match the published ~5-215 range

# Physicochemical property scales (20 standard AAs).
PHYS = {
    "hydropathy": {  # Kyte-Doolittle
        "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5,
        "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9,
        "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9,
        "Y": -1.3, "V": 4.2,
    },
    "weight": {  # residue molecular weight (Da)
        "A": 89, "R": 174, "N": 132, "D": 133, "C": 121, "Q": 146, "E": 147,
        "G": 75, "H": 155, "I": 131, "L": 131, "K": 146, "M": 149, "F": 165,
        "P": 115, "S": 105, "T": 119, "W": 204, "Y": 181, "V": 117,
    },
    "volume": {  # van der Waals volume (A^3) — Grantham v
        "A": 31, "R": 124, "N": 56, "D": 54, "C": 55, "Q": 85, "E": 83,
        "G": 3, "H": 96, "I": 111, "L": 111, "K": 119, "M": 105, "F": 132,
        "P": 32.5, "S": 32, "T": 61, "W": 170, "Y": 136, "V": 84,
    },
    "polarity": {  # Grantham p
        "A": 8.1, "R": 10.5, "N": 11.6, "D": 13.0, "C": 5.5, "Q": 10.5,
        "E": 12.3, "G": 9.0, "H": 10.4, "I": 5.2, "L": 4.9, "K": 11.3,
        "M": 5.7, "F": 5.2, "P": 8.0, "S": 9.2, "T": 8.6, "W": 5.4,
        "Y": 6.2, "V": 5.9,
    },
    "pi": {  # isoelectric point
        "A": 6.0, "R": 10.76, "N": 5.41, "D": 2.77, "C": 5.07, "Q": 5.65,
        "E": 3.22, "G": 5.97, "H": 7.59, "I": 6.02, "L": 5.98, "K": 9.74,
        "M": 5.74, "F": 5.48, "P": 6.30, "S": 5.68, "T": 5.60, "W": 5.89,
        "Y": 5.66, "V": 5.96,
    },
    "charge": {  # approximate net charge at pH 7.4
        "D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.1,
        "A": 0.0, "N": 0.0, "C": 0.0, "Q": 0.0, "G": 0.0, "I": 0.0,
        "L": 0.0, "M": 0.0, "F": 0.0, "P": 0.0, "S": 0.0, "T": 0.0,
        "W": 0.0, "Y": 0.0, "V": 0.0,
    },
}

# Chemical group membership (for group_change feature).
GROUP = {
    "A": "nonpolar", "V": "nonpolar", "L": "nonpolar", "I": "nonpolar",
    "M": "nonpolar", "P": "nonpolar", "G": "nonpolar",
    "F": "aromatic", "W": "aromatic", "Y": "aromatic",
    "C": "polar", "S": "polar", "T": "polar", "N": "polar", "Q": "polar",
    "H": "positive", "K": "positive", "R": "positive",
    "D": "negative", "E": "negative",
}


def _valid(aa) -> bool:
    return isinstance(aa, str) and aa in _GRANTHAM_PROPS


def blosum62(a, b) -> float:
    if isinstance(a, str) and isinstance(b, str) and a in BLOSUM62 and b in BLOSUM62[a]:
        return float(BLOSUM62[a][b])
    return np.nan


def grantham(a, b) -> float:
    if not (_valid(a) and _valid(b)):
        return np.nan
    ca, pa, va = _GRANTHAM_PROPS[a]
    cb, pb, vb = _GRANTHAM_PROPS[b]
    d = (_GRANTHAM_ALPHA * (ca - cb) ** 2
         + _GRANTHAM_BETA * (pa - pb) ** 2
         + _GRANTHAM_GAMMA * (va - vb) ** 2)
    return float(_GRANTHAM_RHO * np.sqrt(d))


def group_change(a, b) -> int:
    ga, gb = GROUP.get(a), GROUP.get(b)
    if ga is None or gb is None:
        return 0
    return int(ga != gb)
