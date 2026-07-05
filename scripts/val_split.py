"""Build a validation set that is balanced across score bands AND across a
key feature dimension (essay length), so val metrics reflect the whole space
rather than the middle-score / medium-length majority.

Strategy:
  - For every score band (1..6) we take up to `val_per_band` essays, but never
    more than `max_frac` of that band (protects the tiny band 6).
  - Within each band we bucket essays into `n_len_bins` length terciles and
    draw evenly from each bucket, so short / medium / long essays are all
    represented at every score level.
Returns boolean masks (val_mask) aligned to df.index order.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_balanced_val(df: pd.DataFrame, length: np.ndarray,
                      val_per_band: int = 70, n_len_bins: int = 3,
                      max_frac: float = 0.4, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    val_pos: list[int] = []
    pos = np.arange(len(df))

    for band in sorted(df["score"].unique()):
        band_pos = pos[df["score"].values == band]
        cap = int(len(band_pos) * max_frac)
        target = min(val_per_band, cap)
        if target <= 0:
            continue

        lens = length[band_pos]
        # tercile bins within this band (fewer bins if not enough unique values)
        nb = min(n_len_bins, max(1, len(np.unique(lens))))
        try:
            bins = pd.qcut(lens, q=nb, labels=False, duplicates="drop")
        except ValueError:
            bins = np.zeros(len(lens), dtype=int)
        bins = np.asarray(bins)

        uniq_bins = np.unique(bins)
        per_bin = max(1, target // len(uniq_bins))
        chosen: list[int] = []
        for b in uniq_bins:
            cand = band_pos[bins == b]
            take = min(per_bin, len(cand))
            chosen.extend(rng.choice(cand, size=take, replace=False).tolist())
        # top up to target if rounding left us short
        if len(chosen) < target:
            remaining = np.setdiff1d(band_pos, np.array(chosen, dtype=int))
            extra = min(target - len(chosen), len(remaining))
            if extra > 0:
                chosen.extend(rng.choice(remaining, size=extra, replace=False).tolist())
        val_pos.extend(chosen)

    val_mask = np.zeros(len(df), dtype=bool)
    val_mask[np.array(val_pos, dtype=int)] = True
    return val_mask


def describe_split(df: pd.DataFrame, length: np.ndarray, val_mask: np.ndarray,
                   n_len_bins: int = 3) -> str:
    """Human-readable summary of the val split coverage."""
    lines = ["score | train | val", "------+-------+----"]
    for band in sorted(df["score"].unique()):
        m = df["score"].values == band
        n_tr = int((m & ~val_mask).sum())
        n_va = int((m & val_mask).sum())
        lines.append(f"  {int(band)}   | {n_tr:5d} | {n_va:3d}")
    lines.append(f"TOTAL | {int((~val_mask).sum()):5d} | {int(val_mask.sum()):3d}")

    # length-dimension coverage in val
    va_len = length[val_mask]
    qs = np.quantile(length, [0, .33, .66, 1.0])
    lo = int((va_len <= qs[1]).sum())
    mid = int(((va_len > qs[1]) & (va_len <= qs[2])).sum())
    hi = int((va_len > qs[2]).sum())
    lines.append(f"\nval length coverage  short/med/long = {lo}/{mid}/{hi}")
    return "\n".join(lines)
