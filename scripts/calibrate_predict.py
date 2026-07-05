"""Distribution-calibrated submissions from saved checkpoints.

The validation set is *balanced* across score bands, so cut points fitted on it
skew the predicted test distribution (esp. band 4). This script keeps using the
held-out val predictions (no leakage) but re-weights each val essay so the set
matches the TRUE training score distribution, then fits the QWK-optimal cut
points under that weighting. Result: thresholds calibrated for the real
distribution.

Usage:
    python calibrate_predict.py --iters 350 700
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from qwk_utils import qwk, OptimizedRounder
from predict_from_ckpt import make_matrix, LIVE, CKPT, DATA


def build_resampled_calset(val_pred, y_val, true_prop, size, seed):
    """Construct a concrete calibration set that follows the TRUE distribution
    by resampling the held-out val predictions with replacement, per class."""
    rng = np.random.default_rng(seed)
    idx_all = []
    for s in sorted(true_prop.index):
        pool = np.where(y_val == s)[0]
        if len(pool) == 0:
            continue
        n_s = max(1, int(round(size * true_prop[s])))
        idx_all.append(rng.choice(pool, size=n_s, replace=True))
    idx = np.concatenate(idx_all)
    rng.shuffle(idx)
    return val_pred[idx], y_val[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, nargs="+", default=[350, 700])
    ap.add_argument("--mode", choices=["reweight", "resample"], default="reweight",
                    help="reweight the full held-out val, or resample it into a "
                         "concrete true-distribution calibration set.")
    ap.add_argument("--seed", type=int, default=2024,
                    help="Random seed for the resampled calibration set.")
    ap.add_argument("--calib-size", type=int, default=1500)
    args = ap.parse_args()

    pre = pickle.load(open(LIVE / "preprocessor.pkl", "rb"))
    val_mask = np.load(LIVE / "val_mask.npy")
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train["score"].values.astype(float)

    y_val = y[val_mask].astype(int)

    # weight each val essay so the val set mimics the TRUE score distribution:
    #   w(s) = true_proportion(s) / val_proportion(s)
    true_prop = train["score"].value_counts(normalize=True)
    val_counts = pd.Series(y_val).value_counts()
    val_total = len(y_val)
    weights = np.array([
        true_prop[s] / (val_counts[s] / val_total) for s in y_val
    ], dtype=float)
    weights *= len(weights) / weights.sum()  # normalise to mean 1

    print("Building validation + test feature matrices ...")
    X_val = make_matrix(train["full_text"].values[val_mask], pre)
    X_test = make_matrix(test["full_text"].values, pre)

    true_pct = (true_prop.sort_index() * 100).round(1).to_dict()
    print(f"\nTrue train distribution (%): {true_pct}\n")

    summary = []
    for it in args.iters:
        path = CKPT / f"model_iter{it}.txt"
        if not path.exists():
            print(f"[skip] no checkpoint for iter {it}")
            continue
        booster = lgb.Booster(model_file=str(path))
        val_pred = booster.predict(X_val)

        if args.mode == "reweight":
            rounder = OptimizedRounder().fit(val_pred, y_val, sample_weight=weights)
            cal_qwk = qwk(y_val, rounder.predict(val_pred), sample_weight=weights)
            suffix, tag = "calibrated", "reweighted full val"
        else:
            cp, cy = build_resampled_calset(val_pred, y_val, true_prop,
                                            args.calib_size, args.seed)
            rounder = OptimizedRounder().fit(cp, cy)
            cal_qwk = qwk(cy, rounder.predict(cp))
            suffix, tag = f"calib_resample_s{args.seed}", \
                f"resampled true-dist set (n={len(cy)}, seed={args.seed})"

        test_pred = booster.predict(X_test)
        test_labels = rounder.predict(test_pred)

        out = LIVE / f"submission_iter{it}_{suffix}.csv"
        pd.DataFrame({"essay_id": test["essay_id"], "score": test_labels}
                     ).to_csv(out, index=False)

        dist = (pd.Series(test_labels).value_counts(normalize=True)
                .sort_index() * 100).round(1).to_dict()
        print(f"=== iter {it} ({tag}) ===")
        print(f"  cut points     : {np.round(rounder.coef_, 3)}")
        print(f"  calib-set QWK  : {cal_qwk:.4f}")
        print(f"  test dist (%)  : {dist}")
        print(f"  submission     : {out.name}\n")
        summary.append((it, cal_qwk, out.name))

    print("================ SUMMARY (calibrated) ================")
    for it, q, name in summary:
        print(f"  iter {it:4d}  weighted_val_QWK={q:.4f}  ->  {name}")


if __name__ == "__main__":
    main()
