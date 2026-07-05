"""Generate a submission from one or more saved LightGBM checkpoints.

For each requested boosting iteration it:
  - rebuilds the exact feature matrix (saved TF-IDF vectorizers + hand-crafted
    features, normalised with the saved train mean/std),
  - loads the checkpoint booster,
  - fits QWK-optimal cut points on the held-out validation set,
  - applies them to the test predictions,
  - writes submission_iter{N}.csv and reports the validation QWK.

Usage:
    python predict_from_ckpt.py --iters 350 700
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import lightgbm as lgb

from qwk_utils import qwk, OptimizedRounder
from features import build_feature_frame

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
LIVE = Path(__file__).resolve().parent / "model_out" / "live"
CKPT = LIVE / "ckpt"


def make_matrix(texts, pre):
    """Rebuild [word-tfidf | char-tfidf | normalised dense features]."""
    Xw = pre["word_vec"].transform(texts)
    Xc = pre["char_vec"].transform(texts)
    f = build_feature_frame(texts, with_spelling=pre["with_spelling"])
    f = f.reindex(columns=pre["feat_mu"].index)  # exact train column order
    fn = ((f - pre["feat_mu"]) / pre["feat_sd"]).values
    return sp.hstack([Xw, Xc, sp.csr_matrix(fn)]).tocsr()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, nargs="+", default=[350, 700])
    args = ap.parse_args()

    pre = pickle.load(open(LIVE / "preprocessor.pkl", "rb"))
    val_mask = np.load(LIVE / "val_mask.npy")
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train["score"].values.astype(float)

    val_texts = train["full_text"].values[val_mask]
    y_val = y[val_mask]

    print("Building validation + test feature matrices ...")
    X_val = make_matrix(val_texts, pre)
    X_test = make_matrix(test["full_text"].values, pre)

    summary = []
    for it in args.iters:
        path = CKPT / f"model_iter{it}.txt"
        if not path.exists():
            print(f"[skip] no checkpoint for iter {it}")
            continue
        booster = lgb.Booster(model_file=str(path))

        val_pred = booster.predict(X_val)
        rounder = OptimizedRounder().fit(val_pred, y_val)
        val_qwk = qwk(y_val, rounder.predict(val_pred))

        test_pred = booster.predict(X_test)
        test_labels = rounder.predict(test_pred)

        out = LIVE / f"submission_iter{it}.csv"
        pd.DataFrame({"essay_id": test["essay_id"], "score": test_labels}
                     ).to_csv(out, index=False)

        dist = pd.Series(test_labels).value_counts().sort_index().to_dict()
        print(f"\n=== iter {it} ===")
        print(f"  val QWK        : {val_qwk:.4f}")
        print(f"  cut points     : {np.round(rounder.coef_, 3)}")
        print(f"  test score dist: {dist}")
        print(f"  submission     : {out}")
        summary.append((it, val_qwk, out.name))

    print("\n================ SUMMARY ================")
    for it, q, name in summary:
        print(f"  iter {it:4d}  val_QWK={q:.4f}  ->  {name}")


if __name__ == "__main__":
    main()
