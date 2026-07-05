"""EFFICIENT model: TF-IDF + hand-crafted features -> LightGBM regression.

CPU-only, trains in a couple of minutes, targets the Efficiency track.
Pipeline:
  1. word (1-2gram) + char (3-5gram) TF-IDF  -> sparse lexical signal
  2. hand-crafted numeric features (see features.py) -> length/structure/vocab
  3. LightGBM regressor on the combined matrix, 5-fold StratifiedKFold
  4. OptimizedRounder learns QWK-optimal cut points on OOF predictions
Outputs: oof/test prediction arrays (for the ensemble) and submission.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

from qwk_utils import qwk, OptimizedRounder
from features import build_feature_frame

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = Path(__file__).resolve().parent / "model_out"
OUT.mkdir(exist_ok=True)

N_FOLDS = 5
SEED = 42


def build_matrix(train_text, test_text, use_spelling):
    word_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9,
                               sublinear_tf=True, max_features=40000,
                               strip_accents="unicode")
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                               min_df=3, sublinear_tf=True, max_features=40000)
    Xw_tr = word_vec.fit_transform(train_text)
    Xw_te = word_vec.transform(test_text)
    Xc_tr = char_vec.fit_transform(train_text)
    Xc_te = char_vec.transform(test_text)

    f_tr = build_feature_frame(train_text, with_spelling=use_spelling)
    f_te = build_feature_frame(test_text, with_spelling=use_spelling)
    # normalise dense features to keep them on a comparable scale
    mu, sd = f_tr.mean(), f_tr.std().replace(0, 1)
    f_tr = ((f_tr - mu) / sd).values
    f_te = ((f_te - mu) / sd).values

    X_tr = sp.hstack([Xw_tr, Xc_tr, sp.csr_matrix(f_tr)]).tocsr()
    X_te = sp.hstack([Xw_te, Xc_te, sp.csr_matrix(f_te)]).tocsr()
    return X_tr, X_te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-spelling", action="store_true",
                    help="Skip the (slow) spell-check feature for max speed.")
    args = ap.parse_args()

    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train["score"].values.astype(float)

    print("Vectorising ...")
    X, X_test = build_matrix(train["full_text"], test["full_text"],
                             use_spelling=not args.no_spelling)
    print("Feature matrix:", X.shape)

    params = dict(objective="regression", metric="rmse", learning_rate=0.05,
                  num_leaves=63, feature_fraction=0.6, bagging_fraction=0.8,
                  bagging_freq=1, min_child_samples=20, n_estimators=3000,
                  random_state=SEED, verbose=-1)

    oof = np.zeros(len(train))
    test_pred = np.zeros(len(test))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    for fold, (tr, va) in enumerate(skf.split(X, train["score"])):
        model = lgb.LGBMRegressor(**params)
        model.fit(X[tr], y[tr], eval_set=[(X[va], y[va])],
                  callbacks=[lgb.early_stopping(80, verbose=False)])
        oof[va] = model.predict(X[va])
        test_pred += model.predict(X_test) / N_FOLDS
        print(f"  fold {fold}: raw-QWK={qwk(y[va], oof[va]):.4f} "
              f"best_iter={model.best_iteration_}")

    rounder = OptimizedRounder().fit(oof, y)
    oof_labels = rounder.predict(oof)
    print(f"\nOOF QWK  (naive round): {qwk(y, oof.round()):.4f}")
    print(f"OOF QWK  (optimised)  : {qwk(y, oof_labels):.4f}")
    print("Learned cut points:", np.round(rounder.coef_, 3))

    test_labels = rounder.predict(test_pred)

    np.save(OUT / "efficient_oof.npy", oof)
    np.save(OUT / "efficient_test.npy", test_pred)
    sub = pd.DataFrame({"essay_id": test["essay_id"], "score": test_labels})
    sub.to_csv(OUT / "submission_efficient.csv", index=False)
    print(f"\nSaved submission -> {OUT / 'submission_efficient.csv'}")
    print(sub["score"].value_counts().sort_index())


if __name__ == "__main__":
    main()
