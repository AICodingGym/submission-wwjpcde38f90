"""ENSEMBLE: blend the efficient + accurate models, then optimise QWK cuts.

Reads the OOF / test prediction arrays each base model saved, searches for the
blend weight that maximises OOF QWK, learns the ordinal cut points on the
blended OOF prediction, and writes the final submission.csv.

Add more base models by dropping their `<name>_oof.npy` / `<name>_test.npy`
into model_out/ and listing them in BASE_MODELS.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from qwk_utils import qwk, OptimizedRounder

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = Path(__file__).resolve().parent / "model_out"

# name -> (oof file, test file). Missing files are skipped with a warning.
BASE_MODELS = {
    "efficient": ("efficient_oof.npy", "efficient_test.npy"),
    "accurate":  ("accurate_oof.npy", "accurate_test.npy"),
}


def load():
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train["score"].values.astype(float)
    oofs, tests, names = [], [], []
    for name, (o, t) in BASE_MODELS.items():
        op, tp = OUT / o, OUT / t
        if not (op.exists() and tp.exists()):
            print(f"[skip] {name}: missing {o}/{t}")
            continue
        oofs.append(np.load(op))
        tests.append(np.load(tp))
        names.append(name)
        print(f"[ok]   {name}: OOF QWK={qwk(y, np.load(op)):.4f}")
    if not oofs:
        raise SystemExit("No base-model predictions found. Train a model first.")
    return train, test, y, np.vstack(oofs), np.vstack(tests), names


def best_weights(oofs, y, step=0.05):
    """Grid-search convex blend weights (works well for 2-3 models)."""
    n = oofs.shape[0]
    if n == 1:
        return np.array([1.0])
    best_w, best_s = None, -1
    if n == 2:
        for w in np.arange(0, 1 + 1e-9, step):
            blend = w * oofs[0] + (1 - w) * oofs[1]
            s = qwk(y, OptimizedRounder().fit(blend, y).predict(blend))
            if s > best_s:
                best_s, best_w = s, np.array([w, 1 - w])
        return best_w
    # >=3 models: random simplex search
    rng = np.random.default_rng(0)
    for _ in range(4000):
        w = rng.dirichlet(np.ones(n))
        blend = (w[:, None] * oofs).sum(0)
        s = qwk(y, OptimizedRounder().fit(blend, y).predict(blend))
        if s > best_s:
            best_s, best_w = s, w
    return best_w


def main():
    train, test, y, oofs, tests, names = load()

    w = best_weights(oofs, y)
    print("\nBlend weights:", {n: round(float(wi), 3) for n, wi in zip(names, w)})

    blend_oof = (w[:, None] * oofs).sum(0)
    blend_test = (w[:, None] * tests).sum(0)

    rounder = OptimizedRounder().fit(blend_oof, y)
    print(f"Ensemble OOF QWK (optimised): {qwk(y, rounder.predict(blend_oof)):.4f}")
    print("Cut points:", np.round(rounder.coef_, 3))

    sub = pd.DataFrame({"essay_id": test["essay_id"],
                        "score": rounder.predict(blend_test)})
    sub.to_csv(OUT / "submission.csv", index=False)
    print("\nSaved final ensemble -> ", OUT / "submission.csv")
    print(sub["score"].value_counts().sort_index())


if __name__ == "__main__":
    main()
