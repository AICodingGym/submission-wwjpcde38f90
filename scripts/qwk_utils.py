"""Quadratic Weighted Kappa helpers shared by every model.

The competition metric is QWK. Because QWK rewards ordinal closeness rather
than exact-class accuracy, the winning recipe is: predict a *continuous* score,
then learn the 5 cut points that turn it into integers 1..6 so as to maximise
QWK on out-of-fold predictions. `OptimizedRounder` does exactly that.
"""

from __future__ import annotations

import numpy as np
from functools import partial
from sklearn.metrics import cohen_kappa_score
import scipy.optimize as opt


def qwk(y_true, y_pred, sample_weight=None) -> float:
    """Quadratic weighted kappa between two integer label arrays.

    `sample_weight` lets you evaluate QWK as if the sample followed a different
    (e.g. the true test) label distribution.
    """
    return cohen_kappa_score(np.asarray(y_true).round().astype(int),
                             np.asarray(y_pred).round().astype(int),
                             weights="quadratic", sample_weight=sample_weight)


class OptimizedRounder:
    """Learn ordinal cut points that map a continuous score to labels 1..6.

    Fit on OOF continuous predictions vs. true labels, then apply to test
    predictions. Uses Nelder-Mead to directly maximise QWK (the true metric),
    which is non-differentiable, so gradient methods don't apply.
    """

    def __init__(self, labels=(1, 2, 3, 4, 5, 6)):
        self.labels = list(labels)
        # initial cuts halfway between consecutive labels: 1.5, 2.5, ...
        self.coef_ = [l + 0.5 for l in self.labels[:-1]]

    def _digitize(self, X, coef):
        return np.digitize(X, sorted(coef)) + self.labels[0]

    def _loss(self, coef, X, y, w):
        return -qwk(y, self._digitize(X, coef), sample_weight=w)

    def fit(self, X, y, sample_weight=None):
        w = None if sample_weight is None else np.asarray(sample_weight, float)
        loss = partial(self._loss, X=np.asarray(X, float),
                       y=np.asarray(y), w=w)
        res = opt.minimize(loss, self.coef_, method="nelder-mead",
                           options={"maxiter": 2000, "xatol": 1e-4})
        self.coef_ = sorted(res.x)
        return self

    def predict(self, X):
        return self._digitize(np.asarray(X, float), self.coef_).astype(int)
