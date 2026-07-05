"""Part 1: Identify factors that drive the essay score.

Samples essays from every score band (1-6) and computes interpretable
features across four families:
  - length      : how much the student wrote
  - structure   : paragraphing, sentence organisation, punctuation
  - vocabulary  : lexical richness / word sophistication
  - spelling    : error rate (proxy for surface correctness)
  - readability : textstat readability indices

It then reports:
  - per-band mean of each feature (does it move monotonically with score?)
  - Pearson & Spearman correlation of each feature with score
  - LightGBM feature importance (which features a model actually uses)

Outputs a CSV summary and a PNG heatmap under scripts/eda_out/.
"""

from __future__ import annotations

import re
import string
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA = Path(__file__).resolve().parent.parent / "data"
OUT = Path(__file__).resolve().parent / "eda_out"
OUT.mkdir(exist_ok=True)

PER_BAND = 400          # essays sampled per score band (capped by availability)
SEED = 42

# ---------------------------------------------------------------------------
# NLP resources
# ---------------------------------------------------------------------------
import nltk
from nltk.corpus import stopwords, words as nltk_words

STOPWORDS = set(stopwords.words("english"))
ENGLISH_VOCAB = set(w.lower() for w in nltk_words.words())

from spellchecker import SpellChecker
import textstat

SPELL = SpellChecker(distance=1)          # distance=1 -> much faster

WORD_RE = re.compile(r"[A-Za-z']+")
SENT_SPLIT = re.compile(r"[.!?]+")


def tokenize_words(text: str) -> list[str]:
    return WORD_RE.findall(text)


def count_spelling_errors(tokens: list[str]) -> int:
    """Words unknown to both the spellchecker and the nltk English vocab."""
    lowered = [t.lower() for t in tokens if t.isalpha() and len(t) > 2]
    if not lowered:
        return 0
    unknown = SPELL.unknown(lowered)
    # Keep only tokens that are also absent from the broader nltk vocabulary,
    # to reduce false positives on proper nouns / valid-but-rare words.
    real_unknown = [w for w in unknown if w not in ENGLISH_VOCAB]
    return len(real_unknown)


def extract_features(text: str) -> dict:
    tokens = tokenize_words(text)
    n_words = len(tokens)
    n_chars = len(text)

    sentences = [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]
    n_sent = max(len(sentences), 1)

    paragraphs = [p for p in text.split("\n") if p.strip()]
    n_para = max(len(paragraphs), 1)

    lower_tokens = [t.lower() for t in tokens]
    unique = set(lower_tokens)
    content_words = [t for t in lower_tokens if t not in STOPWORDS]
    long_words = [t for t in lower_tokens if len(t) >= 7]

    n_spell_err = count_spelling_errors(tokens)

    feats = {
        # ---- length ----
        "word_count": n_words,
        "char_count": n_chars,
        "sentence_count": n_sent,
        "paragraph_count": n_para,
        # ---- structure ----
        "avg_sentence_len": n_words / n_sent,
        "avg_word_len": (sum(len(t) for t in tokens) / n_words) if n_words else 0,
        "sent_per_para": n_sent / n_para,
        "comma_count": text.count(","),
        "comma_per_sent": text.count(",") / n_sent,
        "punct_ratio": sum(text.count(p) for p in string.punctuation) / max(n_chars, 1),
        # ---- vocabulary richness ----
        "unique_word_count": len(unique),
        "type_token_ratio": len(unique) / n_words if n_words else 0,
        "long_word_ratio": len(long_words) / n_words if n_words else 0,
        "content_word_ratio": len(content_words) / n_words if n_words else 0,
        "stopword_ratio": (n_words - len(content_words)) / n_words if n_words else 0,
        # ---- spelling / correctness ----
        "spell_err_count": n_spell_err,
        "spell_err_ratio": n_spell_err / n_words if n_words else 0,
        # ---- readability ----
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
        "gunning_fog": textstat.gunning_fog(text),
        "smog_index": textstat.smog_index(text),
    }
    return feats


def main() -> None:
    df = pd.read_csv(DATA / "train.csv")
    print(f"Loaded {len(df)} training essays.")
    print("Score distribution:\n", df["score"].value_counts().sort_index(), "\n")

    # Stratified sample: up to PER_BAND essays per score band.
    parts = []
    for s, grp in df.groupby("score"):
        n = min(PER_BAND, len(grp))
        parts.append(grp.sample(n=n, random_state=SEED))
    sample = pd.concat(parts).reset_index(drop=True)
    print(f"Sampled {len(sample)} essays across bands "
          f"({sample['score'].value_counts().sort_index().to_dict()}).\n")

    print("Extracting features ...")
    feat_rows = sample["full_text"].apply(extract_features)
    feats = pd.DataFrame(list(feat_rows))
    feats["score"] = sample["score"].values

    feats.to_csv(OUT / "sampled_features.csv", index=False)

    feature_cols = [c for c in feats.columns if c != "score"]

    # ---- per-band means ----
    band_means = feats.groupby("score")[feature_cols].mean().T
    band_means.to_csv(OUT / "feature_means_by_band.csv")

    # ---- correlations ----
    pear = feats[feature_cols].corrwith(feats["score"]).rename("pearson")
    spear = feats[feature_cols].corrwith(feats["score"], method="spearman").rename("spearman")
    corr = pd.concat([pear, spear], axis=1)
    corr["abs_spearman"] = corr["spearman"].abs()
    corr = corr.sort_values("abs_spearman", ascending=False)
    corr.to_csv(OUT / "feature_correlation.csv")

    # ---- LightGBM importance (gain) ----
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split

    X, y = feats[feature_cols], feats["score"]
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2,
                                           random_state=SEED, stratify=y)
    model = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03,
                              num_leaves=31, subsample=0.8,
                              colsample_bytree=0.8, random_state=SEED,
                              verbose=-1)
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)],
              callbacks=[lgb.early_stopping(40, verbose=False)])
    imp = pd.Series(model.booster_.feature_importance(importance_type="gain"),
                    index=feature_cols, name="lgb_gain").sort_values(ascending=False)
    imp.to_csv(OUT / "feature_importance.csv")

    # ---- report ----
    print("\n================ FEATURE CORRELATION WITH SCORE ================")
    print(corr.round(3).to_string())

    print("\n================ PER-BAND MEANS (score 1 -> 6) ================")
    print(band_means.round(2).to_string())

    print("\n================ LIGHTGBM FEATURE IMPORTANCE (gain) ================")
    print(imp.round(1).to_string())

    # ---- heatmap ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        norm = band_means.sub(band_means.min(axis=1), axis=0).div(
            band_means.max(axis=1) - band_means.min(axis=1) + 1e-9, axis=0)
        order = corr.index.tolist()
        norm = norm.loc[order]
        plt.figure(figsize=(9, 10))
        sns.heatmap(norm, annot=band_means.loc[order].round(1), fmt="g",
                    cmap="RdYlGn", cbar_kws={"label": "min-max normalised"})
        plt.title("Feature value by score band (1..6)\nannot = raw mean")
        plt.xlabel("score band")
        plt.tight_layout()
        plt.savefig(OUT / "feature_band_heatmap.png", dpi=130)
        print(f"\nHeatmap saved to {OUT / 'feature_band_heatmap.png'}")
    except Exception as e:  # pragma: no cover
        print("Plotting skipped:", e)

    print(f"\nAll CSV outputs in {OUT}")


if __name__ == "__main__":
    main()
