"""Hand-crafted essay features derived from the Part-1 analysis.

The EDA showed the strongest score signals are, in order:
  word_count / char_count / unique_word_count / sentence_count / comma_count
  (all strongly positive), type_token_ratio & spell_err_ratio (negative), plus
  structure (paragraphs, commas-per-sentence) and word sophistication
  (avg_word_len, long_word_ratio, readability). These features feed the
  efficient LightGBM model and the ensemble's meta-features.
"""

from __future__ import annotations

import re
import string
import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words("english"))
WORD_RE = re.compile(r"[A-Za-z']+")
SENT_SPLIT = re.compile(r"[.!?]+")

# Readability via textstat is optional; guard the import so the module still
# works in a minimal environment.
try:
    import textstat
    _HAS_TEXTSTAT = True
except Exception:  # pragma: no cover
    _HAS_TEXTSTAT = False

# Spelling via pyspellchecker is optional and comparatively slow.
try:
    from spellchecker import SpellChecker
    _SPELL = SpellChecker(distance=1)
except Exception:  # pragma: no cover
    _SPELL = None


def _spell_errors(tokens: list[str]) -> int:
    if _SPELL is None:
        return 0
    lowered = [t.lower() for t in tokens if t.isalpha() and len(t) > 2]
    if not lowered:
        return 0
    return len(_SPELL.unknown(lowered))


def extract_features(text: str, with_spelling: bool = True) -> dict:
    tokens = WORD_RE.findall(text)
    n_words = len(tokens)
    n_chars = len(text)
    sentences = [s for s in SENT_SPLIT.split(text) if s.strip()]
    n_sent = max(len(sentences), 1)
    paragraphs = [p for p in text.split("\n") if p.strip()]
    n_para = max(len(paragraphs), 1)

    lower = [t.lower() for t in tokens]
    unique = set(lower)
    content = [t for t in lower if t not in STOPWORDS]
    long_words = [t for t in lower if len(t) >= 7]

    feats = {
        "word_count": n_words,
        "char_count": n_chars,
        "sentence_count": n_sent,
        "paragraph_count": n_para,
        "avg_sentence_len": n_words / n_sent,
        "avg_word_len": (sum(len(t) for t in tokens) / n_words) if n_words else 0.0,
        "sent_per_para": n_sent / n_para,
        "comma_count": text.count(","),
        "comma_per_sent": text.count(",") / n_sent,
        "punct_ratio": sum(text.count(p) for p in string.punctuation) / max(n_chars, 1),
        "unique_word_count": len(unique),
        "type_token_ratio": len(unique) / n_words if n_words else 0.0,
        "long_word_ratio": len(long_words) / n_words if n_words else 0.0,
        "content_word_ratio": len(content) / n_words if n_words else 0.0,
        "stopword_ratio": (n_words - len(content)) / n_words if n_words else 0.0,
    }
    if with_spelling:
        n_err = _spell_errors(tokens)
        feats["spell_err_count"] = n_err
        feats["spell_err_ratio"] = n_err / n_words if n_words else 0.0
    if _HAS_TEXTSTAT:
        feats["flesch_reading_ease"] = textstat.flesch_reading_ease(text)
        feats["flesch_kincaid_grade"] = textstat.flesch_kincaid_grade(text)
        feats["gunning_fog"] = textstat.gunning_fog(text)
        feats["smog_index"] = textstat.smog_index(text)
    return feats


def build_feature_frame(texts: pd.Series, with_spelling: bool = True) -> pd.DataFrame:
    rows = [extract_features(t, with_spelling=with_spelling) for t in texts]
    return pd.DataFrame(rows).astype(np.float32)
