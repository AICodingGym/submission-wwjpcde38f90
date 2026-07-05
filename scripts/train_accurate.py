"""ACCURATE model: DeBERTa-v3 (or any HF encoder) fine-tuned as a REGRESSOR.

This is the main score driver. We frame scoring as regression (single output +
MSE loss) rather than 6-way classification, because QWK rewards ordinal
closeness and regression + learned cut points consistently beats softmax.

Tuned to train on a single 8GB GPU (e.g. RTX 4060):
  - defaults to deberta-v3-small (best public-score / cost trade-off on this task)
  - max_len=512, batch=8, grad_accum=2  (effective batch 16, ~4GB VRAM)
  - bf16 autocast on Ada GPUs (falls back to fp16, then fp32 on CPU)
  - optional gradient checkpointing for larger backbones

Requirements (install once, needs internet the first time to pull weights):
    pip install transformers accelerate sentencepiece

Examples:
    # small, fits easily on 4060 (~1.5h for 5 folds)
    python train_accurate.py --model microsoft/deberta-v3-small
    # base, still fits on 8GB
    python train_accurate.py --model microsoft/deberta-v3-base --batch-size 4 --grad-accum 4
    # large on 8GB needs checkpointing + tiny batch
    python train_accurate.py --model microsoft/deberta-v3-large \
        --batch-size 2 --grad-accum 8 --gradient-checkpointing --epochs 3
    # quick smoke test (1 fold, 1 epoch)
    python train_accurate.py --folds-to-run 1 --epochs 1

Outputs: accurate_oof.npy, accurate_test.npy, submission_accurate.csv
(shapes/format identical to train_efficient.py so the ensemble can combine them).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold

from qwk_utils import qwk, OptimizedRounder

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = Path(__file__).resolve().parent / "model_out"
OUT.mkdir(exist_ok=True)

SEED = 42


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/deberta-v3-small",
                    help="Any HF encoder (deberta-v3-small/base/large, roberta-large, ...).")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=512,
                    help="512 covers >90%% of essays; raise to 1024 for long-tail.")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--gradient-checkpointing", action="store_true",
                    help="Trade ~20%% speed for ~35%% less VRAM (needed for large on 8GB).")
    ap.add_argument("--folds-to-run", type=int, default=None,
                    help="Run only the first K folds (debugging / time budget).")
    return ap.parse_args()


class EssayDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer, max_len, targets=None):
        self.texts = list(texts)
        self.tok = tokenizer
        self.max_len = max_len
        self.targets = targets

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tok(self.texts[i], truncation=True, max_length=self.max_len,
                       padding="max_length", return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if self.targets is not None:
            item["labels"] = torch.tensor(self.targets[i], dtype=torch.float)
        return item


def main():
    args = get_args()
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              TrainingArguments, Trainer)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    # regression target normalised to help optimisation; de-normalise later
    y = train["score"].values.astype(np.float32)
    y_norm = (y - y.mean()) / y.std()
    y_mean, y_std = y.mean(), y.std()

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    except Exception:
        # DeBERTa-v3's fast tokenizer occasionally fails to build; fall back
        # to the slow SentencePiece tokenizer.
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds = preds.reshape(-1) * y_std + y_mean
        labels = labels.reshape(-1) * y_std + y_mean
        return {"qwk": qwk(labels, preds)}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Ada (40-series) supports bf16, which is numerically safer than fp16 for
    # regression. Prefer bf16 -> fp16 -> fp32.
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    use_fp16 = device == "cuda" and not use_bf16
    if device == "cuda":
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Device: {gpu} ({vram:.1f} GB)  precision={'bf16' if use_bf16 else 'fp16'}")
    else:
        print("Device: CPU (training a transformer on CPU is very slow; use a GPU).")

    oof = np.zeros(len(train), dtype=np.float32)
    test_pred = np.zeros(len(test), dtype=np.float32)
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)

    n_run = args.folds_to_run or args.folds
    test_ds = EssayDataset(test["full_text"], tokenizer, args.max_len)

    for fold, (tr, va) in enumerate(skf.split(train, train["score"])):
        if fold >= n_run:
            break
        print(f"\n===== Fold {fold} =====")
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model, num_labels=1)  # num_labels=1 -> regression head
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.config.use_cache = False

        train_ds = EssayDataset(train["full_text"].iloc[tr].tolist(),
                                tokenizer, args.max_len, y_norm[tr])
        val_ds = EssayDataset(train["full_text"].iloc[va].tolist(),
                              tokenizer, args.max_len, y_norm[va])

        targs = TrainingArguments(
            output_dir=str(OUT / f"ckpt_fold{fold}"),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size * 2,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            weight_decay=0.01,
            warmup_ratio=0.1,
            bf16=use_bf16,
            fp16=use_fp16,
            gradient_checkpointing=args.gradient_checkpointing,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="qwk",
            greater_is_better=True,
            lr_scheduler_type="cosine",
            report_to="none",
            save_total_limit=1,
            dataloader_num_workers=2,
        )
        trainer = Trainer(model=model, args=targs, train_dataset=train_ds,
                          eval_dataset=val_ds, compute_metrics=compute_metrics)
        trainer.train()

        va_pred = trainer.predict(val_ds).predictions.reshape(-1) * y_std + y_mean
        oof[va] = va_pred
        te_pred = trainer.predict(test_ds).predictions.reshape(-1) * y_std + y_mean
        test_pred += te_pred / n_run
        print(f"  fold {fold} QWK (raw round): {qwk(y[va], va_pred):.4f}")

        del model, trainer
        torch.cuda.empty_cache() if device == "cuda" else None

    rounder = OptimizedRounder().fit(oof, y)
    print(f"\nOOF QWK (naive round): {qwk(y, oof.round()):.4f}")
    print(f"OOF QWK (optimised)  : {qwk(y, rounder.predict(oof)):.4f}")

    np.save(OUT / "accurate_oof.npy", oof)
    np.save(OUT / "accurate_test.npy", test_pred)
    sub = pd.DataFrame({"essay_id": test["essay_id"],
                        "score": rounder.predict(test_pred)})
    sub.to_csv(OUT / "submission_accurate.csv", index=False)
    print("Saved submission ->", OUT / "submission_accurate.csv")


if __name__ == "__main__":
    main()
