"""Efficient LightGBM training with a LIVE dashboard, run until you stop it.

- Splits a balanced validation set (uniform across score bands + essay length).
- Trains a LightGBM regressor; boosting ROUNDS are the "steps".
- Every `--eval-interval` rounds it:
    * evaluates train & val RMSE and val QWK,
    * appends them to model_out/live/metrics.json (a web page polls this),
    * saves a checkpoint model_out/live/ckpt/model_iter{N}.txt.
- A tiny HTTP server serves the live dashboard at http://localhost:PORT/.
- Training keeps going up to --max-rounds. To STOP it gracefully at any time,
  create the stop file:   touch model_out/live/STOP
  (the current + all previous checkpoints remain on disk for you to pick from).

Usage:
    python train_efficient_live.py                 # fast (no spell feature)
    python train_efficient_live.py --spelling      # include spell-error feature
    python train_efficient_live.py --eval-interval 25 --port 8000
"""

from __future__ import annotations

import argparse
import json
import pickle
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
import lightgbm as lgb
from lightgbm.callback import EarlyStopException

from qwk_utils import qwk, OptimizedRounder
from features import build_feature_frame
from val_split import make_balanced_val, describe_split

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = Path(__file__).resolve().parent / "model_out" / "live"
CKPT = OUT / "ckpt"
SEED = 42


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-interval", type=int, default=25,
                    help="Evaluate + checkpoint every N boosting rounds.")
    ap.add_argument("--max-rounds", type=int, default=20000,
                    help="Upper bound on boosting rounds (stop earlier via STOP file).")
    ap.add_argument("--val-per-band", type=int, default=70)
    ap.add_argument("--max-frac", type=float, default=0.15,
                    help="Cap val at this fraction of each band (protects rare "
                         "bands, esp. band 6: 0.15*135 ~= 20 to val, ~115 to train).")
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--spelling", action="store_true",
                    help="Include the (slow) spell-error feature.")
    ap.add_argument("--port", type=int, default=8000)
    return ap.parse_args()


def build_matrix(train_text, test_text, use_spelling):
    word_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9,
                               sublinear_tf=True, max_features=40000,
                               strip_accents="unicode")
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                               min_df=3, sublinear_tf=True, max_features=40000)
    Xw = word_vec.fit_transform(train_text)
    Xw_te = word_vec.transform(test_text)
    Xc = char_vec.fit_transform(train_text)
    Xc_te = char_vec.transform(test_text)

    f_tr = build_feature_frame(train_text, with_spelling=use_spelling)
    f_te = build_feature_frame(test_text, with_spelling=use_spelling)
    mu, sd = f_tr.mean(), f_tr.std().replace(0, 1)
    f_trn = ((f_tr - mu) / sd).values
    f_ten = ((f_te - mu) / sd).values

    X = sp.hstack([Xw, Xc, sp.csr_matrix(f_trn)]).tocsr()
    X_te = sp.hstack([Xw_te, Xc_te, sp.csr_matrix(f_ten)]).tocsr()
    preproc = {"word_vec": word_vec, "char_vec": char_vec,
               "feat_mu": mu, "feat_sd": sd, "with_spelling": use_spelling}
    return X, X_te, preproc, f_tr["word_count"].values


def serve(directory: Path, port: int):
    handler = partial(SimpleHTTPRequestHandler, directory=str(directory))
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd


class LiveMonitor:
    """LightGBM callback: eval + checkpoint + write metrics every N rounds."""

    def __init__(self, interval, X_tr_eval, y_tr_eval, X_val, y_val,
                 out_dir: Path, ckpt_dir: Path):
        self.order = 30  # run late in the callback chain
        self.interval = interval
        self.X_tr_eval, self.y_tr_eval = X_tr_eval, y_tr_eval
        self.X_val, self.y_val = X_val, y_val
        self.out = out_dir
        self.ckpt = ckpt_dir
        self.metrics = {"status": "running", "iterations": [], "train_rmse": [],
                        "val_rmse": [], "val_qwk": [], "checkpoints": [],
                        "best": None, "updated": None}
        self.stop_file = out_dir / "STOP"

    def _rmse(self, pred, y):
        return float(np.sqrt(np.mean((pred - y) ** 2)))

    def _dump(self):
        self.metrics["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        (self.out / "metrics.json").write_text(json.dumps(self.metrics))

    def __call__(self, env):
        it = env.iteration + 1
        if it % self.interval != 0 and env.iteration != 0:
            return
        booster = env.model
        tr_pred = booster.predict(self.X_tr_eval, num_iteration=it)
        va_pred = booster.predict(self.X_val, num_iteration=it)
        tr_rmse = self._rmse(tr_pred, self.y_tr_eval)
        va_rmse = self._rmse(va_pred, self.y_val)
        va_qwk = qwk(self.y_val,
                     OptimizedRounder().fit(va_pred, self.y_val).predict(va_pred))

        self.metrics["iterations"].append(it)
        self.metrics["train_rmse"].append(round(tr_rmse, 5))
        self.metrics["val_rmse"].append(round(va_rmse, 5))
        self.metrics["val_qwk"].append(round(va_qwk, 5))

        booster.save_model(str(self.ckpt / f"model_iter{it}.txt"), num_iteration=it)
        self.metrics["checkpoints"].append(it)

        best_i = int(np.argmax(self.metrics["val_qwk"]))
        self.metrics["best"] = {
            "iter": self.metrics["iterations"][best_i],
            "val_qwk": self.metrics["val_qwk"][best_i],
            "val_rmse": self.metrics["val_rmse"][best_i],
        }
        self._dump()
        print(f"[iter {it:5d}] train_rmse={tr_rmse:.4f}  val_rmse={va_rmse:.4f}  "
              f"val_qwk={va_qwk:.4f}  (best qwk={self.metrics['best']['val_qwk']:.4f}"
              f" @ {self.metrics['best']['iter']})", flush=True)

        if self.stop_file.exists():
            self.metrics["status"] = "stopped"
            self._dump()
            print(f"\nSTOP file detected -> stopping at iter {it}. "
                  f"Checkpoints kept in {self.ckpt}", flush=True)
            raise EarlyStopException(env.iteration, va_rmse)


def write_dashboard(path: Path, port: int):
    html = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>AES 2.0 — LightGBM live training</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
 body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:24px;background:#0f1420;color:#e6e9ef}
 h1{font-size:20px;margin:0 0 4px} .sub{color:#8b93a7;font-size:13px;margin-bottom:18px}
 .cards{display:flex;gap:14px;margin-bottom:20px;flex-wrap:wrap}
 .card{background:#1a2130;border:1px solid #2a3245;border-radius:10px;padding:14px 18px;min-width:150px}
 .card .k{color:#8b93a7;font-size:12px} .card .v{font-size:22px;font-weight:600;margin-top:4px}
 .dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}
 .run{background:#3fbf5f;box-shadow:0 0 8px #3fbf5f} .stop{background:#e0554e}
 .wrap{background:#1a2130;border:1px solid #2a3245;border-radius:10px;padding:16px;margin-bottom:18px}
 canvas{max-height:340px}
</style></head><body>
<h1>Automated Essay Scoring 2.0 — LightGBM live training</h1>
<div class="sub">boosting rounds as steps · metrics polled every 2s · to stop: <code>touch model_out/live/STOP</code></div>
<div class="cards">
 <div class="card"><div class="k">Status</div><div class="v" id="status">…</div></div>
 <div class="card"><div class="k">Round</div><div class="v" id="iter">–</div></div>
 <div class="card"><div class="k">Val QWK</div><div class="v" id="qwk">–</div></div>
 <div class="card"><div class="k">Best Val QWK</div><div class="v" id="best">–</div></div>
 <div class="card"><div class="k">Val RMSE</div><div class="v" id="rmse">–</div></div>
 <div class="card"><div class="k">Checkpoints</div><div class="v" id="ckpts">–</div></div>
</div>
<div class="wrap"><canvas id="lossChart"></canvas></div>
<div class="wrap"><canvas id="qwkChart"></canvas></div>
<script>
const ax={scales:{x:{ticks:{color:'#8b93a7'},grid:{color:'#222a3a'}},y:{ticks:{color:'#8b93a7'},grid:{color:'#222a3a'}}},plugins:{legend:{labels:{color:'#e6e9ef'}}},animation:false};
const loss=new Chart(document.getElementById('lossChart'),{type:'line',data:{labels:[],datasets:[
 {label:'train RMSE',data:[],borderColor:'#5b8cff',backgroundColor:'#5b8cff',pointRadius:0,tension:.2},
 {label:'val RMSE',data:[],borderColor:'#ff9f43',backgroundColor:'#ff9f43',pointRadius:0,tension:.2}]},
 options:{...ax,plugins:{...ax.plugins,title:{display:true,text:'Train vs Val loss (RMSE)',color:'#e6e9ef'}}}});
const qwkc=new Chart(document.getElementById('qwkChart'),{type:'line',data:{labels:[],datasets:[
 {label:'val QWK',data:[],borderColor:'#3fbf5f',backgroundColor:'#3fbf5f',pointRadius:0,tension:.2}]},
 options:{...ax,plugins:{...ax.plugins,title:{display:true,text:'Validation QWK (the competition metric)',color:'#e6e9ef'}}}});
async function tick(){
 try{
  const m=await (await fetch('metrics.json?_='+Date.now())).json();
  loss.data.labels=m.iterations; loss.data.datasets[0].data=m.train_rmse; loss.data.datasets[1].data=m.val_rmse; loss.update();
  qwkc.data.labels=m.iterations; qwkc.data.datasets[0].data=m.val_qwk; qwkc.update();
  const running=m.status==='running';
  document.getElementById('status').innerHTML='<span class="dot '+(running?'run':'stop')+'"></span>'+m.status;
  const n=m.iterations.length;
  document.getElementById('iter').textContent=n?m.iterations[n-1]:'–';
  document.getElementById('qwk').textContent=n?m.val_qwk[n-1].toFixed(4):'–';
  document.getElementById('rmse').textContent=n?m.val_rmse[n-1].toFixed(4):'–';
  document.getElementById('best').textContent=m.best?m.best.val_qwk.toFixed(4)+' @'+m.best.iter:'–';
  document.getElementById('ckpts').textContent=m.checkpoints?m.checkpoints.length:'–';
 }catch(e){document.getElementById('status').textContent='waiting for metrics…';}
}
setInterval(tick,2000); tick();
</script></body></html>"""
    path.write_text(html)


def main():
    args = get_args()
    OUT.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    # clear any stale stop file / metrics from a previous run
    (OUT / "STOP").unlink(missing_ok=True)

    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train["score"].values.astype(float)

    print("Vectorising (this takes a bit) ...", flush=True)
    X, X_test, preproc, word_count = build_matrix(
        train["full_text"], test["full_text"], use_spelling=args.spelling)
    pickle.dump(preproc, open(OUT / "preprocessor.pkl", "wb"))
    print("Feature matrix:", X.shape, flush=True)

    val_mask = make_balanced_val(train, word_count, val_per_band=args.val_per_band,
                                 max_frac=args.max_frac)
    print("\nBalanced validation split:")
    print(describe_split(train, word_count, val_mask), "\n")
    np.save(OUT / "val_mask.npy", val_mask)

    tr_idx = np.where(~val_mask)[0]
    va_idx = np.where(val_mask)[0]
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_va, y_va = X[va_idx], y[va_idx]

    # train-eval subsample (same size as val) for a comparable train-loss curve
    rng = np.random.default_rng(SEED)
    sub = rng.choice(len(tr_idx), size=min(len(va_idx), len(tr_idx)), replace=False)
    X_tr_eval, y_tr_eval = X_tr[sub], y_tr[sub]

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    params = dict(objective="regression", metric="rmse",
                  learning_rate=args.learning_rate, num_leaves=63,
                  feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1,
                  min_child_samples=20, seed=SEED, verbose=-1)

    monitor = LiveMonitor(args.eval_interval, X_tr_eval, y_tr_eval,
                          X_va, y_va, OUT, CKPT)
    write_dashboard(OUT / "dashboard.html", args.port)
    serve(OUT, args.port)
    print(f"\n>>> Live dashboard: http://localhost:{args.port}/dashboard.html")
    print(f">>> To stop training: touch {OUT / 'STOP'}\n", flush=True)

    try:
        lgb.train(params, dtrain, num_boost_round=args.max_rounds,
                  callbacks=[monitor])
    except EarlyStopException:
        pass

    if monitor.metrics["status"] == "running":
        monitor.metrics["status"] = "stopped"
        monitor._dump()
    print("\nTraining finished.")
    if monitor.metrics["best"]:
        b = monitor.metrics["best"]
        print(f"Best val QWK = {b['val_qwk']} at iter {b['iter']} "
              f"(checkpoint: {CKPT / ('model_iter%d.txt' % b['iter'])})")
    print(f"All checkpoints in: {CKPT}")
    # keep the process alive so the dashboard stays served
    print("\nDashboard still live. Press Ctrl-C to exit and stop serving.")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
