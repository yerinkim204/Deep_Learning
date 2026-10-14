"""
5개 모델(MLP / GRU / LSTM / TCN / Transformer) 하이퍼파라미터 탐색 + 최종 학습

[1단계: search]  L=10 데이터로 모델마다 아래 격자의 모든 조합을 학습하고
                 검증 PR-AUC가 가장 높은 조합을 고른다. (테스트 데이터는 사용 안 함)
    Learning rate : {0.0005, 0.001, 0.005}
    Hidden size   : {16, 32, 64}   (MLP 은닉 유닛 / GRU·LSTM 은닉 상태 /
                                    TCN 채널 / Transformer d_model, FF=2×d_model)
    Dropout       : {0.1, 0.2, 0.3}
    -> 모델당 27조합, 탐색 시 최대 10 epochs + 조기 종료(patience 3)

[2단계: final]   고른 조합으로 L=5, 10, 20 각각 최대 50 epochs + 조기 종료(patience 5)
                 학습 후 테스트 성능 측정 (임계값 0.5)

고정: seed 42, batch 2048, Adam, weight decay 1e-5, Weighted BCE(pos_weight=N_neg/N_pos)

폴더 구조 (~/fraud 기준)
  fraud/
  ├── hparam_search.py          <- 이 파일
  ├── data/processed/banksim/seq_L{5,10,20}/{train,val,test}.npz
  └── tcn_transformer/models.py  (팀원 TCN/Transformer 정의를 그대로 import)

실행
  python hparam_search.py                 # 탐색 + 최종 학습 전부
  python hparam_search.py --stage search  # 탐색만
  python hparam_search.py --stage final   # 탐색 결과(best_config.json)로 최종 학습만
중간에 멈춰도 다시 실행하면 끝난 조합은 건너뛰고 이어서 진행한다.

결과 (hparam_results/)
  search_results.csv   모든 조합의 검증 PR-AUC
  best_config.json     모델별 최적 조합
  final_summary.csv    최종 테스트 성능 (논문 결과표용)
"""
import argparse
import csv
import functools
import itertools
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

print = functools.partial(print, flush=True)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "tcn_transformer"))
from models import TCNClassifier, TransformerClassifier  # 팀원 모델 그대로 사용

DATA_DIR = os.path.join(HERE, "data", "processed", "banksim")
OUT_DIR = os.path.join(HERE, "hparam_results")

# ---------------------------------------------------------------------------
# 탐색 범위 / 고정값
# ---------------------------------------------------------------------------
MODELS = ["mlp", "gru", "lstm", "tcn", "transformer", "mlp_tx"]

# mlp_tx: 과거 이력 집계 feature를 뺀 transaction-level baseline.
# npz X의 9개 열 중 (category, amount_zscore, sin_dow, cos_dow)의 인덱스로 수정할 것.
# -> check_features.py 로 열 순서 확인
TX_FEATURE_IDX = [0, 1, 2, 3]
GRID = {
    "lr": [0.0005, 0.001, 0.005],
    "hidden": [16, 32, 64],
    "dropout": [0.1, 0.2, 0.3],
}
SEARCH_L = 10
SEARCH_MAX_EPOCHS, SEARCH_PATIENCE = 10, 3
FINAL_LS = [5, 10, 20]
FINAL_MAX_EPOCHS, FINAL_PATIENCE = 50, 5

SEED = 42
BATCH_SIZE = 2048
WEIGHT_DECAY = 1e-5
THRESHOLD = 0.5
RNN_LAYERS = 2


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# 모델 (MLP/GRU/LSTM 은 train_mlp_rnn.py 와 동일 구조, 크기만 파라미터화)
# ---------------------------------------------------------------------------
class MLPBaseline(nn.Module):
    """현재 거래(t)의 특징만 사용. feature_idx가 주어지면 그 열만 사용 (mlp_tx)."""

    def __init__(self, n_features, hidden, dropout, feature_idx=None):
        super().__init__()
        idx = list(range(n_features)) if feature_idx is None else list(feature_idx)
        self.register_buffer("feature_idx", torch.tensor(idx, dtype=torch.long))
        self.net = nn.Sequential(
            nn.Linear(len(idx), hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, is_padding=None):
        return self.net(x[:, -1, :][:, self.feature_idx]).squeeze(-1)


class RNNClassifier(nn.Module):
    def __init__(self, n_features, kind, hidden, dropout, layers=RNN_LAYERS):
        super().__init__()
        rnn_cls = nn.GRU if kind == "gru" else nn.LSTM
        self.rnn = rnn_cls(n_features, hidden, num_layers=layers,
                           batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x, is_padding):
        x = x * (1.0 - is_padding).unsqueeze(-1)
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def build_model(name, n_features, hidden, dropout):
    if name == "mlp":
        return MLPBaseline(n_features, hidden, dropout)
    if name == "mlp_tx":
        return MLPBaseline(n_features, hidden, dropout, feature_idx=TX_FEATURE_IDX)
    if name in ("gru", "lstm"):
        return RNNClassifier(n_features, name, hidden, dropout)
    if name == "tcn":
        return TCNClassifier(n_features=n_features, channels=(hidden,) * 3,
                             kernel_size=3, dropout=dropout)
    if name == "transformer":
        return TransformerClassifier(n_features=n_features, d_model=hidden, nhead=4,
                                     num_layers=2, dim_ff=hidden * 2, dropout=dropout)
    raise ValueError(name)


# ---------------------------------------------------------------------------
# 데이터 / 학습
# ---------------------------------------------------------------------------
_cache = {}


def load_L(L):
    if L not in _cache:
        out = {}
        for split in ("train", "val", "test"):
            d = np.load(os.path.join(DATA_DIR, f"seq_L{L}", f"{split}.npz"), allow_pickle=True)
            out[split] = (torch.from_numpy(d["X"].astype(np.float32)),
                          torch.from_numpy(d["is_padding"].astype(np.float32)),
                          torch.from_numpy(d["y"].astype(np.float32)))
        _cache.clear()
        _cache[L] = out
    return _cache[L]


def run_epoch(model, loader, optimizer, criterion, device, train):
    model.train() if train else model.eval()
    total, n, probs, ys = 0.0, 0, [], []
    with torch.set_grad_enabled(train):
        for x, pad, y in loader:
            x, pad, y = x.to(device), pad.to(device), y.to(device)
            logit = model(x, pad)
            loss = criterion(logit, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * len(y)
            n += len(y)
            probs.append(torch.sigmoid(logit).detach().cpu().numpy())
            ys.append(y.cpu().numpy())
    return total / n, np.concatenate(probs), np.concatenate(ys)


def metrics(y, p):
    pred = (p >= THRESHOLD).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
    }


def train_one(name, L, cfg, max_epochs, patience, device, evaluate_test, tag):
    set_seed(SEED)
    data = load_L(L)
    Xtr, Ptr, ytr = data["train"]
    g = torch.Generator().manual_seed(SEED)
    tr = DataLoader(TensorDataset(Xtr, Ptr, ytr), batch_size=BATCH_SIZE, shuffle=True, generator=g)
    va = DataLoader(TensorDataset(*data["val"]), batch_size=BATCH_SIZE)

    model = build_model(name, Xtr.shape[2], cfg["hidden"], cfg["dropout"]).to(device)
    n_pos = float(ytr.sum())
    pos_weight = torch.tensor((len(ytr) - n_pos) / max(n_pos, 1.0), device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=WEIGHT_DECAY)

    best, best_state, best_ep, bad, t0 = -1.0, None, 0, 0, time.time()
    for ep in range(1, max_epochs + 1):
        run_epoch(model, tr, optimizer, criterion, device, True)
        _, p, y = run_epoch(model, va, optimizer, criterion, device, False)
        val_pr = float(average_precision_score(y, p))
        if val_pr > best:
            best, best_ep, bad = val_pr, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    res = {"model": name, "L": L, **cfg, "val_pr_auc": best, "best_epoch": best_ep,
           "epochs_run": ep, "elapsed_sec": round(time.time() - t0, 1)}
    if evaluate_test:
        model.load_state_dict(best_state)
        te = DataLoader(TensorDataset(*data["test"]), batch_size=BATCH_SIZE)
        _, p, y = run_epoch(model, te, optimizer, criterion, device, False)
        res.update({f"test_{k}": v for k, v in metrics(y, p).items()})
        torch.save(best_state, os.path.join(OUT_DIR, f"final_{tag}.pt"))
    print(f"[{tag}] lr={cfg['lr']} hidden={cfg['hidden']} dropout={cfg['dropout']} "
          f"| val PR-AUC {best:.4f} (best ep {best_ep}/{ep}, {res['elapsed_sec']}s)"
          + (f" | TEST PR-AUC {res['test_pr_auc']:.4f} F1 {res['test_f1']:.4f}" if evaluate_test else ""))
    return res


def load_done(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_row(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


# ---------------------------------------------------------------------------
# 1단계: 탐색
# ---------------------------------------------------------------------------
def stage_search(device):
    path = os.path.join(OUT_DIR, "search_results.csv")
    done = {(r["model"], float(r["lr"]), int(r["hidden"]), float(r["dropout"])) for r in load_done(path)}
    combos = list(itertools.product(GRID["lr"], GRID["hidden"], GRID["dropout"]))
    total = len(MODELS) * len(combos)
    i = 0
    for name in MODELS:
        for lr, hidden, dropout in combos:
            i += 1
            if (name, lr, hidden, dropout) in done:
                continue
            cfg = {"lr": lr, "hidden": hidden, "dropout": dropout}
            res = train_one(name, SEARCH_L, cfg, SEARCH_MAX_EPOCHS, SEARCH_PATIENCE,
                            device, False, f"search {i}/{total} {name}")
            append_row(path, res)

    rows = load_done(path)
    bc_path = os.path.join(OUT_DIR, "best_config.json")
    best = json.load(open(bc_path, encoding="utf-8")) if os.path.exists(bc_path) else {}
    for name in MODELS:
        cand = [r for r in rows if r["model"] == name]
        b = max(cand, key=lambda r: float(r["val_pr_auc"]))
        best[name] = {"lr": float(b["lr"]), "hidden": int(b["hidden"]),
                      "dropout": float(b["dropout"]), "val_pr_auc": float(b["val_pr_auc"])}
    with open(os.path.join(OUT_DIR, "best_config.json"), "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    print("\n===== 탐색 결과: 모델별 최적 조합 (L=10, 검증 PR-AUC 기준) =====")
    print(f"{'model':<12}{'lr':>8}{'hidden':>8}{'dropout':>9}{'val PR-AUC':>12}")
    for name, b in best.items():
        print(f"{name:<12}{b['lr']:>8}{b['hidden']:>8}{b['dropout']:>9}{b['val_pr_auc']:>12.4f}")
    return best


# ---------------------------------------------------------------------------
# 2단계: 최종 학습 + 테스트
# ---------------------------------------------------------------------------
def stage_final(device):
    with open(os.path.join(OUT_DIR, "best_config.json"), encoding="utf-8") as f:
        best = json.load(f)
    path = os.path.join(OUT_DIR, "final_summary.csv")
    done = {(r["model"], int(r["L"])) for r in load_done(path)}
    for L in FINAL_LS:
        for name in MODELS:
            if (name, L) in done:
                continue
            cfg = {k: best[name][k] for k in ("lr", "hidden", "dropout")}
            res = train_one(name, L, cfg, FINAL_MAX_EPOCHS, FINAL_PATIENCE,
                            device, True, f"final {name}_L{L}")
            append_row(path, res)

    rows = load_done(path)
    print("\n===== 최종 결과 (TEST, threshold 0.5) =====")
    print(f"{'model':<12}{'L':>4}{'Acc':>8}{'P':>8}{'R':>8}{'F1':>8}{'ROC-AUC':>10}{'PR-AUC':>9}{'best_ep':>9}")
    for r in rows:
        print(f"{r['model']:<12}{r['L']:>4}{float(r['test_accuracy']):>8.4f}"
              f"{float(r['test_precision']):>8.4f}{float(r['test_recall']):>8.4f}"
              f"{float(r['test_f1']):>8.4f}{float(r['test_roc_auc']):>10.4f}"
              f"{float(r['test_pr_auc']):>9.4f}{r['best_epoch']:>5}/{r['epochs_run']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["all", "search", "final"], default="all")
    ap.add_argument("--models", nargs="*", default=None,
                    help="이 모델들만 실행 (예: --models mlp_tx). 생략하면 전체(이미 끝난 건 건너뜀)")
    args, _ = ap.parse_known_args()
    if args.models:
        MODELS[:] = [m for m in MODELS if m in args.models]
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if args.stage in ("all", "search"):
        stage_search(device)
    if args.stage in ("all", "final"):
        stage_final(device)


if __name__ == "__main__":
    main()
