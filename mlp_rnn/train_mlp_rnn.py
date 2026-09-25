"""
MLP / GRU / LSTM 사기거래 탐지 학습 스크립트 (BankSim)
팀원 tcn_transformer/train.py 와 동일한 학습·평가 프로토콜을 따른다.
  - seed 42, batch 2048, 최대 50 epochs + 조기 종료(patience 5, 검증 PR-AUC 기준),
    Adam(lr 1e-3, weight_decay 1e-5)
  - BCEWithLogitsLoss(pos_weight = 정상/사기 비율, train 기준)
  - 매 에포크 검증 PR-AUC 최고 시점의 가중치로 테스트 평가
  - 분류 임계값 0.5 고정
  - dropout 0.1, 은닉 크기 32 (TCN/Transformer 와 동일 규모)

입력 데이터: data/processed/banksim/seq_L{5,10,20}/{train,val,test}.npz
  X: (N, L, 9), is_padding: (N, L) 1=padding, y: (N,)

사용법 (이 파일은 프로젝트 폴더의 mlp_rnn/ 폴더에 둠)
  터미널:   python mlp_rnn/train_mlp_rnn.py
  주피터:   %run train_mlp_rnn.py            (노트북이 mlp_rnn 폴더에 있을 때)
  옵션 예:  --models gru lstm --L 10

결과: mlp_rnn/results/summary.csv, {model}_L{L}.json
"""
import argparse
import functools
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

print = functools.partial(print, flush=True)  # 주피터에서 바로바로 출력

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(HERE, "..", "data", "processed", "banksim")
DEFAULT_OUT = os.path.join(HERE, "results")

# ---------------------------------------------------------------------------
# 팀원 train.py 와 동일하게 고정하는 하이퍼파라미터 (공정 비교)
# ---------------------------------------------------------------------------
SEED = 42
BATCH_SIZE = 2048
MAX_EPOCHS = 50
PATIENCE = 5      # 검증 PR-AUC가 5 에포크 연속 개선 없으면 조기 종료
LR = 1e-3
WEIGHT_DECAY = 1e-5
DECISION_THRESHOLD = 0.5
HIDDEN = 32
DROPOUT = 0.1
RNN_LAYERS = 2


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_split(path):
    d = np.load(path, allow_pickle=True)
    X = torch.from_numpy(d["X"].astype(np.float32))
    pad = torch.from_numpy(d["is_padding"].astype(np.float32))
    y = torch.from_numpy(d["y"].astype(np.float32))
    return X, pad, y


# ---------------------------------------------------------------------------
# 모델
# ---------------------------------------------------------------------------
class MLPBaseline(nn.Module):
    """Transaction-level baseline: 현재 시점(t) 거래의 특징 9개만 사용 (과거 이력 없음).
    팀원 models.py 의 MLPBaseline 과 동일한 구조."""

    def __init__(self, n_features=9, hidden=HIDDEN, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, is_padding=None):
        return self.net(x[:, -1, :]).squeeze(-1)   # 마지막 시점 = 현재 거래


class RNNClassifier(nn.Module):
    """GRU / LSTM. 마지막 시점(t, 현재 거래)의 은닉 상태로 분류.
    분류기 구조는 TCN/Transformer 와 동일: FC(32) -> ReLU -> Dropout -> FC(1)."""

    def __init__(self, n_features=9, kind="gru", hidden=HIDDEN,
                 layers=RNN_LAYERS, dropout=DROPOUT):
        super().__init__()
        rnn_cls = nn.GRU if kind == "gru" else nn.LSTM
        self.rnn = rnn_cls(n_features, hidden, num_layers=layers,
                           batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x, is_padding):
        x = x * (1.0 - is_padding).unsqueeze(-1)    # padding 위치 0으로
        out, _ = self.rnn(x)                        # (B, L, H)
        return self.head(out[:, -1, :]).squeeze(-1)  # 마지막 시점(t)은 항상 실제 거래


def build_model(name, n_features):
    if name == "mlp":
        return MLPBaseline(n_features)
    return RNNClassifier(n_features, kind=name)


# ---------------------------------------------------------------------------
# 학습 / 평가 (팀원 train.py 와 동일한 방식)
# ---------------------------------------------------------------------------
def run_epoch(model, loader, optimizer, criterion, device, train=True):
    model.train() if train else model.eval()
    total_loss, n = 0.0, 0
    all_probs, all_y = [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, pad, y in loader:
            x, pad, y = x.to(device), pad.to(device), y.to(device)
            logit = model(x, pad)
            loss = criterion(logit, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            n += len(y)
            all_probs.append(torch.sigmoid(logit).detach().cpu().numpy())
            all_y.append(y.detach().cpu().numpy())
    return total_loss / n, np.concatenate(all_probs), np.concatenate(all_y)


def compute_metrics(y_true, y_prob, threshold=DECISION_THRESHOLD):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
    }


def run(name, L, data_dir, out_dir, device):
    set_seed(SEED)
    seq_dir = os.path.join(data_dir, f"seq_L{L}")
    Xtr, Ptr, ytr = load_split(os.path.join(seq_dir, "train.npz"))
    Xva, Pva, yva = load_split(os.path.join(seq_dir, "val.npz"))
    Xte, Pte, yte = load_split(os.path.join(seq_dir, "test.npz"))

    g = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(TensorDataset(Xtr, Ptr, ytr), batch_size=BATCH_SIZE,
                              shuffle=True, generator=g)
    val_loader = DataLoader(TensorDataset(Xva, Pva, yva), batch_size=BATCH_SIZE)
    test_loader = DataLoader(TensorDataset(Xte, Pte, yte), batch_size=BATCH_SIZE)

    model = build_model(name, Xtr.shape[2]).to(device)
    n_pos = float(ytr.sum())
    n_neg = len(ytr) - n_pos
    pos_weight = torch.tensor(n_neg / max(n_pos, 1.0), dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    tag = f"{name}_L{L}"
    print(f"\n===== {tag} | train {len(ytr):,} (사기 {int(n_pos):,}) | "
          f"val {len(yva):,} | test {len(yte):,} | device {device} =====")

    history, best_val_prauc, best_state, best_epoch, bad = [], -1.0, None, 0, 0
    t0 = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        te = time.time()
        train_loss, _, _ = run_epoch(model, train_loader, optimizer, criterion, device, True)
        val_loss, val_probs, val_y = run_epoch(model, val_loader, optimizer, criterion, device, False)
        vm = compute_metrics(val_y, val_probs)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **vm})
        print(f"[{tag}] epoch {epoch}/{MAX_EPOCHS} train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_recall={vm['recall']:.4f} "
              f"val_pr_auc={vm['pr_auc']:.4f} | {time.time() - te:.1f}s")
        if vm["pr_auc"] > best_val_prauc:
            best_val_prauc, best_epoch, bad = vm["pr_auc"], epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"[{tag}] early stopping (best epoch {best_epoch})")
                break
    elapsed = time.time() - t0

    model.load_state_dict(best_state)
    _, test_probs, test_y = run_epoch(model, test_loader, optimizer, criterion, device, False)
    tm = compute_metrics(test_y, test_probs)
    print(f"[{tag}] TEST acc={tm['accuracy']:.4f} P={tm['precision']:.4f} "
          f"R={tm['recall']:.4f} F1={tm['f1']:.4f} ROC-AUC={tm['roc_auc']:.4f} "
          f"PR-AUC={tm['pr_auc']:.4f} | best epoch {best_epoch}/{epoch} (elapsed {elapsed:.1f}s)")

    result = {
        "model": name, "L": L,
        "hyperparams": {"seed": SEED, "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS,
                        "patience": PATIENCE,
                        "lr": LR, "weight_decay": WEIGHT_DECAY,
                        "pos_weight": float(pos_weight), "decision_threshold": DECISION_THRESHOLD,
                        "hidden": HIDDEN, "dropout": DROPOUT,
                        "rnn_layers": RNN_LAYERS if name != "mlp" else None},
        "best_epoch": best_epoch, "epochs_run": epoch, "best_val_pr_auc": best_val_prauc,
        "history": history, "test_metrics": tm, "elapsed_sec": elapsed,
    }
    torch.save(best_state, os.path.join(out_dir, f"{tag}.pt"))
    with open(os.path.join(out_dir, f"{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=DEFAULT_DATA)
    ap.add_argument("--out_dir", default=DEFAULT_OUT)
    ap.add_argument("--models", nargs="+", default=["mlp", "gru", "lstm"],
                    choices=["mlp", "gru", "lstm"])
    ap.add_argument("--L", nargs="+", type=int, default=[5, 10, 20])
    args, _ = ap.parse_known_args()   # 주피터(%run)에서도 오류 없이 동작

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU 있으면 GPU 사용

    results = []
    for L in args.L:
        for name in args.models:
            results.append(run(name, L, args.data_dir, args.out_dir, device))

    cols = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    with open(os.path.join(args.out_dir, "summary.csv"), "w", encoding="utf-8-sig") as f:
        f.write("model,L," + ",".join(cols) + ",best_epoch,epochs_run,elapsed_sec\n")
        for r in results:
            f.write(f"{r['model']},{r['L']}," +
                    ",".join(f"{r['test_metrics'][c]:.4f}" for c in cols) +
                    f",{r['best_epoch']},{r['epochs_run']},{r['elapsed_sec']:.1f}\n")

    print("\n===== 요약 (TEST, threshold 0.5) =====")
    print(f"{'model':<6}{'L':>4}{'Acc':>8}{'P':>8}{'R':>8}{'F1':>8}{'ROC-AUC':>10}{'PR-AUC':>9}{'best_ep':>9}")
    for r in results:
        m = r["test_metrics"]
        print(f"{r['model']:<6}{r['L']:>4}{m['accuracy']:>8.4f}{m['precision']:>8.4f}"
              f"{m['recall']:>8.4f}{m['f1']:>8.4f}{m['roc_auc']:>10.4f}{m['pr_auc']:>9.4f}"
              f"{r['best_epoch']:>5}/{r['epochs_run']}")
    print(f"\n저장 위치: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()