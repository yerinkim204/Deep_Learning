"""
사용법:
  python train.py --L 5 --model tcn
  python train.py --L 10 --model transformer

하이퍼파라미터는 모든 L/모델 조합에서 동일하게 고정한다 (공정 비교 원칙).
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, accuracy_score, precision_recall_curve,
)

from dataset import SequenceFraudDataset
from models import TCNClassifier, TransformerClassifier

# ---------------------------------------------------------------------------
# 모든 L / 모델에 공통으로 고정하는 하이퍼파라미터 (공정 비교를 위해 절대 안 바꿈)
# ---------------------------------------------------------------------------
SEED = 42
BATCH_SIZE = 2048
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-5
DECISION_THRESHOLD = 0.5  # sigmoid(logit) >= 0.5 -> fraud 예측


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(model_name, n_features):
    if model_name == "tcn":
        return TCNClassifier(n_features=n_features, channels=(32, 32, 32), kernel_size=3, dropout=0.1)
    elif model_name == "transformer":
        return TransformerClassifier(n_features=n_features, d_model=32, nhead=4, num_layers=2, dim_ff=64, dropout=0.1)
    else:
        raise ValueError(model_name)


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
    probs = np.concatenate(all_probs)
    ys = np.concatenate(all_y)
    return total_loss / n, probs, ys


def compute_metrics(y_true, y_prob, threshold=DECISION_THRESHOLD):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
    }


def main(L, model_name, data_root="data", out_dir="results"):
    set_seed(SEED)
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cpu")

    train_ds = SequenceFraudDataset(f"{data_root}/seq_L{L}/train.npz")
    val_ds = SequenceFraudDataset(f"{data_root}/seq_L{L}/val.npz")
    test_ds = SequenceFraudDataset(f"{data_root}/seq_L{L}/test.npz")

    # --- 데이터 누수 점검: train/val/test 고객 ID가 시간상 어떻게 나뉘는지는 전처리 단계에서
    #     이미 절대 시간 기준으로 분리됨(README 5단계). 여기서는 스크립트 레벨에서
    #     "정규화 통계가 train으로만 fit됐는지"는 재현 불가하므로, 대신 각 split의 크기와
    #     fraud 비율이 README에 보고된 값과 일치하는지를 재확인해 파일 정합성을 검증한다.
    n_train, n_val, n_test = len(train_ds), len(val_ds), len(test_ds)
    fraud_rate = lambda ds: ds.y.mean()
    leakage_check = {
        "n_train": n_train, "n_val": n_val, "n_test": n_test,
        "fraud_rate_train": float(fraud_rate(train_ds)),
        "fraud_rate_val": float(fraud_rate(val_ds)),
        "fraud_rate_test": float(fraud_rate(test_ds)),
    }

    g = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = build_model(model_name, n_features=train_ds.n_features).to(device)

    # 클래스 불균형 보정: train의 (정상/사기) 비율로 pos_weight 계산 (모든 L/모델 공통 방식)
    n_pos = train_ds.y.sum()
    n_neg = len(train_ds.y) - n_pos
    pos_weight = torch.tensor(n_neg / max(n_pos, 1.0), dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    ckpt_path = f"{out_dir}/_ckpt_{model_name}_L{L}.pt"
    start_epoch = 1
    history = []
    best_val_prauc = -1
    best_state = None
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        history = ckpt["history"]
        best_val_prauc = ckpt["best_val_prauc"]
        best_state = ckpt["best_state"]
        print(f"[{model_name}/L={L}] resumed from epoch {ckpt['epoch']} checkpoint")

    t0 = time.time()
    for epoch in range(start_epoch, EPOCHS + 1):
        train_loss, _, _ = run_epoch(model, train_loader, optimizer, criterion, device, train=True)
        val_loss, val_probs, val_y = run_epoch(model, val_loader, optimizer, criterion, device, train=False)
        val_metrics = compute_metrics(val_y, val_probs)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **val_metrics})
        print(f"[{model_name}/L={L}] epoch {epoch}/{EPOCHS} "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_recall={val_metrics['recall']:.4f} val_pr_auc={val_metrics['pr_auc']:.4f}")
        if val_metrics["pr_auc"] > best_val_prauc:
            best_val_prauc = val_metrics["pr_auc"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        torch.save({
            "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "epoch": epoch, "history": history,
            "best_val_prauc": best_val_prauc, "best_state": best_state,
        }, ckpt_path)
    elapsed = time.time() - t0

    model.load_state_dict(best_state)
    test_loss, test_probs, test_y = run_epoch(model, test_loader, optimizer, criterion, device, train=False)
    test_metrics = compute_metrics(test_y, test_probs)
    print(f"[{model_name}/L={L}] TEST: {test_metrics}  (elapsed {elapsed:.1f}s)")

    result = {
        "model": model_name, "L": L,
        "hyperparams": {
            "seed": SEED, "batch_size": BATCH_SIZE, "epochs": EPOCHS, "lr": LR,
            "weight_decay": WEIGHT_DECAY, "pos_weight": float(pos_weight),
            "decision_threshold": DECISION_THRESHOLD,
        },
        "leakage_check": leakage_check,
        "history": history,
        "test_metrics": test_metrics,
        "elapsed_sec": elapsed,
    }
    out_path = f"{out_dir}/{model_name}_L{L}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"saved -> {out_path}")
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--model", choices=["tcn", "transformer"], required=True)
    args = ap.parse_args()
    main(args.L, args.model)
