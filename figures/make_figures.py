"""그림 3·4·5 생성 (MLP-tx 포함). 실행: python make_figures.py [final_summary.csv]"""
import sys, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

# ---- 한글 폰트 ----
for cand in ["Noto Sans CJK KR", "Noto Sans CJK JP", "NanumGothic", "Malgun Gothic", "AppleGothic"]:
    if any(f.name == cand for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = cand; break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 11

csv = sys.argv[1] if len(sys.argv) > 1 else "final_summary.csv"
df = pd.read_csv(csv)
OUT = os.path.dirname(os.path.abspath(__file__))

NAME = {"mlp_tx": "MLP-tx", "mlp": "MLP-agg", "gru": "GRU", "lstm": "LSTM", "transformer": "Transformer", "tcn": "TCN"}
ORDER = ["mlp_tx", "mlp", "gru", "lstm", "transformer", "tcn"]
SEQ = ["gru", "lstm", "transformer", "tcn"]
LS = [5, 10, 20]
pr = {(r.model, int(r.L)): r.test_pr_auc for r in df.itertuples()}
f1 = {(r.model, int(r.L)): r.test_f1 for r in df.itertuples()}
base = pr[("mlp", 10)]
tx = pr[("mlp_tx", 10)]

# ================= 그림 3: MLP-agg 대비 PR-AUC 차이 =================
fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
rows = ["tcn", "gru", "lstm", "transformer", "mlp_tx"]
for ax, L in zip(axes, LS):
    vals = [pr[(m, L)] - base for m in rows]
    colors = ["#3b6fb6" if v > 0.01 else ("#d97b2b" if v < -0.01 else "#9aa3ad") for v in vals]
    y = np.arange(len(rows))[::-1]
    ax.axvspan(-0.01, 0.01, color="#e6e6e6", zorder=0)
    ax.axvline(0, color="black", lw=1)
    ax.barh(y, vals, color=colors, height=0.55)
    for yi, v, m in zip(y, vals, rows):
        txt = f"{v:+.3f} ({pr[(m, L)]:.3f})"
        if v >= 0:
            ax.text(v + 0.003, yi, txt, va="center", ha="left", fontsize=9.5)
        elif v > -0.05:
            ax.text(v - 0.003, yi, txt, va="center", ha="right", fontsize=9.5)
        else:  # 큰 음수 막대: 막대 안쪽에 흰 글씨
            ax.text(v + 0.004, yi, txt, va="center", ha="left", fontsize=9.5, color="white", fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels([NAME[m] for m in rows])
    ax.set_xlim(-0.12, 0.08); ax.set_title(f"L = {L}", fontweight="bold")
    ax.set_xlabel("PR-AUC 차이 (모델 − MLP-agg)")
    ax.grid(axis="x", lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.suptitle("TCN·GRU·LSTM은 MLP-agg보다 PR-AUC가 높았고, Transformer는 차이가 작거나 낮았으며, 이력 feature를 뺀 MLP-tx는 크게 낮았다",
             fontsize=12.5, fontweight="bold")
fig.text(0.5, 0.005, f"기준선(0) = MLP-agg (PR-AUC {base:.3f}, L과 무관) · 괄호 안은 실제 PR-AUC · 회색 띠 = ±0.01 · MLP-tx는 L과 무관하게 동일 · 세 패널 x축 범위 동일",
         ha="center", fontsize=9.5, color="#555")
fig.tight_layout(rect=(0, 0.04, 1, 0.94))
fig.savefig(os.path.join(OUT, "fig3_prauc_diff.png"), dpi=300); plt.close(fig)

# ================= 그림 4: 히트맵 =================
hm_rows = ["mlp_tx", "mlp", "gru", "lstm", "tcn", "transformer"]
def heat(ax, data, cmap, title, fmt="{:.3f}"):
    M = np.array([[data[(m, L)] for L in LS] for m in hm_rows])
    im = ax.imshow(M, cmap=cmap, aspect="auto")
    vmin, vmax = M.min(), M.max()
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, fmt.format(v), ha="center", va="center",
                    color="white" if (v - vmin) / (vmax - vmin) > 0.55 else "black", fontsize=10.5)
    for j in range(M.shape[1]):
        i = int(np.argmax(M[:, j]))
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="red", lw=2.2))
    ax.set_xticks(range(3)); ax.set_xticklabels([f"L={L}" for L in LS])
    ax.set_yticks(range(len(hm_rows))); ax.set_yticklabels([NAME[m] for m in hm_rows])
    ax.set_title(title); ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    return im
fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
heat(axes[0], pr, "Blues", "PR-AUC 히트맵 (빨간 테두리 = 해당 L 1위)")
heat(axes[1], f1, "Oranges", "F1 히트맵 (빨간 테두리 = 해당 L 1위)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig4_heatmap.png"), dpi=300); plt.close(fig)

# ================= 그림 5: L별 모델 비교 =================
bar_models = ["mlp_tx", "mlp", "gru", "lstm", "tcn", "transformer"]
bar_colors = {"mlp_tx": "#bfbfbf", "mlp": "#808080", "gru": "#4daf4a", "lstm": "#8e6bbf",
              "tcn": "#3b6fb6", "transformer": "#e07b39"}
fig, ax = plt.subplots(figsize=(13, 6))
w = 0.13; x = np.arange(len(LS))
for k, m in enumerate(bar_models):
    vals = [pr[(m, L)] for L in LS]
    xs = x + (k - (len(bar_models) - 1) / 2) * w
    ax.bar(xs, vals, w, label=NAME[m], color=bar_colors[m])
    for xi, v in zip(xs, vals):
        ax.text(xi, v + 0.002, f"{v:.3f}", ha="center", va="bottom", rotation=90, fontsize=9.5)
ax.set_ylim(0.70, 0.90); ax.set_xticks(x); ax.set_xticklabels([f"L={L}" for L in LS])
ax.set_xlabel("시퀀스 길이 L"); ax.set_ylabel("Test PR-AUC")
ax.grid(axis="y", lw=0.4, alpha=0.5); ax.set_axisbelow(True)
ax.legend(ncol=6, loc="lower center", frameon=False, bbox_to_anchor=(0.5, 1.0))
ax.text(0.01, 0.985, "※ y축은 0.70부터 시작(차이를 보려고 절단) · MLP-tx, MLP-agg는 L과 무관", transform=ax.transAxes,
        fontsize=9, color="#555", va="top")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig5_bar_by_L.png"), dpi=300); plt.close(fig)
print("saved fig3/fig4/fig5")
