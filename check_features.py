"""npz의 feature 열 순서 확인용. 실행: python check_features.py"""
import numpy as np, os
p = os.path.join("data", "processed", "banksim", "seq_L10", "train.npz")
d = np.load(p, allow_pickle=True)
print("keys:", d.files)
for k in d.files:
    v = d[k]
    print(k, getattr(v, "shape", None), getattr(v, "dtype", None))
    if v.dtype.kind in "OU" or v.ndim <= 1 and v.size < 20:
        print("   ->", v)
X = d["X"]
print("\n열별 통계 (마지막 시점 t 기준) — 어느 열이 category(0~14 정수) / sin,cos(-1~1) / zscore 인지 추정")
for j in range(X.shape[2]):
    col = X[:, -1, j]
    print(f"  col {j}: min={col.min():8.3f} max={col.max():8.3f} mean={col.mean():8.3f} "
          f"n_unique={len(np.unique(col[:20000]))}")
