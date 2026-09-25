# BankSim 사기 거래 탐지: 개별 거래 모델 vs 시퀀스 모델

BankSim 데이터로 **개별 거래만 보는 모델(MLP)**과 **과거 거래 이력을 함께 보는 시퀀스 모델(GRU, LSTM, TCN, Transformer)**의 사기 탐지 성능을 비교한 실험 코드와 결과입니다.

## 한눈에 보기

- **비교 모델:** MLP, GRU, LSTM, TCN, Transformer (5개)
- **시퀀스 길이 L:** 5, 10, 20
- **주 평가 지표:** PR-AUC (사기 거래 비율이 약 1.2%인 불균형 데이터이므로)
- **핵심 결과:** TCN이 모든 L에서 PR-AUC가 가장 높음 (최고 0.8707, L=5). GRU·LSTM·TCN은 MLP보다 PR-AUC가 약 0.03~0.05 높음. Transformer는 MLP와 비슷하거나 낮음.

## 최종 결과 (Test, threshold 0.5, seed 42)

| Model | L | Acc | Precision | Recall | F1 | ROC-AUC | PR-AUC | best epoch |
|---|---|---|---|---|---|---|---|---|
| MLP | 5 | 0.9729 | 0.2751 | 0.9260 | 0.4241 | 0.9889 | 0.8180 | 26/31 |
| GRU | 5 | 0.9657 | 0.2342 | 0.9625 | 0.3768 | 0.9947 | 0.8460 | 15/20 |
| LSTM | 5 | 0.9602 | 0.2039 | 0.9281 | 0.3343 | 0.9918 | 0.8203 | 19/24 |
| **TCN** | 5 | 0.9765 | 0.3076 | 0.9469 | 0.4644 | 0.9958 | **0.8707** | 29/34 |
| Transformer | 5 | 0.9766 | 0.3060 | 0.9219 | 0.4595 | 0.9938 | 0.8158 | 10/15 |
| MLP | 10 | 0.9729 | 0.2751 | 0.9260 | 0.4241 | 0.9889 | 0.8180 | 26/31 |
| GRU | 10 | 0.9635 | 0.2222 | 0.9542 | 0.3605 | 0.9937 | 0.8466 | 19/24 |
| LSTM | 10 | 0.9781 | 0.3231 | 0.9396 | 0.4808 | 0.9944 | 0.8476 | 13/18 |
| **TCN** | 10 | 0.9792 | 0.3340 | 0.9396 | 0.4928 | 0.9950 | **0.8538** | 13/18 |
| Transformer | 10 | 0.9778 | 0.3183 | 0.9250 | 0.4736 | 0.9943 | 0.8242 | 17/22 |
| MLP | 20 | 0.9729 | 0.2751 | 0.9260 | 0.4241 | 0.9889 | 0.8180 | 26/31 |
| GRU | 20 | 0.9786 | 0.3265 | 0.9281 | 0.4831 | 0.9940 | 0.8483 | 18/23 |
| LSTM | 20 | 0.9780 | 0.3199 | 0.9250 | 0.4754 | 0.9938 | 0.8481 | 19/24 |
| **TCN** | 20 | 0.9658 | 0.2321 | 0.9437 | 0.3726 | 0.9935 | **0.8508** | 15/20 |
| Transformer | 20 | 0.9749 | 0.2908 | 0.9250 | 0.4425 | 0.9931 | 0.8049 | 10/15 |

- MLP는 현재 거래(t)의 특징만 쓰므로 L과 관계없이 결과가 같습니다.
- Recall은 높고 Precision은 낮은데, 사기 거래 가중치(약 78배)와 임계값 0.5 때문입니다.
- seed 1개로 한 번 돌린 결과라, 차이가 0.01 이하인 비교(예: GRU와 LSTM, L=10·20)는 우연일 수 있습니다.
- 원본 수치: `hparam_results/final_summary.csv`

## 폴더 구조

```
.
├── hparam_search.py            # 5개 모델 하이퍼파라미터 탐색 + 최종 학습 (최종 결과 생성 코드)
├── mlp_rnn/
│   └── train_mlp_rnn.py        # MLP/GRU/LSTM 단독 학습 (고정 설정)
├── tcn_transformer/
│   ├── models.py               # TCN, Transformer 모델 정의
│   ├── dataset.py              # npz 데이터 로더
│   └── train.py                # TCN/Transformer 단독 학습 (최대 30 epochs + 조기 종료)
├── hparam_results/
│   ├── search_results.csv      # 탐색한 135개 조합의 검증 PR-AUC
│   ├── best_config.json        # 모델별 최적 하이퍼파라미터
│   ├── final_summary.csv       # 최종 테스트 결과 (위 표)
│   └── final_*.pt              # 최종 모델 가중치 15개
├── search.log                  # hparam_search.py 실행 로그
└── requirements_lab.txt        # 실험 환경 라이브러리 버전 (pip freeze)
```

**논문 결과는 `hparam_search.py`로 만든 `hparam_results/`를 기준으로 합니다.** `mlp_rnn/`와 `tcn_transformer/train.py`는 탐색 전에 고정 설정으로 돌려 본 코드입니다.

## 실험 설정

### 데이터

- BankSim 원본을 팀 저장소의 `preprocess_banksim.py`로 전처리한 `seq_L{5,10,20}/{train,val,test}.npz`를 사용합니다.
- 거래 발생 시점 기준으로 분할: Train 417,847건(사기 5,280건) / Val 87,702건 / Test 89,093건
- 입력: 길이 L의 거래 시퀀스 (특징 9개) + 패딩 마스크. 레이블은 마지막 시점 t 거래의 사기 여부입니다.
- 데이터 파일은 용량 때문에 저장소에 포함하지 않았습니다. `data/processed/banksim/` 아래에 두고 실행하세요.

### 하이퍼파라미터 탐색 (1단계)

L=10 데이터로 모델마다 아래 27개 조합을 학습하고, **검증 PR-AUC**가 가장 높은 조합을 골랐습니다. 탐색 단계는 최대 10 epochs, patience 3입니다.

| Hyperparameter | Search Range |
|---|---|
| Learning rate | {0.0005, 0.001, 0.005} |
| Hidden size (MLP 유닛 / GRU·LSTM 은닉 상태 / TCN 채널 / Transformer d_model) | {16, 32, 64} |
| Dropout | {0.1, 0.2, 0.3} |

선택된 값은 `hparam_results/best_config.json`에 있습니다.

### 최종 학습 (2단계)

고른 조합으로 L=5, 10, 20 각각 학습 후 테스트합니다.

| 항목 | 값 |
|---|---|
| Max epochs / Early stopping | 50 / patience 5 (검증 PR-AUC 기준) |
| Optimizer | Adam, weight decay 1e-5 |
| Batch size | 2,048 |
| Loss | Weighted BCE, pos_weight = N_neg / N_pos ≈ 78.14 |
| Decision threshold | 0.5 |
| Seed | 42 |
| 모델 선택 | 검증 PR-AUC가 가장 높은 에포크의 가중치 |

### 고정 구조

| Model | 구조 |
|---|---|
| MLP | 현재 거래 9개 특징 입력, 은닉층 2개 |
| GRU / LSTM | 2층, 마지막 시점 은닉 상태 → FC(32) → ReLU → Dropout → FC(1) |
| TCN | Causal dilated conv 블록 3개 (kernel 3, dilation 1·2·4), 잔차 연결 + BatchNorm |
| Transformer | 인코더 2층, head 4, FF = 2 × d_model, 사인·코사인 위치 인코딩, padding mask |

## 실행 방법

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch numpy scikit-learn

# 데이터를 data/processed/banksim/seq_L{5,10,20}/ 에 넣은 뒤
nohup python hparam_search.py > search.log 2>&1 &
tail -f search.log
```

- `python hparam_search.py --stage search`: 탐색만 실행
- `python hparam_search.py --stage final`: `best_config.json`으로 최종 학습만 실행
- 중간에 멈춰도 다시 실행하면 끝난 조합은 건너뜁니다.
- RTX 4090 기준으로 전체 약 2~3시간이 걸렸습니다.

## 실험 환경

| 항목 | 내용 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 (1장 사용) |
| NVIDIA Driver / CUDA | 580.173.02 / 13.0 |
| PyTorch | 2.14.0+cu130 |
| NumPy | 2.2.6 |
| scikit-learn | 1.7.2 |

전체 패키지 버전은 `requirements_lab.txt`를 참고하세요.

## 한계

- seed 1개(42)로 한 번 실행한 결과입니다.
- 탐색은 L=10에서만 했고, 같은 값을 L=5·20에도 썼습니다.
- 임계값 0.5 고정이라 Precision이 낮게 나옵니다.
- BankSim은 시뮬레이션 데이터라 실제 금융 데이터에서의 일반화는 추가 검증이 필요합니다.
