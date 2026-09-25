"""
TCN / Transformer 모델.
입력 X: (batch, L, 9), is_padding: (batch, L) [1=padding, 0=real]
출력: (batch,) sigmoid 이전 logit (BCEWithLogitsLoss와 함께 사용)
"""
import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# TCN (Temporal Convolutional Network)
#   - Causal + Dilated 1D Conv 블록을 쌓는다.
#   - padding 위치는 conv에 들어가기 전에 0으로 마스킹해서, 최소한 padding이
#     "진짜 신호"처럼 섞여 들어가는 것을 줄인다 (완전한 attention mask는 아니지만
#     합성곱 구조 특성상 이 정도 마스킹이 일반적인 처리 방식).
# ---------------------------------------------------------------------------
class CausalConv1dBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation  # causal padding: 왼쪽에만 패딩
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=0, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=0, dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.norm1 = nn.BatchNorm1d(out_ch)
        self.norm2 = nn.BatchNorm1d(out_ch)

    def _causal_pad(self, x):
        return nn.functional.pad(x, (self.pad, 0))  # (left, right) - 왼쪽만 패딩 = 미래 안 봄

    def forward(self, x):  # x: (batch, ch, L)
        residual = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.norm1(self.conv1(self._causal_pad(x))))
        out = self.dropout(out)
        out = self.relu(self.norm2(self.conv2(self._causal_pad(out))))
        out = self.dropout(out)
        return self.relu(out + residual)


class TCNClassifier(nn.Module):
    def __init__(self, n_features=9, channels=(32, 32, 32), kernel_size=3, dropout=0.1):
        super().__init__()
        layers = []
        in_ch = n_features
        for i, out_ch in enumerate(channels):
            dilation = 2 ** i  # 층이 깊어질수록 dilation을 늘려 더 먼 과거까지 봄
            layers.append(CausalConv1dBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(in_ch, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1)
        )

    def forward(self, x, is_padding):
        # x: (batch, L, 9) -> is_padding=1인 위치는 0으로 죽여서 conv에 진짜 신호처럼 안 들어가게 함
        mask = (1.0 - is_padding).unsqueeze(-1)  # (batch, L, 1) 1=real
        x = x * mask
        x = x.transpose(1, 2)          # (batch, 9, L) -- conv1d는 채널-우선
        out = self.tcn(x)              # (batch, ch, L)
        last = out[:, :, -1]           # 마지막 시점(t, 현재 거래) 벡터만 사용 -> 라벨이 t 기준이므로
        logit = self.head(last).squeeze(-1)
        return logit


# ---------------------------------------------------------------------------
# Transformer Encoder
#   - is_padding을 src_key_padding_mask로 그대로 넘겨 padding 위치를 attention에서 제외.
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):  # x: (batch, L, d_model)
        return x + self.pe[:, : x.size(1)]


class TransformerClassifier(nn.Module):
    def __init__(self, n_features=9, d_model=32, nhead=4, num_layers=2, dim_ff=64, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1)
        )

    def forward(self, x, is_padding):
        # x: (batch, L, 9), is_padding: (batch, L) 1=padding
        h = self.input_proj(x)
        h = self.pos_enc(h)
        key_padding_mask = is_padding.bool()  # True=마스킹(무시)할 위치, PyTorch 규약 그대로
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        last = h[:, -1, :]  # 마지막 시점(t) 벡터 -- 라벨이 t 기준이라 항상 실제 거래(패딩 아님)
        logit = self.head(last).squeeze(-1)
        return logit


# ---------------------------------------------------------------------------
# MLP baseline (참고용 -- 담당은 아니지만 본인 결과와 비교하려면 필요)
#   입력: 현재 시점(t) feature 9개만 사용 (과거 이력 없음)
# ---------------------------------------------------------------------------
class MLPBaseline(nn.Module):
    def __init__(self, n_features=9, hidden=32, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, is_padding=None):
        # x: (batch, 9) -- 시퀀스가 아니라 현재 시점 하나
        return self.net(x).squeeze(-1)
