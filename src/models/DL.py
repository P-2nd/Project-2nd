import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, recall_score, roc_auc_score, accuracy_score

import json
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, recall_score, roc_auc_score,
    accuracy_score, confusion_matrix,
)




data = pd.read_csv('../../data/processed/churn_preprocessed_full.csv')

df = pd.read_csv("../../data/processed/churn_preprocessed_full.csv")

X = df.drop(columns=["Churn"])   # 타겟을 제외한 모든 피처
y = df["Churn"]                  # 타겟 변수

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,        # 이탈률 38.57% 비율을 양쪽에 동일하게 유지
    random_state=42,
)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, recall_score, roc_auc_score, accuracy_score

"""
Gym Membership Churn - 딥러닝 베이직 모델 (MLP)
================================================
- 데이터  : 전처리 완료된 churn_preprocessed_full.csv
- 모델    : Linear + ReLU 은닉층 3개 (128 -> 64 -> 32), 출력층 Sigmoid
- 손실함수: BCELoss (모델 출력이 확률이므로)
- 평가    : ROC-AUC, F1, Recall, Accuracy + 추론 시간
- 저장    : JSON(요약) + Parquet(ROC 곡선용 원천 데이터)

실행 위치: src/models/DL.py 기준 (상대경로 ../../data)
"""

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ------------------------------------------------------------
# 0. 설정
# ------------------------------------------------------------
CSV_PATH   = "../../data/processed/churn_preprocessed_full.csv"
SAVE_PATH  = "../../data/evaluation/"      # 평가 결과 저장 폴더 (팀 규칙에 맞게 수정)
MODEL_NAME = "MLP"
PERCENT    = "100"             # pct50 데이터 사용 시 "50"으로 변경
EPOCHS     = 1
PRINT_EVERY = 10              # 10 에폭마다 loss 출력
BATCH_SIZE = 16384
LR         = 0.01

# ------------------------------------------------------------
# 1. 데이터 로드 및 X, y 분리
# ------------------------------------------------------------
df = pd.read_csv(CSV_PATH)

X = df.drop(columns=["Churn", "PT_Session_Count"])   # 타겟,PT_Session_Count을 제외한 모든 피처
y = df["Churn"]                  # 타겟 변수

print(f"[Data] X={X.shape}, y={y.shape}")

# ------------------------------------------------------------
# 2. Train / Test 분할 (클래스 비율 유지)
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,        # 이탈률 38.57% 비율을 양쪽에 동일하게 유지
    random_state=SEED,
)
print(f"[Split] train={len(y_train):,}, test={len(y_test):,}")

# ------------------------------------------------------------
# 3. 텐서 변환 및 DataLoader
# ------------------------------------------------------------
X_train_t = torch.tensor(X_train.values.astype(np.float32))
X_test_t  = torch.tensor(X_test.values.astype(np.float32))
y_train_t = torch.tensor(y_train.values.astype(np.float32))
y_test_t  = torch.tensor(y_test.values.astype(np.float32))

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                          batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(TensorDataset(X_test_t, y_test_t),
                          batch_size=BATCH_SIZE, shuffle=False)

# ------------------------------------------------------------
# 4. 베이직 MLP: 은닉층 3개 + 출력층 Sigmoid
# ------------------------------------------------------------
class ChurnMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),  # 은닉층 1
            nn.ReLU(),
            nn.Linear(128, 32),      # 은닉층 2
            nn.ReLU(),
            nn.Linear(32, 8),       # 은닉층 3
            nn.ReLU(),
            nn.Linear(8, 1),        # 출력층
            nn.Sigmoid(),            # 0~1 확률로 변환
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


device = torch.device("cpu")
model = ChurnMLP(in_dim=X_train.shape[1]).to(device)
print(f"[Device] {device}")
print(model)

# ------------------------------------------------------------
# 5. 손실함수 / 옵티마이저
#    모델 안에 Sigmoid가 있으므로 BCELoss 사용
# ------------------------------------------------------------
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ------------------------------------------------------------
# 6. 학습 루프
# ------------------------------------------------------------
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(yb)


    print(f"Epoch {epoch:05d} | loss = {total_loss / len(y_train):.4f}")

# ------------------------------------------------------------
# 7. 평가 (+ 추론 시간 측정)
# ------------------------------------------------------------
model.eval()
probs = []

start_time = time.time()
with torch.no_grad():
    for xb, _ in test_loader:
        probs.append(model(xb.to(device)).cpu().numpy())
total_time = time.time() - start_time          # 전체 추론 시간(초)

probs = np.concatenate(probs)
preds = (probs >= 0.5).astype(int)

y_true   = y_test.values.astype(int)           # 실제 라벨
y_scores = probs                               # 예측 확률 (ROC 곡선용)

avg_latency_ms = total_time / len(y_true) * 1000   # 샘플당 평균 지연(ms)
auc_value = roc_auc_score(y_true, y_scores)
cm = confusion_matrix(y_true, preds)

print("\n===== Test 성능 =====")
print(f"ROC-AUC  : {auc_value:.4f}")
print(f"F1-score : {f1_score(y_true, preds):.4f}")
print(f"Recall   : {recall_score(y_true, preds):.4f}")
print(f"Accuracy : {accuracy_score(y_true, preds):.4f} (참고용)")
print(f"추론 시간 : {total_time:.2f}초 (샘플당 {avg_latency_ms:.4f}ms)")
print("Confusion Matrix:\n", cm)

# ------------------------------------------------------------
# 8. 결과 저장 (JSON + Parquet, 팀 공통 포맷)
# ------------------------------------------------------------
# 파케이는 AUC 그래프 그리기용으로 원천 데이터만 저장
# df_roc = pd.DataFrame({"y_true": y_true, "y_score": y_scores})
#
# summary_data = {
#     "performance": {
#         "total_time_sec": total_time,
#         "avg_latency_ms": avg_latency_ms,
#         "total_samples": len(y_true)
#     },
#     "confusion_matrix": cm.tolist(),
#     "auc_score": auc_value
# }
#
# with open(f"{SAVE_PATH}/{MODEL_NAME}_{PERCENT}_eval.json", "w", encoding="utf-8") as f:
#     json.dump(summary_data, f, indent=4, ensure_ascii=False)
#
# df_roc.to_parquet(
#     f"{SAVE_PATH}/{MODEL_NAME}_auc_roc.parquet", index=False)
#
# print(f"\n[저장 완료] {SAVE_PATH}/{MODEL_NAME}_{PERCENT}_eval.json")
# print(f"[저장 완료] {SAVE_PATH}/{MODEL_NAME}_auc_roc.parquet")