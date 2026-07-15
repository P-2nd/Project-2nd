import pandas as pd
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler

churn = pd.read_csv('../../data/raw/gym_churn_1M_dataset.csv', sep=',', na_values=[''], quotechar='"')

# 결측치 처리
churn["Cardio_Preference"] = churn["Cardio_Preference"].fillna("No Preference")
churn["Supplement_Usage"] = churn["Supplement_Usage"].fillna("No Protein Supplements")

# 컬럼 삭제
churn = churn.drop('Member_ID', axis=1)
churn = churn.drop('Treadmill_Avg_Speed_Kmh', axis=1)
churn = churn.drop('Treadmill_Avg_Incline_Pct', axis=1)
churn = churn.drop('Gender', axis=1)
churn = churn.drop('Membership_Type', axis=1)
churn = churn.drop('Peak_Hour_Preference', axis=1)
churn = churn.drop('Monthly_Fee', axis=1)

# 날짜 파생변수 생성
churn["Membership_Start_Date"] = pd.to_datetime(churn["Membership_Start_Date"])
churn["Start_Year"] = churn["Membership_Start_Date"].dt.year
churn["Start_Month"] = churn["Membership_Start_Date"].dt.month
churn["Start_Weekday"] = churn["Membership_Start_Date"].dt.weekday
reference_date = churn["Membership_Start_Date"].max()
churn["Membership_Days"] = (reference_date - churn["Membership_Start_Date"]).dt.days
churn = churn.drop(columns=["Membership_Start_Date"])

# 인코딩
churn = pd.get_dummies(
    churn,
    columns=["Cardio_Preference", "Supplement_Usage", "Profile_Type"],
    drop_first=True
)

# Feature / Target 분리
X = churn.drop(columns=["Churn"])
y = churn["Churn"]

# Train/Test 분리 (스케일링 전에 먼저 분리 → 데이터 누수 방지)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 스케일링 대상 컬럼
standard_cols = [
    "Age",
    "Monthly_Visits",
    "Avg_Workout_Duration_Min",
    "Group_Class_Attendance",
    "Avg_Equipment_Wait_Time_Min",
    "Start_Year",
    "Start_Month",
    "Start_Weekday",
    "Membership_Days",
]

robust_cols = [
    "PT_Session_Count",
    "Late_Payment_Count",
]

standard_scaler = StandardScaler()
robust_scaler = RobustScaler()

# Train 기준으로 fit, Train/Test 각각 transform (누수 방지)
X_train[standard_cols] = standard_scaler.fit_transform(X_train[standard_cols])
X_test[standard_cols] = standard_scaler.transform(X_test[standard_cols])

X_train[robust_cols] = robust_scaler.fit_transform(X_train[robust_cols])
X_test[robust_cols] = robust_scaler.transform(X_test[robust_cols])

# XGBoost 모델 학습 (Feature Importance 확인용)
xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1
)

xgb_model.fit(X_train, y_train)

# Feature Importance 추출
importance_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": xgb_model.feature_importances_
}).sort_values("Importance", ascending=False).reset_index(drop=True)

# 중요도 기준 상위 50% 컬럼 선택
n_features = len(importance_df)
top_n = n_features // 2
top_features = importance_df["Feature"][:top_n].tolist()

# ------------------------------------------------------------
# 저장 1: 전처리 완료 데이터 (전체 Feature, Train/Test 재결합)
# ------------------------------------------------------------
save_dir = "../../data/processed"
os.makedirs(save_dir, exist_ok=True)

full_train = X_train.copy()
full_train["Churn"] = y_train
full_test = X_test.copy()
full_test["Churn"] = y_test

churn_preprocessed = pd.concat([full_train, full_test]).sort_index()
churn_preprocessed.to_csv(os.path.join(save_dir, "churn_preprocessed_full.csv"), index=False)

# ------------------------------------------------------------
# 저장 2: 상위 50%(11개) Feature만 선택한 데이터
# ------------------------------------------------------------
# 스케일링 반영된 churn_preprocessed에서 top_features만 선택 (일관성 유지)
churn_top50 = churn_preprocessed[top_features + ["Churn"]]

churn_top50.to_csv(os.path.join(save_dir, "churn_preprocessed_pct50.csv"), index=False)