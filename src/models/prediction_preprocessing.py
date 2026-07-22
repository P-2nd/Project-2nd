"""회원 이탈 예측 화면에서 사용하는 학습 전처리 재현 도구.

저장된 평가 모델은 ``preprocessing.py``가 만든 수치형 데이터로 학습됐다.
당시 scaler 객체는 파일로 저장되지 않았으므로, 이 모듈은 같은 원본 데이터와
같은 train/test 분할을 사용해 scaler를 다시 적합한다. 새 임의 규칙을 적용하지
않고, 회원 1명의 입력을 당시 모델 피처 순서로 변환하기 위한 코드다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler


RANDOM_STATE = 42
TEST_SIZE = 0.2

STANDARD_COLUMNS = [
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
ROBUST_COLUMNS = ["PT_Session_Count", "Late_Payment_Count"]

NUMERIC_INPUT_COLUMNS = [
    "Age",
    "Monthly_Visits",
    "Avg_Workout_Duration_Min",
    "Group_Class_Attendance",
    "PT_Session_Count",
    "Avg_Equipment_Wait_Time_Min",
    "Late_Payment_Count",
]


def fit_prediction_preprocessor(raw_data_path: str | Path) -> dict[str, Any]:
    """Fit the exact scaler setup used by the existing preprocessing script.

    The original script fits scalers after a stratified 80/20 split of the
    complete feature frame. Selecting just the numerical source columns here
    leaves row order and split indices unchanged, while avoiding an unnecessary
    one-million-row dummy-encoding operation on every Streamlit process.
    """

    raw_data_path = Path(raw_data_path)
    use_columns = ["Membership_Start_Date", "Churn", *NUMERIC_INPUT_COLUMNS]
    raw = pd.read_csv(raw_data_path, usecols=use_columns)

    start_dates = pd.to_datetime(raw["Membership_Start_Date"])
    reference_date = start_dates.max()

    numeric_features = raw[NUMERIC_INPUT_COLUMNS].copy()
    numeric_features["Start_Year"] = start_dates.dt.year
    numeric_features["Start_Month"] = start_dates.dt.month
    numeric_features["Start_Weekday"] = start_dates.dt.weekday
    numeric_features["Membership_Days"] = (reference_date - start_dates).dt.days

    train_features, _ = train_test_split(
        numeric_features,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=raw["Churn"],
    )

    standard_scaler = StandardScaler().fit(train_features[STANDARD_COLUMNS])
    robust_scaler = RobustScaler().fit(train_features[ROBUST_COLUMNS])

    campaign_columns = [
        "Monthly_Visits",
        "Group_Class_Attendance",
        "PT_Session_Count",
        "Avg_Equipment_Wait_Time_Min",
        "Late_Payment_Count",
    ]
    campaign_thresholds = raw[campaign_columns].quantile([0.25, 0.75])

    return {
        "reference_date": reference_date,
        "standard_scaler": standard_scaler,
        "robust_scaler": robust_scaler,
        "campaign_thresholds": campaign_thresholds.to_dict(),
    }


def transform_member_input(
    member: Mapping[str, Any],
    preprocessor: Mapping[str, Any],
    model_feature_names: list[str],
) -> pd.DataFrame:
    """Convert one form submission to the saved model's feature schema."""

    membership_start_date = pd.Timestamp(member["Membership_Start_Date"])
    reference_date = pd.Timestamp(preprocessor["reference_date"])

    row = {
        "Age": float(member["Age"]),
        "Monthly_Visits": float(member["Monthly_Visits"]),
        "Avg_Workout_Duration_Min": float(member["Avg_Workout_Duration_Min"]),
        "Group_Class_Attendance": float(member["Group_Class_Attendance"]),
        "PT_Session_Count": float(member["PT_Session_Count"]),
        "Avg_Equipment_Wait_Time_Min": float(member["Avg_Equipment_Wait_Time_Min"]),
        "Late_Payment_Count": float(member["Late_Payment_Count"]),
        "Start_Year": membership_start_date.year,
        "Start_Month": membership_start_date.month,
        "Start_Weekday": membership_start_date.weekday(),
        "Membership_Days": (reference_date - membership_start_date).days,
    }
    frame = pd.DataFrame([row])
    frame[STANDARD_COLUMNS] = preprocessor["standard_scaler"].transform(
        frame[STANDARD_COLUMNS]
    )
    frame[ROBUST_COLUMNS] = preprocessor["robust_scaler"].transform(
        frame[ROBUST_COLUMNS]
    )

    categorical_values = {
        "Cardio_Preference": member["Cardio_Preference"],
        "Supplement_Usage": member["Supplement_Usage"],
        "Profile_Type": member["Profile_Type"],
    }
    for feature_name in model_feature_names:
        if feature_name.startswith("Cardio_Preference_"):
            value = feature_name.removeprefix("Cardio_Preference_").replace("_", " ")
            frame[feature_name] = int(categorical_values["Cardio_Preference"] == value)
        elif feature_name.startswith("Supplement_Usage_"):
            value = feature_name.removeprefix("Supplement_Usage_").replace("_", " ")
            frame[feature_name] = int(categorical_values["Supplement_Usage"] == value)
        elif feature_name.startswith("Profile_Type_"):
            value = feature_name.removeprefix("Profile_Type_")
            frame[feature_name] = int(categorical_values["Profile_Type"] == value)

    missing_features = set(model_feature_names) - set(frame.columns)
    if missing_features:
        raise ValueError(f"지원하지 않는 모델 피처: {sorted(missing_features)}")

    return frame.loc[:, model_feature_names]


def campaign_thresholds(preprocessor: Mapping[str, Any]) -> dict[str, dict[float, float]]:
    """Return EDA distribution quartiles retained while fitting the preprocessor."""

    return preprocessor["campaign_thresholds"]
