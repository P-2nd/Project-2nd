"""LightGBM 배포 Pipeline에서 사용하는 원본 입력 변환기."""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


RAW_INPUT_FEATURES = [
    "Age",
    "Membership_Start_Date",
    "Monthly_Visits",
    "Avg_Workout_Duration_Min",
    "Group_Class_Attendance",
    "PT_Session_Count",
    "Avg_Equipment_Wait_Time_Min",
    "Late_Payment_Count",
    "Cardio_Preference",
    "Supplement_Usage",
    "Profile_Type",
]

DEPLOYMENT_FEATURES = [
    "Late_Payment_Count",
    "Monthly_Visits",
    "PT_Session_Count",
    "Start_Year",
    "Membership_Days",
    "Group_Class_Attendance",
    "Avg_Equipment_Wait_Time_Min",
    "Age",
    "Supplement_Usage_No Protein Supplements",
    "Cardio_Preference_Rowing",
    "Avg_Workout_Duration_Min",
]


class RawPct50FeatureTransformer(BaseEstimator, TransformerMixin):
    """원본 폼 입력을 LightGBM 50 모델용 11개 수치 피처로 변환한다."""

    def __init__(self, reference_date: str):
        self.reference_date = reference_date

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = pd.DataFrame(X).copy()
        missing = [column for column in RAW_INPUT_FEATURES if column not in frame.columns]
        if missing:
            raise ValueError(f"배포 입력에 필요한 컬럼이 없습니다: {', '.join(missing)}")

        dates = pd.to_datetime(frame["Membership_Start_Date"], errors="coerce")
        reference_date = pd.Timestamp(self.reference_date)
        result = pd.DataFrame(index=frame.index)
        for column in (
            "Late_Payment_Count",
            "Monthly_Visits",
            "PT_Session_Count",
            "Group_Class_Attendance",
            "Avg_Equipment_Wait_Time_Min",
            "Age",
            "Avg_Workout_Duration_Min",
        ):
            result[column] = pd.to_numeric(frame[column], errors="coerce")

        result["Start_Year"] = dates.dt.year
        result["Membership_Days"] = (reference_date - dates).dt.days.clip(lower=0)
        result["Supplement_Usage_No Protein Supplements"] = (
            frame["Supplement_Usage"].fillna("No Protein Supplements")
            .eq("No Protein Supplements")
            .astype(int)
        )
        result["Cardio_Preference_Rowing"] = (
            frame["Cardio_Preference"].fillna("No Preference").eq("Rowing").astype(int)
        )
        return result[DEPLOYMENT_FEATURES]
