"""회원 입력을 기존 LightGBM 평가 모델로 점수화하는 Streamlit 화면."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import joblib
import streamlit as st

from src.models.prediction_preprocessing import (
    campaign_thresholds,
    fit_prediction_preprocessor,
    transform_member_input,
)


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "gym_churn_1M_dataset.csv"
MODEL_PATH = BASE_DIR / "data" / "evaluation" / "saved_models" / "lightgbm_full_eval.joblib"
MODEL_LABEL = "LightGBM Full Evaluation (lightgbm_full_eval.joblib)"


@st.cache_resource(show_spinner=False)
def load_prediction_model(model_path: str):
    """Load the shared model once per Streamlit process."""

    return joblib.load(model_path)


@st.cache_resource(show_spinner="기존 전처리 기준을 불러오는 중입니다...")
def load_prediction_preprocessor(raw_data_path: str):
    """Recreate and retain the existing preprocessing scalers once."""

    return fit_prediction_preprocessor(raw_data_path)


def _risk_level(probability: float) -> str:
    if probability < 0.30:
        return "저위험"
    if probability < 0.60:
        return "중위험"
    return "고위험"


def _campaign_suggestions(member: dict, preprocessor: dict) -> list[str]:
    thresholds = campaign_thresholds(preprocessor)
    suggestions = []

    if member["Monthly_Visits"] <= thresholds["Monthly_Visits"][0.25]:
        suggestions.append("방문 리마인드와 개인 운동 루틴을 제안하세요.")
    if (
        member["Group_Class_Attendance"] <= thresholds["Group_Class_Attendance"][0.25]
        or member["PT_Session_Count"] <= thresholds["PT_Session_Count"][0.25]
    ):
        suggestions.append("그룹 수업 체험 또는 PT 상담을 제안하세요.")
    if member["Avg_Equipment_Wait_Time_Min"] >= thresholds["Avg_Equipment_Wait_Time_Min"][0.75]:
        suggestions.append("혼잡 시간대 회피와 기구 예약 방법을 안내하세요.")
    if member["Late_Payment_Count"] >= thresholds["Late_Payment_Count"][0.75]:
        suggestions.append("결제 상담 또는 분할 결제 옵션을 안내하세요.")

    if not suggestions:
        suggestions.append("현재 활동 패턴을 유지할 수 있도록 정기적인 프로그램 안내를 제공하세요.")
    return suggestions


def show_prediction() -> None:
    """Render the member churn prediction page."""

    st.title("Churn prediction (회원 이탈 예측)")
    st.caption("회원 정보를 입력하고 제출하면 저장된 평가 모델로 이탈 확률을 계산합니다.")

    with st.form("churn_prediction_form"):
        left, right = st.columns(2)
        with left:
            age = st.number_input("나이", min_value=14, max_value=100, value=35, step=1)
            membership_start_date = st.date_input(
                "가입일",
                value=date(2025, 1, 1),
                min_value=date(2023, 5, 21),
                max_value=date(2026, 2, 18),
                help="학습 데이터의 가입일 범위 안에서 입력하세요.",
            )
            monthly_visits = st.number_input("월 방문 횟수", min_value=0, max_value=31, value=8, step=1)
            workout_duration = st.number_input(
                "평균 운동 시간 (분)", min_value=0, max_value=300, value=60, step=1
            )
            group_class_attendance = st.number_input(
                "그룹 수업 참석 횟수", min_value=0, max_value=31, value=4, step=1
            )
            pt_session_count = st.number_input("PT 수업 횟수", min_value=0, max_value=31, value=0, step=1)
        with right:
            equipment_wait_time = st.number_input(
                "평균 기구 대기 시간 (분)", min_value=0.0, max_value=120.0, value=12.0, step=0.5
            )
            late_payment_count = st.number_input("연체 횟수", min_value=0, max_value=100, value=0, step=1)
            cardio_preference = st.selectbox(
                "유산소 운동 선호", ["Cycling", "Elliptical", "No Preference", "Rowing", "Treadmill"]
            )
            supplement_usage = st.selectbox(
                "보충제 사용 여부",
                ["Creatine", "Full Stack", "No Protein Supplements", "Pre-Workout", "Whey Protein"],
            )
            profile_type = st.selectbox(
                "회원 프로필 유형", ["Casual", "Drop-out", "Hardcore", "Regular"]
            )

        submitted = st.form_submit_button("이탈 확률 예측", type="primary", icon=":material/analytics:")

    if not submitted:
        return

    member = {
        "Age": age,
        "Membership_Start_Date": membership_start_date,
        "Monthly_Visits": monthly_visits,
        "Avg_Workout_Duration_Min": workout_duration,
        "Group_Class_Attendance": group_class_attendance,
        "PT_Session_Count": pt_session_count,
        "Avg_Equipment_Wait_Time_Min": equipment_wait_time,
        "Late_Payment_Count": late_payment_count,
        "Cardio_Preference": cardio_preference,
        "Supplement_Usage": supplement_usage,
        "Profile_Type": profile_type,
    }

    try:
        model = load_prediction_model(str(MODEL_PATH))
        preprocessor = load_prediction_preprocessor(str(RAW_DATA_PATH))
        feature_names = list(model.feature_name_)
        model_input = transform_member_input(member, preprocessor, feature_names)
        churn_class_index = list(model.classes_).index(1)
        churn_probability = float(model.predict_proba(model_input)[0, churn_class_index])
    except (FileNotFoundError, ValueError, AttributeError) as error:
        st.error(f"예측을 준비하지 못했습니다: {error}")
        return

    risk_level = _risk_level(churn_probability)
    probability_column, risk_column, model_column = st.columns(3)
    probability_column.metric("이탈 확률", f"{churn_probability:.1%}")
    risk_column.metric("위험 등급", risk_level)
    model_column.metric("사용 모델", "LightGBM")
    st.caption(MODEL_LABEL)
    st.info("이는 모델의 예측 결과이며 인과관계를 의미하지 않습니다.")

    st.subheader("운영 캠페인 제안")
    for suggestion in _campaign_suggestions(member, preprocessor):
        st.write(f"- {suggestion}")
    st.caption("위 제안은 개인별 인과 해석이 아니라 데이터 분포를 참고한 EDA 기반 운영 제안입니다.")
