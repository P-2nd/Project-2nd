from datetime import date

import streamlit as st

from utils import (
    find_deployment_artifacts,
    load_deployment_bundle,
    predict_customer,
    retention_recommendations,
    risk_level,
)


def show_prediction():
    st.title("👤 개별 고객 이탈 예측")
    st.caption("고객 행동 정보를 입력하고 이탈 확률, 위험 등급, 권장 유지 활동을 확인합니다.")

    model_path, metadata_path = find_deployment_artifacts()
    pipeline_ready = model_path is not None and metadata_path is not None
    if pipeline_ready:
        st.success(f"배포 Pipeline 연결됨: {model_path.name}")
    else:
        st.warning(
            "LightGBM 배포 Pipeline 또는 메타데이터가 아직 없습니다. 입력 폼과 유지 활동 추천은 "
            "사용할 수 있지만 실제 이탈 확률은 Pipeline 전달 후 활성화됩니다."
        )

    customer = _customer_form()

    st.subheader("권장 유지 활동")
    for title, description in retention_recommendations(customer):
        st.markdown(f"**{title}**  \n{description}")

    predict_clicked = st.button(
        "이탈 위험 예측",
        type="primary",
        disabled=not pipeline_ready,
        width="stretch",
    )

    if predict_clicked:
        try:
            model, metadata = load_deployment_bundle(str(model_path), str(metadata_path))
            probability, threshold = predict_customer(customer, model, metadata)
        except Exception as exc:
            st.error(f"예측을 수행하지 못했습니다: {exc}")
            return

        level, color = risk_level(probability)
        c1, c2, c3 = st.columns(3)
        c1.metric("이탈 확률", f"{probability:.1%}")
        c2.metric("운영 임계값", f"{threshold:.2f}")
        c3.metric("위험 등급", level)
        st.progress(probability)
        if probability >= threshold:
            st.error(f"이 고객은 이탈 관리 대상입니다. 위험 등급: :{color}[{level}]")
        else:
            st.success(f"현재 운영 임계값에서는 유지 대상으로 분류됩니다. 위험 등급: :{color}[{level}]")

    with st.expander("배포 Pipeline 연동 규격"):
        st.code(
            "models/lightgbm_pct50_pipeline.joblib\n"
            "models/lightgbm_pct50_metadata.json\n\n"
            "metadata 필수: input_mode='raw', input_features=[...], "
            "threshold_from_validation=0.xx"
        )


def _customer_form():
    with st.form("customer_input"):
        st.subheader("고객 정보 입력")
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("나이", min_value=16, max_value=100, value=30)
        start_date = c2.date_input("가입일", value=date(2024, 1, 1), max_value=date.today())
        cardio = c3.selectbox(
            "선호 유산소 운동",
            ["No Preference", "Cycling", "Elliptical", "Rowing", "Treadmill"],
        )

        c4, c5, c6 = st.columns(3)
        visits = c4.number_input("월 방문 횟수", min_value=0, max_value=31, value=9)
        duration = c5.number_input("평균 운동 시간(분)", min_value=0, max_value=240, value=72)
        wait_time = c6.number_input("평균 기구 대기 시간(분)", min_value=0.0, max_value=60.0, value=12.0)

        c7, c8, c9 = st.columns(3)
        group = c7.number_input("그룹 수업 참석 횟수", min_value=0, max_value=31, value=5)
        pt = c8.number_input("PT 이용 횟수", min_value=0, max_value=31, value=0)
        late = c9.number_input("연체 횟수", min_value=0, max_value=20, value=3)

        c10, c11 = st.columns(2)
        supplement = c10.selectbox(
            "보충제 이용",
            ["No Protein Supplements", "Whey Protein", "Pre-Workout", "Full Stack", "Other"],
        )
        profile = c11.selectbox("회원 프로필", ["Casual", "Regular", "Hardcore", "Drop-out"])
        st.form_submit_button("입력 내용 적용", width="stretch")

    membership_days = max((date.today() - start_date).days, 0)
    return {
        "Age": int(age),
        "Membership_Start_Date": start_date.isoformat(),
        "Monthly_Visits": int(visits),
        "Avg_Workout_Duration_Min": int(duration),
        "Group_Class_Attendance": int(group),
        "PT_Session_Count": int(pt),
        "Avg_Equipment_Wait_Time_Min": float(wait_time),
        "Late_Payment_Count": int(late),
        "Cardio_Preference": cardio,
        "Supplement_Usage": supplement,
        "Profile_Type": profile,
        "Membership_Days": membership_days,
        "Start_Year": start_date.year,
        "Start_Month": start_date.month,
        "Start_Weekday": start_date.weekday(),
    }
