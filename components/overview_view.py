from pathlib import Path

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parents[1]
EDA_DIR = BASE_DIR / "data" / "eda" / "korea_desc"


def show_overview():
    st.title("🏋️ Gym Churn Insight")
    st.markdown(
        "회원 이용·결제 행동을 바탕으로 이탈 위험을 파악하고, 한정된 캠페인 자원을 "
        "어디에 먼저 사용할지 결정하는 대시보드입니다."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("전체 회원", "1,000,000명")
    c2.metric("현재 이탈률", "38.57%")
    c3.metric("유지 회원", "614,298명")
    c4.metric("이탈 회원", "385,702명")

    st.info(
        "최종 후보는 **LightGBM · 상위 50% 피처**입니다. 기본 임계값 0.50에서 "
        "Precision은 약 60%, Recall은 약 38%이므로 캠페인 여력에 맞춘 임계값 검토가 필요합니다."
    )

    st.subheader("유지 회원과 이탈 회원의 행동 차이")
    behavior = pd.DataFrame(
        [
            ["월 방문 수", 10.42, 8.70, "낮은 방문 빈도는 이탈 위험 신호"],
            ["평균 운동 시간(분)", 73.32, 71.17, "이탈 회원의 운동 시간이 다소 짧음"],
            ["그룹 수업 참석", 6.04, 5.42, "참여가 낮을수록 이탈이 증가하는 경향"],
            ["PT 이용 횟수", 0.94, 0.60, "PT 참여가 낮을수록 이탈이 증가하는 경향"],
            ["기구 대기 시간(분)", 11.78, 12.58, "대기 시간이 길수록 이탈이 소폭 증가"],
            ["연체 횟수", 2.64, 3.57, "가장 뚜렷한 이탈 위험 신호"],
        ],
        columns=["지표", "유지 회원 평균", "이탈 회원 평균", "관찰 결과"],
    )
    st.dataframe(behavior, hide_index=True, width="stretch")

    tab1, tab2, tab3 = st.tabs(["이탈 분포", "변수 분포", "상관관계"])
    with tab1:
        _show_image("churn_class_distribution.png", "유지·이탈 클래스 분포")
    with tab2:
        _show_image("numeric_distribution.png", "핵심 수치형 변수 분포")
    with tab3:
        _show_image("correlation_heatmap_2.png", "전처리 후 변수 상관관계")

    left, right = st.columns(2)
    with left:
        st.subheader("핵심 위험 신호")
        st.markdown(
            """
            1. 결제 연체 횟수 증가
            2. 월 방문 및 운동 참여 감소
            3. PT·그룹 수업 참여 감소
            4. 기구 대기 시간 증가
            """
        )
    with right:
        st.subheader("해석 시 주의사항")
        st.markdown(
            """
            - 관찰된 차이는 인과관계를 의미하지 않습니다.
            - 현재 데이터는 합성 데이터일 가능성이 있어 실제 운영 데이터에서 재검증해야 합니다.
            - `Profile_Type_Drop-out`은 결과를 암시하므로 타깃 누수 가능성을 확인해야 합니다.
            """
        )


def _show_image(filename, caption):
    path = EDA_DIR / filename
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.warning(f"EDA 이미지가 없습니다: {filename}")
