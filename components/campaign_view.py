import pandas as pd
import streamlit as st

from utils import calculate_threshold_metrics, load_parquet


FINAL_MODEL_KEY = "lightgbm_50"


def show_campaign():
    st.title("🎯 임계값·캠페인 시뮬레이터")
    st.caption(
        "LightGBM 상위 50% 피처 모델의 테스트 예측으로 임계값에 따른 대상 규모와 오탐·미탐을 비교합니다."
    )

    roc_df = load_parquet(FINAL_MODEL_KEY)
    if roc_df is None or roc_df.empty:
        st.error("LightGBM 50 ROC 원천 데이터가 없어 시뮬레이션을 실행할 수 없습니다.")
        return

    left, right = st.columns([2, 1])
    threshold = left.slider(
        "이탈 분류 임계값",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.01,
        help="낮추면 더 많은 이탈 고객을 찾지만 캠페인 대상과 오탐이 증가합니다.",
    )
    capacity = right.number_input(
        "최대 캠페인 가능 인원",
        min_value=100,
        max_value=int(len(roc_df)),
        value=min(50000, len(roc_df)),
        step=100,
    )
    metrics = calculate_threshold_metrics(roc_df, threshold)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("캠페인 후보", f"{metrics['selected']:,}명", f"전체의 {metrics['selected'] / metrics['total']:.1%}")
    c2.metric("Precision", f"{metrics['precision']:.1%}")
    c3.metric("Recall", f"{metrics['recall']:.1%}")
    c4.metric("F1 Score", f"{metrics['f1']:.3f}")

    st.subheader("운영 영향")
    impact = pd.DataFrame(
        [
            ["실제 이탈을 찾아낸 고객", metrics["tp"], "TP · 캠페인 우선 대상"],
            ["유지 고객을 잘못 선정", metrics["fp"], "FP · 불필요한 캠페인 비용"],
            ["놓치는 실제 이탈 고객", metrics["fn"], "FN · 이탈 방지 기회 손실"],
            ["정확히 제외한 유지 고객", metrics["tn"], "TN · 캠페인 제외"],
        ],
        columns=["구분", "인원", "의미"],
    )
    st.dataframe(
        impact,
        hide_index=True,
        width="stretch",
        column_config={"인원": st.column_config.NumberColumn(format="localized")},
    )

    selected = _campaign_candidates(roc_df, threshold, int(capacity))
    threshold_count = metrics["selected"]
    if threshold_count > capacity:
        st.warning(
            f"임계값 기준 후보 {threshold_count:,}명 중 위험도가 높은 {capacity:,}명만 선정합니다. "
            "나머지는 캠페인 대기군으로 관리하세요."
        )
    else:
        st.success(f"임계값 기준 후보 {threshold_count:,}명이 캠페인 가능 인원 안에 들어옵니다.")

    st.subheader("위험도 상위 캠페인 대상")
    st.dataframe(selected.head(100), hide_index=True, width="stretch")
    export_candidates = selected.drop(
        columns=["actual_churn_for_evaluation"],
        errors="ignore",
    )
    st.download_button(
        "캠페인 대상 CSV 다운로드",
        data=export_candidates.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"lightgbm_50_campaign_threshold_{threshold:.2f}.csv",
        mime="text/csv",
        width="stretch",
    )
    st.caption(
        "`evaluation_row_id`는 실제 회원 ID가 아니라 테스트 데이터 행 식별자입니다. "
        "운영 시 CRM의 `Member_ID`로 교체해야 합니다. 실제 이탈 정답은 다운로드 파일에서 제외됩니다."
    )

    with st.expander("실제 발송 전 운영 제외 기준", expanded=True):
        st.checkbox("이미 탈퇴 처리된 회원 제외", value=True, disabled=True)
        st.checkbox("최근 동일 캠페인을 받은 회원 제외", value=True, disabled=True)
        st.checkbox("마케팅 수신 미동의 회원 제외", value=True, disabled=True)
        st.info("현재 평가 데이터에는 위 CRM 필드가 없어 실제 제외는 CRM 연동 후 적용됩니다.")

    with st.expander("캠페인 효과 검증 방법"):
        st.markdown(
            """
            - 선정 대상 일부를 무작위 대조군으로 유지합니다.
            - 캠페인 수신군과 대조군의 이탈률·재방문율·유지율을 비교합니다.
            - 관찰 기간 종료 후 실제 이탈 결과로 Precision·Recall과 비용 대비 효과를 다시 계산합니다.
            """
        )


def _campaign_candidates(roc_df, threshold, capacity):
    candidates = roc_df.reset_index(names="evaluation_row_id").copy()
    candidates = candidates[candidates["y_score"] >= threshold]
    candidates = candidates.sort_values("y_score", ascending=False).head(capacity)
    candidates["risk_grade"] = pd.cut(
        candidates["y_score"],
        bins=[-float("inf"), 0.4, 0.7, float("inf")],
        labels=["유지", "주의", "고위험"],
        right=False,
    )
    return candidates.rename(
        columns={"y_score": "churn_probability", "y_true": "actual_churn_for_evaluation"}
    )
