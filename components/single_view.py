import pandas as pd
import streamlit as st

from utils import (
    calculate_threshold_metrics,
    display_model_name,
    get_plot_paths,
    load_feature_importance,
    load_params,
    load_parquet,
    load_result,
)


def show_single(model_key):
    result = load_result(model_key)
    if result is None:
        st.error("선택한 모델의 결과 파일이 없습니다.")
        return

    st.title(f"🔎 {display_model_name(model_key)}")
    st.caption("동일한 Test 데이터에서 계산한 성능과 모델 산출물을 상세히 확인합니다.")

    roc_df = load_parquet(model_key)
    threshold_metrics = calculate_threshold_metrics(roc_df, 0.5) if roc_df is not None else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{result['accuracy']:.4f}")
    c2.metric("Precision", f"{result['precision']:.4f}")
    c3.metric("Recall", f"{result['recall']:.4f}")
    c4.metric("F1 Score", f"{result['f1_score']:.4f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("ROC-AUC", f"{result['auc_score']:.4f}")
    c6.metric("PR-AUC", f"{threshold_metrics['pr_auc']:.4f}" if threshold_metrics else "N/A")
    c7.metric("평균 추론 시간", f"{result['avg_latency_ms']:.4f} ms")
    c8.metric("Test 표본", f"{result['total_samples']:,}명")

    if model_key == "lightgbm_50":
        st.success(
            "현재 최종 후보 모델입니다. 전체 피처와 유사한 성능을 11개 피처로 달성했지만, "
            "임계값 0.50에서 Recall이 약 38%이므로 운영 목적에 맞는 조정이 필요합니다."
        )

    tab1, tab2, tab3 = st.tabs(["평가 그래프", "주요 피처", "실험 정보"])
    with tab1:
        _show_evaluation_plots(model_key, result)
    with tab2:
        _show_feature_importance(model_key)
    with tab3:
        _show_experiment(model_key, result)


def _show_evaluation_plots(model_key, result):
    left, right = st.columns(2)
    plots = get_plot_paths(model_key)
    with left:
        st.subheader("Confusion Matrix")
        if plots["cm"].exists():
            st.image(str(plots["cm"]), width="stretch")
        else:
            st.warning("혼동행렬 이미지가 없습니다.")
    with right:
        st.subheader("ROC Curve")
        if plots["roc"].exists():
            st.image(str(plots["roc"]), width="stretch")
        else:
            st.warning("ROC 이미지가 없습니다.")

    cm = result["confusion_matrix"]
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    st.dataframe(
        pd.DataFrame(
            [
                ["TP", tp, "실제 이탈 고객을 이탈로 탐지"],
                ["FP", fp, "유지 고객을 이탈로 잘못 분류"],
                ["FN", fn, "실제 이탈 고객을 놓침"],
                ["TN", tn, "유지 고객을 정확히 분류"],
            ],
            columns=["구분", "인원", "해석"],
        ),
        hide_index=True,
        width="stretch",
    )


def _show_feature_importance(model_key):
    with st.spinner("피처 중요도를 불러오는 중입니다."):
        importance = load_feature_importance(model_key)
    if importance is None:
        st.info(
            "이 모델은 피처 중요도를 제공하지 않거나 모델 파일이 너무 커 자동 로드를 건너뛰었습니다. "
            "LightGBM·XGBoost·Logistic Regression 모델에서 확인할 수 있습니다."
        )
        return
    st.bar_chart(importance.set_index("Feature"), horizontal=True)
    st.dataframe(importance, hide_index=True, width="stretch")
    st.caption("피처 중요도는 연관성과 예측 기여도를 나타내며 인과관계를 의미하지 않습니다.")


def _show_experiment(model_key, result):
    feature_set = "상위 50% 피처(11개)" if "_50" in model_key else "전체 피처(22개)"
    left, right = st.columns(2)
    left.write(f"**피처 구성:** {feature_set}")
    left.write(f"**Test 표본:** {result['total_samples']:,}명")
    right.write(f"**전체 추론 시간:** {result['total_time_sec']:.4f}초")
    right.write(f"**기본 임계값:** 0.50")

    params = load_params(model_key)
    with st.expander("하이퍼파라미터"):
        if params:
            st.json(params)
        else:
            st.info("저장된 하이퍼파라미터가 없습니다.")
    with st.expander("평가 원천 데이터"):
        st.code(result.get("parquet", {}).get("path", "ROC 데이터 경로 없음"))
