# =====================================================
# components/single_view.py
# =====================================================

import streamlit as st

from utils import (
    load_result,
    load_params,
    get_plot_paths
)


def show_single(model_key):

    result = load_result(model_key)

    if result is None:

        st.error("결과 파일이 없습니다.")

        return

    st.title(result["model_name"])

    st.subheader("모델 성능 안내")

    st.markdown(
        """
선택한 모델이 회원 이탈을 얼마나 잘 예측하는지 확인하는 화면입니다.

- **Accuracy**: 전체 회원 중 예측이 맞은 비율
- **Precision**: 이탈로 예측한 회원 중 실제 이탈한 회원의 비율
- **Recall**: 실제 이탈 회원 중 모델이 찾아낸 비율
- **F1 Score**: Precision과 Recall의 균형 지표
- **ROC AUC**: 여러 임계값에서 이탈 회원과 유지 회원을 구분하는 성능
"""
    )

    st.caption("이탈 회원을 놓치는 비용이 크다면 Accuracy와 함께 Recall, F1 Score를 확인하세요.")

    st.divider()

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Accuracy (정확도)",
        f"{result['accuracy']:.4f}"
    )

    c2.metric(
        "Precision (정밀도)",
        f"{result['precision']:.4f}"
    )

    c3.metric(
        "Recall (재현율)",
        f"{result['recall']:.4f}"
    )

    c4, c5, c6 = st.columns(3)

    c4.metric(
        "F1 Score (F1 점수)",
        f"{result['f1_score']:.4f}",
        help=(
            "Precision(정밀도)과 Recall(재현율)의 조화평균입니다. "
            "두 값 중 낮은 값의 영향을 크게 받아, 이탈 탐지와 오탐 사이의 균형을 확인할 때 사용합니다."
        )
    )

    c5.metric(
        "ROC AUC (ROC-AUC)",
        f"{result['auc_score']:.4f}",
        help=(
            "여러 임계값에서 이탈 회원과 유지 회원을 구분하는 성능입니다. "
            "0.5는 무작위 분류 수준이며, 1에 가까울수록 두 집단을 더 잘 구분합니다."
        )
    )

    c6.metric(
        "Latency (지연 시간, ms)",
        f"{result['avg_latency_ms']:.4f}"
    )

    st.divider()

    c1, c2 = st.columns(2)

    plots = get_plot_paths(model_key)

    with c1:

        st.subheader("Confusion Matrix (혼동 행렬)")

        if plots["cm"].exists():

            st.image(
                str(plots["cm"]),
                width="stretch"
            )

        else:

            st.warning("Confusion Matrix 없음")

    with c2:

        st.subheader("ROC Curve (ROC 곡선)")

        if plots["roc"].exists():

            st.image(
                str(plots["roc"]),
                width="stretch"
            )

        else:

            st.warning("ROC Plot 없음")

    st.divider()

    st.subheader("Hyper Parameters (하이퍼파라미터)")

    params = load_params(model_key)

    if params:

        st.json(params)

    else:

        st.info("Parameter 없음")

    st.divider()

    st.subheader("Experiment (실험 정보)")

    c1, c2 = st.columns(2)

    c1.write(f"Total Samples (전체 샘플 수): {result["total_samples"]:.4f}")

    c2.write(f"Inference (추론 시간, ms): {result['avg_latency_ms']:.4f}")

    st.write(
        "Execution (실행 시간, sec):",
        f"{result['total_time_sec']:.4f}"
    )

    st.write(
        "Feature Set (피처 구성):",
        result["experiment"]["feature_set"]
    )

    st.write(
        "Comparison Axis (비교 기준):",
        result["experiment"]["comparison_axis"]
    )

    st.divider()

    st.subheader("Confusion Matrix Value (혼동 행렬 값)")

    cm = result["confusion_matrix"]

    st.table({
        "": ["Actual 0", "Actual 1"],
        "Predicted 0": [cm[0][0], cm[1][0]],
        "Predicted 1": [cm[0][1], cm[1][1]]
    })

    st.divider()

    st.subheader("Parquet (ROC 데이터 파일)")

    st.code(
        result["parquet"]["path"]
    )
