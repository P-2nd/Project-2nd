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

    st.divider()

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Accuracy",
        f"{result['accuracy']:.4f}"
    )

    c2.metric(
        "Precision",
        f"{result['precision']:.4f}"
    )

    c3.metric(
        "Recall",
        f"{result['recall']:.4f}"
    )

    c4, c5, c6 = st.columns(3)

    c4.metric(
        "F1 Score",
        f"{result['f1_score']:.4f}"
    )

    c5.metric(
        "ROC AUC",
        f"{result['auc_score']:.4f}"
    )

    c6.metric(
        "Latency(ms)",
        f"{result['avg_latency_ms']:.4f}"
    )

    st.divider()

    c1, c2 = st.columns(2)

    plots = get_plot_paths(model_key)

    with c1:

        st.subheader("Confusion Matrix")

        if plots["cm"].exists():

            st.image(
                str(plots["cm"]),
                use_container_width=True
            )

        else:

            st.warning("Confusion Matrix 없음")

    with c2:

        st.subheader("ROC Curve")

        if plots["roc"].exists():

            st.image(
                str(plots["roc"]),
                use_container_width=True
            )

        else:

            st.warning("ROC Plot 없음")

    st.divider()

    st.subheader("Hyper Parameters")

    params = load_params(model_key)

    if params:

        st.json(params)

    else:

        st.info("Parameter 없음")

    st.divider()

    st.subheader("Experiment")

    c1, c2 = st.columns(2)

    c1.write(f"Total Samples : {result["total_samples"]:.4f}")

    c2.write(f"Inference(ms) : {result['avg_latency_ms']:.4f}")

    st.write(
        "Execution(sec) :",
        f"{result['total_time_sec']:.4f}"
    )

    st.write(
        "Feature Set :",
        result["experiment"]["feature_set"]
    )

    st.write(
        "Comparison Axis :",
        result["experiment"]["comparison_axis"]
    )

    st.divider()

    st.subheader("Confusion Matrix Value")

    cm = result["confusion_matrix"]

    st.table({
        "": ["Actual 0", "Actual 1"],
        "Predicted 0": [cm[0][0], cm[1][0]],
        "Predicted 1": [cm[0][1], cm[1][1]]
    })

    st.divider()

    st.subheader("Parquet")

    st.code(
        result["parquet"]["path"]
    )