# =====================================================
# components/compare_view.py
# =====================================================

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve

from utils import (
    load_result,
    load_parquet,
    get_plot_paths
)


METRIC_LABELS = {
    "Accuracy": "Accuracy (정확도)",
    "Precision": "Precision (정밀도)",
    "Recall": "Recall (재현율)",
    "F1": "F1 Score (F1 점수)",
    "AUC": "ROC AUC (ROC-AUC)",
    "Latency(ms)": "Latency (지연 시간, ms)",
}


def show_compare(model_keys):

    if len(model_keys) < 2:
        st.info("2개 이상의 모델을 선택하세요.")
        return

    # =====================================================
    # Result Load
    # =====================================================

    rows = []

    for key in model_keys:

        result = load_result(key)

        if result is None:
            continue

        rows.append({
            "Model": result["model_name"],
            "Accuracy": result["accuracy"],
            "Precision": result["precision"],
            "Recall": result["recall"],
            "F1": result["f1_score"],
            "AUC": result["auc_score"],
            "Latency(ms)": result["avg_latency_ms"]
        })

    df = pd.DataFrame(rows)

    st.title("Model Compare (모델 비교)")

    # =====================================================
    # Metric Table
    # =====================================================

    st.divider()
    st.subheader("Metric Table (성능 지표 표)")

    st.dataframe(
        df.rename(columns={"Model": "Model (모델)", **METRIC_LABELS}),
        hide_index=True
    )

    # =====================================================
    # Best Model
    # =====================================================

    st.divider()
    st.subheader("🏆 Best Model (지표별 최고 모델)")

    best_df = pd.DataFrame({
        "Metric": [
            METRIC_LABELS["Accuracy"],
            METRIC_LABELS["Precision"],
            METRIC_LABELS["Recall"],
            METRIC_LABELS["F1"],
            METRIC_LABELS["AUC"],
            METRIC_LABELS["Latency(ms)"],
        ],
        "Best Model": [
            df.loc[df["Accuracy"].idxmax(), "Model"],
            df.loc[df["Precision"].idxmax(), "Model"],
            df.loc[df["Recall"].idxmax(), "Model"],
            df.loc[df["F1"].idxmax(), "Model"],
            df.loc[df["AUC"].idxmax(), "Model"],
            df.loc[df["Latency(ms)"].idxmin(), "Model"]
        ],
        "Value": [
            f'{df["Accuracy"].max():.4f}',
            f'{df["Precision"].max():.4f}',
            f'{df["Recall"].max():.4f}',
            f'{df["F1"].max():.4f}',
            f'{df["AUC"].max():.4f}',
            f'{df["Latency(ms)"].min():.4f}'
        ]
    }).rename(
        columns={
            "Metric": "Metric (지표)",
            "Best Model": "Best Model (최고 모델)",
            "Value": "Value (값)",
        }
    )

    st.dataframe(
        best_df,
        hide_index=True
    )

    # =====================================================
    # Overall Ranking
    # =====================================================

    st.divider()
    st.subheader("🏆 Overall Ranking (종합 순위)")

    ranking = df.copy()

    ranking["Accuracy Rank"] = ranking["Accuracy"].rank(ascending=False, method="min")
    ranking["Precision Rank"] = ranking["Precision"].rank(ascending=False, method="min")
    ranking["Recall Rank"] = ranking["Recall"].rank(ascending=False, method="min")
    ranking["F1 Rank"] = ranking["F1"].rank(ascending=False, method="min")
    ranking["AUC Rank"] = ranking["AUC"].rank(ascending=False, method="min")
    ranking["Latency Rank"] = ranking["Latency(ms)"].rank(ascending=True, method="min")

    ranking["Total Score"] = (
            ranking["Accuracy Rank"] +
            ranking["Precision Rank"] +
            ranking["Recall Rank"] +
            ranking["F1 Rank"] +
            ranking["AUC Rank"] +
            ranking["Latency Rank"]
    )

    ranking = ranking.sort_values("Total Score")

    ranking.insert(
        0,
        "Rank",
        range(1, len(ranking) + 1)
    )

    st.dataframe(
        ranking[
            [
                "Rank",
                "Model",
                "Total Score",
                "Accuracy",
                "Precision",
                "Recall",
                "F1",
                "AUC",
                "Latency(ms)"
            ]
        ].rename(
            columns={
                "Rank": "Rank (순위)",
                "Model": "Model (모델)",
                "Total Score": "Total Score (종합 점수)",
                **METRIC_LABELS,
            }
        ),
        hide_index=True
    )

    # =====================================================
    # Metric Compare
    # =====================================================

    st.divider()

    metric = st.selectbox(
        "Compare Metric (비교 지표)",
        list(METRIC_LABELS.values())
    )

    metric_key = next(
        key for key, label in METRIC_LABELS.items()
        if label == metric
    )

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(
        df["Model"],
        df[metric_key]
    )

    ax.set_ylabel(metric_key)

    plt.xticks(rotation=20)

    st.pyplot(fig)

    # =====================================================
    # ROC Curve
    # =====================================================

    st.divider()
    st.subheader("ROC Curve (ROC 곡선)")

    fig, ax = plt.subplots(figsize=(7, 7))

    for key in model_keys:

        result = load_result(key)
        roc_df = load_parquet(key)

        if roc_df is None:
            continue

        fpr, tpr, _ = roc_curve(
            roc_df["y_true"],
            roc_df["y_score"]
        )

        ax.plot(
            fpr,
            tpr,
            label=f"{result['model_name']} ({result['auc_score']:.3f})"
        )

    ax.plot([0, 1], [0, 1], "--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()

    st.pyplot(fig)

    # =====================================================
    # Confusion Matrix
    # =====================================================

    st.divider()
    st.subheader("Confusion Matrix (혼동 행렬)")

    cols = st.columns(len(model_keys))

    for col, key in zip(cols, model_keys):

        result = load_result(key)

        cm = pd.DataFrame(
            result["confusion_matrix"],
            index=["Actual 0", "Actual 1"],
            columns=["Pred 0", "Pred 1"]
        )

        fig, ax = plt.subplots(figsize=(4, 4))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax
        )

        ax.set_title(result["model_name"])

        col.pyplot(fig)



    # =====================================================
    # Saved Plot
    # =====================================================

    st.divider()
    st.subheader("Saved Plot (저장된 그래프)")

    cols = st.columns(len(model_keys))

    for col, key in zip(cols, model_keys):

        plots = get_plot_paths(key)

        with col:

            st.write(key)

            if plots["roc"].exists():
                st.image(
                    str(plots["roc"]),
                    caption="ROC Curve (ROC 곡선)",
                    width="stretch"
                )

            if plots["cm"].exists():
                st.image(
                    str(plots["cm"]),
                    caption="Confusion Matrix (혼동 행렬)",
                    width="stretch"
                )
