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

    st.title("Model Compare")

    # =====================================================
    # Metric Table
    # =====================================================

    st.divider()
    st.subheader("Metric Table")

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    # =====================================================
    # Best Model
    # =====================================================

    st.divider()
    st.subheader("🏆 Best Model")

    best_df = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC AUC",
            "Latency(ms)"
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
    })

    st.dataframe(
        best_df,
        hide_index=True,
        use_container_width=True
    )

    # =====================================================
    # Overall Ranking
    # =====================================================

    st.divider()
    st.subheader("🏆 Overall Ranking")

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
        ],
        hide_index=True,
        use_container_width=True
    )

    # =====================================================
    # Metric Compare
    # =====================================================

    st.divider()

    metric = st.selectbox(
        "Compare Metric",
        [
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "AUC",
            "Latency(ms)"
        ]
    )

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(
        df["Model"],
        df[metric]
    )

    ax.set_ylabel(metric)

    plt.xticks(rotation=20)

    st.pyplot(fig)

    # =====================================================
    # ROC Curve
    # =====================================================

    st.divider()
    st.subheader("ROC Curve")

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
    st.subheader("Confusion Matrix")

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
    st.subheader("Saved Plot")

    cols = st.columns(len(model_keys))

    for col, key in zip(cols, model_keys):

        plots = get_plot_paths(key)

        with col:

            st.write(key)

            if plots["roc"].exists():
                st.image(
                    str(plots["roc"]),
                    caption="ROC",
                    use_container_width=True
                )

            if plots["cm"].exists():
                st.image(
                    str(plots["cm"]),
                    caption="Confusion Matrix",
                    use_container_width=True
                )