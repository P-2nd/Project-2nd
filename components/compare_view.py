import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve

from utils import display_model_name, load_parquet, load_result


METRIC_LABELS = {
    "Accuracy": "Accuracy",
    "Precision": "Precision",
    "Recall": "Recall",
    "F1": "F1 Score",
    "ROC-AUC": "ROC-AUC",
    "PR-AUC": "PR-AUC",
    "Latency(ms)": "평균 추론 시간(ms)",
}

WEIGHT_PRESETS = {
    "이탈 탐지 우선": {"Recall": 0.40, "F1": 0.25, "PR-AUC": 0.20, "Precision": 0.10, "ROC-AUC": 0.05},
    "균형": {"Recall": 0.25, "F1": 0.25, "PR-AUC": 0.20, "Precision": 0.15, "ROC-AUC": 0.15},
    "캠페인 비용 우선": {"Precision": 0.40, "F1": 0.25, "PR-AUC": 0.20, "Recall": 0.10, "ROC-AUC": 0.05},
}


def show_compare(model_keys):
    st.title("📊 모델 비교")
    st.caption("동일한 Test 조건에서 모델의 이탈 탐지 성능과 운영 적합성을 비교합니다.")

    rows, roc_sources = _load_comparison_data(model_keys)
    if len(rows) < 2:
        st.info("결과가 존재하는 모델을 2개 이상 선택하세요.")
        return

    df = pd.DataFrame(rows)
    st.dataframe(
        df.rename(columns={"Model": "모델", **METRIC_LABELS}),
        hide_index=True,
        width="stretch",
        column_config={key: st.column_config.NumberColumn(format="%.4f") for key in METRIC_LABELS.values()},
    )

    preset = st.selectbox("모델 선정 기준", list(WEIGHT_PRESETS), index=0)
    weights = WEIGHT_PRESETS[preset]
    ranked = df.copy()
    ranked["Business Score"] = sum(ranked[key] * weight for key, weight in weights.items())
    ranked = ranked.sort_values("Business Score", ascending=False).reset_index(drop=True)
    ranked.insert(0, "순위", range(1, len(ranked) + 1))

    winner = ranked.iloc[0]
    st.success(
        f"**{preset} 기준 추천:** {winner['Model']} · Business Score {winner['Business Score']:.4f}"
    )
    st.caption("가중치: " + ", ".join(f"{key} {value:.0%}" for key, value in weights.items()))
    st.dataframe(
        ranked[["순위", "Model", "Business Score", "Recall", "Precision", "F1", "PR-AUC", "ROC-AUC"]]
        .rename(columns={"Model": "모델"}),
        hide_index=True,
        width="stretch",
    )

    metric = st.selectbox("비교 지표", list(METRIC_LABELS), index=2)
    chart = px.bar(
        df,
        x="Model",
        y=metric,
        color="Model",
        text_auto=".4f",
        labels={"Model": "모델", metric: METRIC_LABELS[metric]},
    )
    chart.update_layout(showlegend=False)
    st.plotly_chart(chart, width="stretch")

    roc_tab, pr_tab, cm_tab = st.tabs(["ROC Curve", "Precision-Recall Curve", "Confusion Matrix"])
    with roc_tab:
        st.plotly_chart(_roc_figure(roc_sources), width="stretch")
    with pr_tab:
        st.plotly_chart(_pr_figure(roc_sources), width="stretch")
    with cm_tab:
        _show_confusion_matrices(model_keys)

    st.info(
        "프로젝트의 최종 후보는 **LightGBM · 상위 50% 피처**입니다. 모델 순위는 절대적인 정답이 아니라 "
        "캠페인 목적에 따라 달라지며, 최종 결정은 임계값·대상 규모·비용을 함께 고려해야 합니다."
    )


def _load_comparison_data(model_keys):
    rows = []
    sources = {}
    for key in model_keys:
        result = load_result(key)
        roc_df = load_parquet(key)
        if result is None or roc_df is None:
            continue
        pr_auc = average_precision_score(roc_df["y_true"], roc_df["y_score"])
        label = display_model_name(key)
        rows.append(
            {
                "Model": label,
                "Accuracy": result["accuracy"],
                "Precision": result["precision"],
                "Recall": result["recall"],
                "F1": result["f1_score"],
                "ROC-AUC": result["auc_score"],
                "PR-AUC": pr_auc,
                "Latency(ms)": result["avg_latency_ms"],
            }
        )
        sources[label] = roc_df
    return rows, sources


def _roc_figure(sources):
    fig = go.Figure()
    for label, roc_df in sources.items():
        fpr, tpr, _ = roc_curve(roc_df["y_true"], roc_df["y_score"])
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=label))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="무작위 기준", line={"dash": "dash"}))
    fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    return fig


def _pr_figure(sources):
    fig = go.Figure()
    for label, roc_df in sources.items():
        precision, recall, _ = precision_recall_curve(roc_df["y_true"], roc_df["y_score"])
        fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=label))
    fig.update_layout(xaxis_title="Recall", yaxis_title="Precision")
    return fig


def _show_confusion_matrices(model_keys):
    for start in range(0, len(model_keys), 2):
        columns = st.columns(2)
        for column, key in zip(columns, model_keys[start:start + 2]):
            result = load_result(key)
            if result is None:
                continue
            cm = result["confusion_matrix"]
            with column:
                st.markdown(f"**{display_model_name(key)}**")
                st.dataframe(
                    pd.DataFrame(
                        cm,
                        index=["실제 유지", "실제 이탈"],
                        columns=["예측 유지", "예측 이탈"],
                    ),
                    width="stretch",
                )
