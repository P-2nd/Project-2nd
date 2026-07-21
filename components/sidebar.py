# =====================================================
# components/sidebar.py
# =====================================================

import streamlit as st

from utils import display_model_name, load_config


config = load_config()

MODELS = config["models"]

EXCLUDE_FEATURES = config.get("exclude_features", [])

PAGES = [
    "고객 현황",
    "개별 고객 예측",
    "임계값·캠페인",
    "모델 비교",
    "모델 상세",
]


def navigation_sidebar():
    st.sidebar.title("🏋️ Gym Churn")
    page = st.sidebar.radio("메뉴", PAGES)
    st.sidebar.divider()
    st.sidebar.caption("최종 후보 모델")
    st.sidebar.success("LightGBM · 상위 50% 피처")
    st.sidebar.caption("기본 운영 임계값 0.50 · Test 200,000명")
    return page


# =====================================================
# Single Model
# =====================================================

def single_sidebar():
    model = st.sidebar.selectbox(
        "모델",
        MODELS,
        index=MODELS.index("lightgbm") if "lightgbm" in MODELS else 0,
    )

    dataset = st.sidebar.selectbox(
        "피처 구성",
        ["50", "full"],
        format_func=lambda value: "상위 50% 피처" if value == "50" else "전체 피처",
    )

    excluded = st.sidebar.selectbox(
        "추가 제외 피처",
        [None, *EXCLUDE_FEATURES],
        format_func=lambda value: "없음" if value is None else value,
    )

    model_key = f"{model}_{dataset}"
    if excluded:
        model_key += f"_without_{excluded}"

    return model_key


# =====================================================
# Compare
# =====================================================

def compare_sidebar():
    model_keys = []

    for model in MODELS:

        model_keys.append(f"{model}_full")
        model_keys.append(f"{model}_50")

        for feature in EXCLUDE_FEATURES:

            model_keys.append(
                f"{model}_full_without_{feature}"
            )

            model_keys.append(
                f"{model}_50_without_{feature}"
            )

    selected = st.sidebar.multiselect(
        "비교 모델",
        model_keys,
        default=["lightgbm_50", "xgboost_50"],
        format_func=display_model_name,
    )

    return selected
