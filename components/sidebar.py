# =====================================================
# components/sidebar.py
# =====================================================

import streamlit as st

from utils import load_config


config = load_config()

MODELS = config["models"]

EXCLUDE_FEATURES = config.get("exclude_features", [])


# =====================================================
# Single Model
# =====================================================

def single_sidebar():

    st.sidebar.title("⚙️ Model Option")

    model = st.sidebar.selectbox(
        "Model",
        MODELS
    )

    dataset = st.sidebar.selectbox(
        "Dataset",
        [
            "full",
            "50"
        ]
    )

    feature = st.sidebar.radio(
        "Feature",
        [
            "Full",
            "Without"
        ]
    )

    model_key = f"{model}_{dataset}"

    if feature == "Without":

        if EXCLUDE_FEATURES:

            model_key += (
                f"_without_{EXCLUDE_FEATURES[0]}"
            )

    return model_key


# =====================================================
# Compare
# =====================================================

def compare_sidebar():

    st.sidebar.title("📊 Compare")

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
        "Models",
        model_keys,
        default=model_keys[:2]
    )

    return selected