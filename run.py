# =====================================================
# streamlit_app.py
# =====================================================

import streamlit as st

from components.sidebar import (
    single_sidebar,
    compare_sidebar
)

from components.single_view import (
    show_single
)

from components.compare_view import (
    show_compare
)

from components.prediction_view import (
    show_prediction
)


# =====================================================
# Page Config
# =====================================================

st.set_page_config(

    page_title="Gym Churn Dashboard",

    page_icon="🏋️",

    layout="wide"

)


# =====================================================
# Sidebar
# =====================================================

page = st.sidebar.radio(

    "Menu",

    [

        "Single Model",

        "Compare Models",

        "Churn Prediction (회원 이탈 예측)"

    ]

)


# =====================================================
# Single
# =====================================================

if page == "Single Model":

    model_key = single_sidebar()

    show_single(model_key)


# =====================================================
# Compare
# =====================================================

elif page == "Compare Models":

    model_keys = compare_sidebar()

    show_compare(model_keys)


# =====================================================
# Churn Prediction
# =====================================================

else:

    show_prediction()
