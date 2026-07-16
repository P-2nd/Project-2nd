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

        "Compare Models"

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

else:

    model_keys = compare_sidebar()

    show_compare(model_keys)