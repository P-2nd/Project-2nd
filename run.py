import streamlit as st

from components.campaign_view import show_campaign
from components.compare_view import show_compare
from components.overview_view import show_overview
from components.prediction_view import show_prediction
from components.sidebar import compare_sidebar, navigation_sidebar, single_sidebar
from components.single_view import show_single


st.set_page_config(
    page_title="Gym Churn Insight",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem;}
    [data-testid="stMetric"] {
        background: color-mix(in srgb, var(--background-color) 92%, #ff4b4b 8%);
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 0.75rem;
        padding: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

page = navigation_sidebar()

if page == "고객 현황":
    show_overview()
elif page == "개별 고객 예측":
    show_prediction()
elif page == "임계값·캠페인":
    show_campaign()
elif page == "모델 비교":
    selected_models = compare_sidebar()
    show_compare(selected_models)
else:
    selected_model = single_sidebar()
    show_single(selected_model)
