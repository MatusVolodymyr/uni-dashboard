"""Main entry point — redirect to Overview."""
import streamlit as st

st.set_page_config(
    page_title="Дашборд опитування студентів",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.switch_page("pages/1_Overview.py")
