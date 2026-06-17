"""Cached data loader shared by all pages."""
import pandas as pd
import streamlit as st
from pathlib import Path

PARQUET = Path(__file__).parent.parent / "data" / "feedback.parquet"
TEACHERS_PARQUET = Path(__file__).parent.parent / "data" / "teachers.parquet"


@st.cache_data
def load() -> pd.DataFrame:
    return pd.read_parquet(PARQUET)


@st.cache_data
def load_teachers() -> pd.DataFrame:
    """Long table: one row per response × canonical teacher (with role)."""
    return pd.read_parquet(TEACHERS_PARQUET)
