"""Cached data loader shared by all pages."""
import pandas as pd
import streamlit as st
from pathlib import Path

PARQUET = Path(__file__).parent.parent / "data" / "feedback.parquet"


@st.cache_data
def load() -> pd.DataFrame:
    return pd.read_parquet(PARQUET)
