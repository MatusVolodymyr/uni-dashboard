import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.loader import load
from src.metrics import SCORE_COLS
from src.access import access_control

st.set_page_config(page_title="Якість даних", page_icon="🧪", layout="wide")
st.title("🧪 Якість даних")
st.caption(
    "Сторінка для прозорості: наскільки даним можна довіряти і де їх межі. "
    "Це важливо для коректної інтерпретації всіх інших сторінок."
)

df = load()
df, role, scope_faculty = access_control(df)

# ── Coverage KPIs ─────────────────────────────────────────────────────────────
total = len(df)
n_score_cells = total * len(SCORE_COLS)
missing = int(df[SCORE_COLS].isna().sum().sum())
useful = int(df["comment_useful"].sum())
trivial = int((df["has_comment"] & ~df["comment_useful"]).sum())

k = st.columns(5)
k[0].metric("Відповідей", f"{total:,}")
k[1].metric("Курсів", f"{df['course'].nunique():,}")
k[2].metric("Пропущених оцінок", f"{missing:,}", help=f"з {n_score_cells:,} клітинок")
k[3].metric("Змістовних коментарів", f"{useful:,}")
k[4].metric("Тривіальних коментарів", f"{trivial:,}")

st.divider()

# ── Reliability: distribution of n per course ─────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Надійність: розподіл n по курсах")
    st.caption("Скільки курсів потрапляє в кожен діапазон кількості відповідей. "
               "Курси з малим n (≤9) статистично ненадійні й приховані в порівняннях за замовчуванням.")
    course_n = df.groupby("course").size()
    bins = [
        ("1–4", ((course_n >= 1) & (course_n <= 4)).sum()),
        ("5–9", ((course_n >= 5) & (course_n <= 9)).sum()),
        ("10–19", ((course_n >= 10) & (course_n <= 19)).sum()),
        ("20–49", ((course_n >= 20) & (course_n <= 49)).sum()),
        ("50+", (course_n >= 50).sum()),
    ]
    labels = [b[0] for b in bins]
    values = [int(b[1]) for b in bins]
    colors = ["#d62728", "#ff7f0e", "#bcbd22", "#2ca02c", "#1f77b4"]
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors,
                           text=values, textposition="outside"))
    fig.update_layout(xaxis_title="Кількість відповідей на курс", yaxis_title="Курсів",
                      height=340, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width='stretch')

    below_20 = int((course_n < 20).sum())
    st.caption(f"**{below_20} з {len(course_n)} курсів** мають < 20 відповідей "
               f"({below_20 / len(course_n) * 100:.0f}%).")

with col2:
    st.subheader("Коментарі: змістовні vs тривіальні")
    st.caption("Близько 42% відповідей мають коментар, але частина з них — беззмістовні "
               "(«так», «-», «все добре»). Для аналізу враховуються лише змістовні.")
    no_comment = total - int(df["has_comment"].sum())
    parts = [
        ("Без коментаря", no_comment, "#d3d3d3"),
        ("Тривіальні", trivial, "#bcbd22"),
        ("Змістовні", useful, "#2ca02c"),
    ]
    fig2 = go.Figure(go.Bar(
        x=[p[1] for p in parts], y=[p[0] for p in parts], orientation="h",
        marker_color=[p[2] for p in parts],
        text=[f"{p[1]:,}" for p in parts], textposition="outside",
    ))
    fig2.update_layout(xaxis_title="Відповідей", height=340,
                       margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig2, width='stretch')

st.divider()

# ── Known limitations ─────────────────────────────────────────────────────────
st.subheader("Відомі обмеження даних")
st.markdown(
    """
- **Поле «Група» повністю порожнє** — аналіз по групах неможливий.
- **Студенти анонімізовані** (`Анонімний1`, `Анонімний2`…) — неможливо відстежити окремого
  респондента чи побудувати достовірний підрахунок унікальних студентів.
- **Немає дати/семестру** — тренди в часі поки недоступні; з'являться, коли дані
  поповнюватимуться по семестрах.
- **Сильне зміщення до 5** (≈80% — п'ятірки) — середня оцінка малоінформативна;
  спираємось на розподіл, частку низьких оцінок і відхилення від середнього.
- **Тональність коментарів** визначається за середньою оцінкою відповіді (проксі), а не
  повноцінним NLP — це наближення.
"""
)

st.divider()

# ── Per-faculty coverage table ────────────────────────────────────────────────
st.subheader("Покриття по факультетах")
fac = df.groupby("faculty").agg(
    responses=("avg_overall", "count"),
    courses=("course", "nunique"),
    comment_rate=("comment_useful", "mean"),
).reset_index().sort_values("responses", ascending=False)
fac["comment_rate"] = (fac["comment_rate"] * 100).round(1)
fac = fac.rename(columns={
    "faculty": "Факультет", "responses": "Відповідей",
    "courses": "Курсів", "comment_rate": "% змістовних коментарів",
})
st.dataframe(fac.style.format({"% змістовних коментарів": "{:.1f}%"}),
             width='stretch', hide_index=True, height=400)
