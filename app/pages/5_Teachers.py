import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.loader import load
from src.metrics import SCORE_COLS, Q01_COLS, Q03_COLS, Q05_COLS, QUESTION_LABELS, shrunk_mean
from src.charts import horizontal_bar_questions, score_distribution_bar
from src.ui import download_csv, render_comments, explain_shrunk
from src.access import access_control

st.set_page_config(page_title="Викладачі", page_icon="👩‍🏫", layout="wide")
st.title("👩‍🏫 Аналіз викладачів")
st.info(
    "⚠️ Ці дані — для **підтримки викладачів**, а не для «рейтингу ганьби». "
    "Оцінки показуються лише за достатньої кількості відповідей, рейтинг згладжено "
    "за обсягом вибірки, і завжди подається розподіл, а не лише середнє. "
    "Персональні відповіді студентів не розкриваються."
)

df_full = load()
df_full, role, scope_faculty = access_control(df_full)

with st.sidebar:
    st.header("Фільтри")
    faculties = ["Всі"] + sorted(df_full["faculty"].unique())
    sel_faculty = st.selectbox("Факультет", faculties)
    min_n = st.slider("Мін. відповідей на викладача", 5, 50, 20, 5)
    teacher_type = st.radio("Тип викладача", ["Лектор", "Практик"])

df = df_full.copy()
if sel_faculty != "Всі":
    df = df[df["faculty"] == sel_faculty]

# ── Aggregation per teacher type ─────────────────────────────────────────────
if teacher_type == "Лектор":
    teacher_col = "lecturer"
    score_cols = Q03_COLS
    avg_col = "avg_lecturer"
else:
    teacher_col = "practitioner"
    score_cols = Q05_COLS
    avg_col = "avg_practitioner"

# Filter out empty teacher names
df_t = df[df[teacher_col].str.len() > 0].copy()

teacher_agg = df_t.groupby(teacher_col).agg(
    n=(avg_col, "count"),
    avg=(avg_col, "mean"),
    comment_count=("comment_useful", "sum"),
    faculty=("faculty", lambda s: s.mode().iloc[0] if len(s) else ""),
).reset_index()

low_rates = (
    df_t.groupby(teacher_col)
    .apply(lambda x: (x[score_cols].values.flatten() <= 3).mean() * 100, include_groups=False)
    .reset_index(name="low_rate")
)
teacher_agg = teacher_agg.merge(low_rates, on=teacher_col, how="left")

# Shrink toward the global mean for this teacher type so small samples don't dominate
prior = df_t[avg_col].mean()
teacher_agg["shrunk"] = shrunk_mean(teacher_agg["avg"], teacher_agg["n"], prior, strength=20.0)
teacher_agg = teacher_agg[teacher_agg["n"] >= min_n].sort_values("shrunk", ascending=True)

st.caption(f"Показано {len(teacher_agg)} викладачів з n ≥ {min_n} відповідей. "
           f"Сортування — за згладженою оцінкою (поправка на обсяг вибірки).")
explain_shrunk()

# ── Top/bottom table ──────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📋 Рейтинг викладачів", "🔍 Профіль викладача"])

with tab1:
    col_low, col_high = st.columns(2)

    def render_table(data, title):
        disp = data[[teacher_col, "faculty", "n", "shrunk", "avg"]].copy()
        disp["shrunk"] = disp["shrunk"].round(2)
        disp["avg"] = disp["avg"].round(2)
        disp = disp.rename(columns={
            teacher_col: "Викладач",
            "faculty": "Факультет",
            "n": "n",
            "shrunk": "Згладжена",
            "avg": "Сира",
        })
        st.markdown(f"**{title}**")
        st.dataframe(
            disp.style.background_gradient(subset=["Згладжена"], cmap="RdYlGn", vmin=1, vmax=5)
            .format({"Згладжена": "{:.2f}", "Сира": "{:.2f}"}),
            width='stretch',
            hide_index=True,
        )

    with col_low:
        render_table(teacher_agg.head(20), f"Потребують уваги — {teacher_type}")
    with col_high:
        render_table(teacher_agg.tail(20).sort_values("shrunk", ascending=False), f"Найвищі оцінки — {teacher_type}")

    st.divider()
    st.subheader(f"Усі викладачі ({teacher_type.lower()}и) з n ≥ {min_n}")
    st.caption("Повний список з можливістю сортування за будь-якою колонкою (клік по заголовку).")
    disp_full = teacher_agg[[teacher_col, "faculty", "n", "shrunk", "avg", "low_rate", "comment_count"]].copy()
    disp_full["shrunk"] = disp_full["shrunk"].round(2)
    disp_full["avg"] = disp_full["avg"].round(2)
    disp_full["low_rate"] = disp_full["low_rate"].round(1)
    disp_full["comment_count"] = disp_full["comment_count"].astype(int)
    disp_full = disp_full.rename(columns={
        teacher_col: "Викладач",
        "faculty": "Факультет",
        "n": "Відповідей",
        "shrunk": "Згладжена",
        "avg": "Сира середня",
        "low_rate": "% ≤3",
        "comment_count": "Корисних коментарів",
    })
    st.dataframe(
        disp_full.style.background_gradient(subset=["Згладжена"], cmap="RdYlGn", vmin=1, vmax=5)
        .background_gradient(subset=["% ≤3"], cmap="YlOrRd", vmin=0, vmax=30)
        .format({"Згладжена": "{:.2f}", "Сира середня": "{:.2f}", "% ≤3": "{:.1f}%"}),
        width='stretch',
        height=450,
        hide_index=True,
    )
    download_csv(disp_full, f"teachers_{teacher_type}.csv", key="dl_teachers")

with tab2:
    available = sorted(df_t[df_t[teacher_col].isin(teacher_agg[teacher_col])][teacher_col].unique())
    if not available:
        st.warning("Немає викладачів з достатньою кількістю відповідей.")
    else:
        sel_teacher = st.selectbox("Виберіть викладача", available)
        df_teacher = df_t[df_t[teacher_col] == sel_teacher]

        mc = st.columns(4)
        mc[0].metric("Відповідей", len(df_teacher))
        mc[1].metric("Середня оцінка", f"{df_teacher[avg_col].mean():.2f}")
        low_pct = (df_teacher[score_cols].values.flatten() <= 3).mean() * 100
        mc[2].metric("Оцінок ≤ 3", f"{low_pct:.1f}%")
        mc[3].metric("Коментарів", int(df_teacher["has_comment"].sum()))

        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Профіль питань")
            q_means = df_teacher[score_cols].mean()
            fig_prof = horizontal_bar_questions(q_means, QUESTION_LABELS, title=sel_teacher)
            st.plotly_chart(fig_prof, width='stretch')

        with c2:
            st.subheader("Розподіл оцінок")
            scores = df_teacher[score_cols].values.flatten()
            scores = scores[~np.isnan(scores)].astype(int)
            st.plotly_chart(
                score_distribution_bar(pd.Series(scores), title=f"{sel_teacher} — розподіл"),
                width='stretch',
            )

        # Courses taught
        st.subheader(f"Курси — {sel_teacher}")
        courses_df = df_teacher.groupby("course").agg(
            n=("avg_overall", "count"),
            avg_discipline=("avg_discipline", "mean"),
            avg_teacher=(avg_col, "mean"),
        ).reset_index().sort_values("avg_teacher", ascending=True)
        courses_df["avg_discipline"] = courses_df["avg_discipline"].round(2)
        courses_df["avg_teacher"] = courses_df["avg_teacher"].round(2)
        courses_df = courses_df.rename(columns={
            "course": "Курс",
            "n": "Відповідей",
            "avg_discipline": "Дисципліна",
            "avg_teacher": f"Оцінка ({teacher_type})",
        })
        st.dataframe(
            courses_df.style.background_gradient(
                subset=[f"Оцінка ({teacher_type})"], cmap="RdYlGn", vmin=1, vmax=5
            ).format({"Дисципліна": "{:.2f}", f"Оцінка ({teacher_type})": "{:.2f}"}),
            width='stretch',
            hide_index=True,
        )

        # Embedded comments for this teacher
        st.subheader("Що писали студенти")
        render_comments(df_teacher, key_prefix="teacher")
