import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.loader import load
from src.metrics import (
    SCORE_COLS, Q01_COLS, Q03_COLS, Q05_COLS, QUESTION_LABELS,
    THEME_LABELS, course_summary,
)
from src.charts import scatter_risk, treemap_courses, horizontal_bar_questions, score_distribution_bar
from src.ui import download_csv, explain_shrunk
from src.access import access_control

st.set_page_config(page_title="Курси", page_icon="📚", layout="wide")
st.title("📚 Аналіз курсів")
st.caption(
    "Мета — знайти **курси, що потребують уваги**, а не скласти «рейтинг ганьби». "
    "Рейтинг рахується за **згладженою оцінкою** (Bayesian shrinkage): курси з малою "
    "кількістю відповідей підтягуються до середнього, тож кілька випадкових оцінок не "
    "виштовхують курс у топ проблемних. Питання «Навантаження» виключене з оцінки якості "
    "(це калібрування «важко/легко», а не якість)."
)

df_full = load()
df_full, role, scope_faculty = access_control(df_full)

fc1, fc2, fc3 = st.columns(3)
with fc1:
    faculties = ["Всі"] + sorted(df_full["faculty"].unique())
    sel_faculty = st.selectbox("Факультет", faculties)
with fc2:
    if sel_faculty != "Всі":
        specs = ["Всі"] + sorted(df_full[df_full["faculty"] == sel_faculty]["specialty"].unique())
    else:
        specs = ["Всі"]
    sel_spec = st.selectbox("Спеціальність", specs)
with fc3:
    min_n = st.slider("Мін. відповідей на курс", 5, 50, 20, 5)

df = df_full.copy()
if sel_faculty != "Всі":
    df = df[df["faculty"] == sel_faculty]
if sel_spec != "Всі":
    df = df[df["specialty"] == sel_spec]

summary = course_summary(df, min_n=min_n)

# ── Risk scatter ─────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📋 Таблиця курсів", "📈 Scatter ризику", "🗂️ Treemap"])

with tab1:
    st.subheader(f"Курси, що потребують уваги (n ≥ {min_n})")

    show_mode = st.radio(
        "Сортування",
        ["Згладжена оцінка (рекомендовано)", "Сира середня", "Найвища частка ≤3", "Найбільше відповідей"],
        horizontal=True,
    )

    if show_mode.startswith("Згладжена"):
        disp = summary.sort_values("shrunk_quality")
    elif show_mode == "Сира середня":
        disp = summary.sort_values("avg_quality")
    elif show_mode == "Найвища частка ≤3":
        disp = summary.sort_values("low_score_rate", ascending=False)
    else:
        disp = summary.sort_values("n", ascending=False)

    disp_table = disp[[
        "faculty", "specialty", "course", "n", "shrunk_quality", "avg_quality", "avg_workload",
        "low_score_rate", "lecturer", "practitioner", "weakest_question", "comment_count"
    ]].copy()
    disp_table["low_score_rate"] = (disp_table["low_score_rate"] * 100).round(1)
    for c in ("shrunk_quality", "avg_quality", "avg_workload"):
        disp_table[c] = disp_table[c].round(2)
    disp_table["comment_count"] = disp_table["comment_count"].astype(int)

    disp_table = disp_table.rename(columns={
        "faculty": "Факультет",
        "specialty": "Спеціальність",
        "course": "Курс",
        "lecturer": "Лектор",
        "practitioner": "Практик",
        "n": "n",
        "shrunk_quality": "Згладжена",
        "avg_quality": "Сира якість",
        "avg_workload": "Навантаження",
        "low_score_rate": "% ≤3",
        "weakest_question": "Слабке питання",
        "comment_count": "Корисних коментарів",
    })

    st.caption("**Згладжена** — рейтингова оцінка з поправкою на n. **Сира якість** — проста "
               "середня по 11 питаннях якості (без навантаження). **Навантаження** — окремо, "
               "низьке = курс сприймається як важкий. **% ≤3** — частка відповідей із середньою ≤3.")
    explain_shrunk()
    st.dataframe(
        disp_table.style.background_gradient(
            subset=["Згладжена", "Сира якість"], cmap="RdYlGn", vmin=1, vmax=5
        ).background_gradient(
            subset=["% ≤3"], cmap="YlOrRd", vmin=0, vmax=30
        ).format({"Згладжена": "{:.2f}", "Сира якість": "{:.2f}",
                  "Навантаження": "{:.2f}", "% ≤3": "{:.1f}%"}),
        width='stretch',
        height=500,
        hide_index=True,
    )
    download_csv(disp_table, "courses_risk.csv", key="dl_courses")

with tab2:
    st.caption("Кожна точка — курс. По горизонталі — кількість відповідей, по вертикалі — "
               "частка низьких оцінок (≤3). Цікаві курси — **праворуч вгорі**: багато відповідей "
               "і водночас багато негативу (це не випадковість).")
    if df["faculty"].nunique() == 1:
        st.plotly_chart(scatter_risk(summary, color_col="specialty", color_label="Спеціальність"),
                        width='stretch')
    else:
        st.plotly_chart(scatter_risk(summary), width='stretch')

with tab3:
    st.caption("Площа прямокутника — кількість відповідей. Показує, де зосереджений обсяг "
               "фідбеку: факультет → спеціальність → курс.")
    st.plotly_chart(treemap_courses(df), width='stretch')

st.divider()

# ── Course drill-down ─────────────────────────────────────────────────────────
st.subheader("Деталі курсу")

all_courses = sorted(df["course"].unique())
sel_course = st.selectbox("Виберіть курс", all_courses)
df_course = df[df["course"] == sel_course]

if len(df_course) < 3:
    st.warning("Замало відповідей для аналізу цього курсу.")
else:
    meta_cols = st.columns(4)
    meta_cols[0].metric("Відповідей", len(df_course))
    meta_cols[1].metric("Лектор", df_course["lecturer"].mode().iloc[0] if len(df_course) else "—")
    meta_cols[2].metric("Практик", df_course["practitioner"].mode().iloc[0] if len(df_course) else "—")
    meta_cols[3].metric("Коментарів", int(df_course["has_comment"].sum()))

    q_means = df_course[SCORE_COLS].mean()
    q_global = df[SCORE_COLS].mean()

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Профіль курсу")
        fig_prof = horizontal_bar_questions(q_means, QUESTION_LABELS, title=sel_course)
        st.plotly_chart(fig_prof, width='stretch')

    with c2:
        st.subheader("Порівняння з загальним середнім")
        compare_df = pd.DataFrame({
            "Питання": [QUESTION_LABELS[c] for c in SCORE_COLS],
            "Курс": q_means.values,
            "Загалом": q_global.values,
        })
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Bar(
            y=compare_df["Питання"],
            x=compare_df["Курс"],
            orientation="h",
            name="Курс",
            marker_color="#1f77b4",
        ))
        fig_cmp.add_trace(go.Scatter(
            y=compare_df["Питання"],
            x=compare_df["Загалом"],
            mode="markers",
            name="Загалом",
            marker=dict(color="black", size=8, symbol="diamond"),
        ))
        fig_cmp.update_layout(
            xaxis=dict(range=[1, 5.4]),
            height=380,
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig_cmp, width='stretch')

    # Score distribution per block
    st.subheader("Розподіл оцінок по блоках")
    b1, b2, b3 = st.columns(3)
    for col_container, cols, label in [
        (b1, Q01_COLS, "Дисципліна"),
        (b2, Q03_COLS, "Лектор"),
        (b3, Q05_COLS, "Практик"),
    ]:
        scores = df_course[cols].values.flatten()
        scores = scores[~np.isnan(scores)].astype(int)
        with col_container:
            st.plotly_chart(
                score_distribution_bar(pd.Series(scores), title=label),
                width='stretch',
            )

    # Embedded comments for this course
    st.subheader("Що писали студенти")
    from src.ui import render_comments
    render_comments(df_course, key_prefix="course")
