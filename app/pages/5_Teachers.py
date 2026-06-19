import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
from src.loader import load, load_teachers
from src.metrics import (
    QUESTION_LABELS, ROLE_CONFIG, teacher_summary, top_teachers, teacher_question_ranking,
)
from src.charts import horizontal_bar_questions, score_distribution_bar
from src.ui import download_csv, render_comments, explain_shrunk
from src.access import access_control

st.set_page_config(page_title="Викладачі", page_icon="👩‍🏫", layout="wide")
st.title("👩‍🏫 Аналіз викладачів")
st.info(
    "⚠️ Дані — для **підтримки викладачів**, а не «рейтингу ганьби». Імена студенти "
    "вписували вручну, тож вони **автоматично дедупльовані** (об'єднано різні написання "
    "та ініціали, розділено співвикладачів). Лектори (Q02) і практики (Q04) рахуються "
    "**окремо**. Показуються лише викладачі з достатнім n; рейтинг згладжено за обсягом вибірки."
)

# Access scope (faculty lock for deans) comes from the feedback table
_, role_access, scope_faculty = access_control(load())
teachers_all = load_teachers()
if scope_faculty:
    teachers_all = teachers_all[teachers_all["faculty"] == scope_faculty]

fc1, fc2, fc3 = st.columns([2, 1, 1])
with fc1:
    faculties = ["Всі"] + sorted(teachers_all["faculty"].unique())
    sel_faculty = st.selectbox("Факультет", faculties)
with fc2:
    min_n = st.slider("Мін. відповідей на викладача", 5, 50, 20, 5)
with fc3:
    teacher_type = st.radio("Категорія", ["Лектор", "Практик"], horizontal=True,
                            help="Q02 (лектор) і Q04 (практик) — різні ролі, рахуються окремо.")

tdf = teachers_all if sel_faculty == "Всі" else teachers_all[teachers_all["faculty"] == sel_faculty]
score_col, qcols = ROLE_CONFIG[teacher_type]
summary = teacher_summary(tdf, teacher_type, min_n=min_n)

tab_top, tab_rank, tab_byq, tab_profile = st.tabs(
    ["🏆 Топ викладачів", "📋 Повний рейтинг", "🎯 За темою питання", "🔍 Профіль викладача"]
)

# ── Leaderboards: top-rated for each category ────────────────────────────────
with tab_top:
    st.caption(f"Найвище оцінені викладачі кожної категорії (n ≥ {min_n}), за згладженою оцінкою.")
    explain_shrunk()

    def leaderboard(role_name):
        top = top_teachers(tdf, role_name, min_n=min_n, k=10)
        st.markdown(f"**🏅 Топ-10 — {role_name.lower()}и**")
        if len(top) == 0:
            st.info("Немає викладачів із достатнім n.")
            return
        d = top[["teacher", "faculty", "n", "shrunk", "avg"]].copy()
        d["shrunk"] = d["shrunk"].round(2)
        d["avg"] = d["avg"].round(2)
        d = d.rename(columns={"teacher": "Викладач", "faculty": "Факультет",
                              "n": "n", "shrunk": "Згладжена", "avg": "Сира"})
        st.dataframe(
            d.style.background_gradient(subset=["Згладжена"], cmap="RdYlGn", vmin=4.0, vmax=5.0)
            .format({"Згладжена": "{:.2f}", "Сира": "{:.2f}"}),
            width="stretch", hide_index=True,
        )

    cL, cP = st.columns(2)
    with cL:
        leaderboard("Лектор")
    with cP:
        leaderboard("Практик")

# ── Full ranking for the selected category ───────────────────────────────────
with tab_rank:
    st.caption(f"Усі **{teacher_type.lower()}и** з n ≥ {min_n}. Клік по заголовку колонки — сортування.")
    if len(summary) == 0:
        st.warning("Немає викладачів із достатнім n за поточних фільтрів.")
    else:
        col_low, col_high = st.columns(2)

        def mini(data, title):
            d = data[["teacher", "faculty", "n", "shrunk", "avg"]].copy()
            d["shrunk"] = d["shrunk"].round(2)
            d["avg"] = d["avg"].round(2)
            d = d.rename(columns={"teacher": "Викладач", "faculty": "Факультет",
                                  "n": "n", "shrunk": "Згладжена", "avg": "Сира"})
            st.markdown(f"**{title}**")
            st.dataframe(
                d.style.background_gradient(subset=["Згладжена"], cmap="RdYlGn", vmin=4.0, vmax=5.0)
                .format({"Згладжена": "{:.2f}", "Сира": "{:.2f}"}),
                width="stretch", hide_index=True,
            )

        with col_low:
            mini(summary.tail(15).sort_values("shrunk"), f"Потребують уваги — {teacher_type.lower()}и")
        with col_high:
            mini(summary.head(15), f"Найвищі оцінки — {teacher_type.lower()}и")

        st.divider()
        full = summary[["teacher", "faculty", "faculties", "n", "courses",
                        "shrunk", "avg", "low_rate", "comment_count"]].copy()
        for c in ("shrunk", "avg"):
            full[c] = full[c].round(2)
        full["low_rate"] = full["low_rate"].round(1)
        full["comment_count"] = full["comment_count"].astype(int)
        full = full.rename(columns={
            "teacher": "Викладач", "faculty": "Осн. факультет", "faculties": "К-сть фак.",
            "n": "Відповідей", "courses": "Курсів", "shrunk": "Згладжена",
            "avg": "Сира середня", "low_rate": "% ≤3", "comment_count": "Корисних коментарів",
        })
        st.dataframe(
            full.style.background_gradient(subset=["Згладжена"], cmap="RdYlGn", vmin=4.0, vmax=5.0)
            .background_gradient(subset=["% ≤3"], cmap="YlOrRd", vmin=0, vmax=30)
            .format({"Згладжена": "{:.2f}", "Сира середня": "{:.2f}", "% ≤3": "{:.1f}%"}),
            width="stretch", height=460, hide_index=True,
        )
        download_csv(full, f"teachers_{teacher_type}.csv", key="dl_teachers")

# ── Ranking by a single question topic ───────────────────────────────────────
with tab_byq:
    st.caption(
        f"Рейтинг **{teacher_type.lower()}ів** за окремою темою питання (а не за загальною "
        f"оцінкою блоку). Оцінка теми згладжена так само, як і загальна (поправка на n ≥ {min_n})."
    )
    q_label_to_col = {QUESTION_LABELS[c]: c for c in qcols}
    sel_qlabel = st.selectbox("Тема питання", list(q_label_to_col.keys()))
    qcol = q_label_to_col[sel_qlabel]

    rk = teacher_question_ranking(tdf, teacher_type, qcol, min_n=min_n)
    if len(rk) == 0:
        st.warning("Немає викладачів із достатнім n за поточних фільтрів.")
    else:
        st.markdown(f"**Тема: «{sel_qlabel}» — {len(rk)} {teacher_type.lower()}ів**")
        c_best, c_worst = st.columns(2)

        def q_table(container, data, title):
            d = data[["teacher", "faculty", "n", "q_shrunk", "q_avg"]].copy()
            d["q_shrunk"] = d["q_shrunk"].round(2)
            d["q_avg"] = d["q_avg"].round(2)
            d = d.rename(columns={"teacher": "Викладач", "faculty": "Факультет", "n": "n",
                                  "q_shrunk": "Згладжена (тема)", "q_avg": "Сира (тема)"})
            with container:
                st.markdown(f"**{title}**")
                st.dataframe(
                    d.style.background_gradient(subset=["Згладжена (тема)"], cmap="RdYlGn", vmin=4.0, vmax=5.0)
                    .format({"Згладжена (тема)": "{:.2f}", "Сира (тема)": "{:.2f}"}),
                    width="stretch", hide_index=True,
                )

        q_table(c_best, rk.head(15), f"🏅 Найкращі за «{sel_qlabel}»")
        q_table(c_worst, rk.tail(15).sort_values("q_shrunk"), f"Потребують уваги за «{sel_qlabel}»")

        full_q = rk[["teacher", "faculty", "n", "q_shrunk", "q_avg", "low_rate"]].copy()
        full_q["q_shrunk"] = full_q["q_shrunk"].round(2)
        full_q["q_avg"] = full_q["q_avg"].round(2)
        full_q["low_rate"] = full_q["low_rate"].round(1)
        full_q = full_q.rename(columns={
            "teacher": "Викладач", "faculty": "Факультет", "n": "Відповідей",
            "q_shrunk": "Згладжена (тема)", "q_avg": "Сира (тема)", "low_rate": "% ≤3 (тема)",
        })
        download_csv(full_q, f"teachers_{teacher_type}_{qcol}.csv", key="dl_byq")

# ── Single-teacher profile ───────────────────────────────────────────────────
with tab_profile:
    if len(summary) == 0:
        st.warning("Немає викладачів із достатнім n за поточних фільтрів.")
    else:
        names = summary["teacher"].tolist()
        sel = st.selectbox(f"Виберіть викладача ({teacher_type.lower()}а)", names)
        row = summary[summary["teacher"] == sel].iloc[0]
        rows = tdf[(tdf["teacher"] == sel) & (tdf["role"] == teacher_type)]

        mc = st.columns(5)
        mc[0].metric("Роль", teacher_type)
        mc[1].metric("Відповідей", int(row["n"]))
        mc[2].metric("Згладжена", f"{row['shrunk']:.2f}")
        mc[3].metric("Оцінок ≤3", f"{row['low_rate']:.1f}%")
        mc[4].metric("Корисних коментарів", int(row["comment_count"]))

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Середні за темами питань")
            q_means = pd.Series({c: row[c] for c in qcols})
            st.plotly_chart(horizontal_bar_questions(q_means, QUESTION_LABELS, title=sel),
                            width="stretch")
        with c2:
            st.subheader("Розподіл оцінок")
            scores = rows[qcols].values.flatten()
            scores = scores[~np.isnan(scores)].astype(int)
            st.plotly_chart(score_distribution_bar(pd.Series(scores), title=f"{sel} — розподіл"),
                            width="stretch")

        st.subheader(f"Курси — {sel}")
        courses_df = rows.groupby("course").agg(
            n=("avg_overall", "count"),
            avg_teacher=(score_col, "mean"),
        ).reset_index().sort_values("avg_teacher", ascending=True)
        courses_df["avg_teacher"] = courses_df["avg_teacher"].round(2)
        courses_df = courses_df.rename(columns={
            "course": "Курс", "n": "Відповідей", "avg_teacher": f"Оцінка ({teacher_type})",
        })
        st.dataframe(
            courses_df.style.background_gradient(subset=[f"Оцінка ({teacher_type})"],
                                                 cmap="RdYlGn", vmin=4.0, vmax=5.0)
            .format({f"Оцінка ({teacher_type})": "{:.2f}"}),
            width="stretch", hide_index=True,
        )

        st.subheader("Що писали студенти")
        render_comments(rows, key_prefix="teacher")
