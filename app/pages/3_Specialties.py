import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.loader import load, load_teachers
from src.metrics import (
    SCORE_COLS, QUESTION_LABELS, department_summary, course_summary,
    teacher_summary, top_teachers,
)
from src.charts import horizontal_bar_questions
from src.ui import download_csv, render_comments, explain_shrunk
from src.access import access_control

st.set_page_config(page_title="Спеціальності", page_icon="🏛️", layout="wide")
st.title("🏛️ Аналіз за спеціальностями")
st.caption(
    "Рівень між факультетом і курсом: **Факультет → Спеціальність → Курс**. "
    "Тут можна порівняти спеціальності по всьому університету та подивитися кожну окремо — "
    "її відхилення **і від факультету, і від університету**, курси, викладачів і коментарі."
)

uni = load()                                   # full data — stable university baseline
df_full, role, scope_faculty = access_control(uni)
is_dean = scope_faculty is not None
teachers_all = load_teachers()
if is_dean:
    teachers_all = teachers_all[teachers_all["faculty"] == scope_faculty]

fc1, fc2 = st.columns(2)
with fc1:
    if not is_dean:
        faculties = ["Всі"] + sorted(df_full["faculty"].unique())
        sel_faculty = st.selectbox("Факультет", faculties)
    else:
        sel_faculty = scope_faculty
        st.selectbox("Факультет", [scope_faculty], disabled=True)
with fc2:
    min_n = st.slider("Мін. відповідей на спеціальність", 5, 100, 20, 5)

df = df_full if sel_faculty in ("Всі", scope_faculty) else df_full[df_full["faculty"] == sel_faculty]
teachers_scoped = teachers_all if sel_faculty in ("Всі", scope_faculty) else \
    teachers_all[teachers_all["faculty"] == sel_faculty]

tab_rank, tab_drill = st.tabs(["📋 Рейтинг спеціальностей", "🔍 Профіль спеціальності"])

# ── University-wide department ranking ───────────────────────────────────────
with tab_rank:
    st.subheader(f"Рейтинг спеціальностей (n ≥ {min_n})")
    st.caption("Сортування за **згладженою оцінкою** (поправка на обсяг вибірки). "
               "«Слабке питання» — найнижча з 11 тем якості для спеціальності.")
    explain_shrunk()

    ds = department_summary(df, min_n=min_n)
    if len(ds) == 0:
        st.warning("Жодна спеціальність не має достатньо відповідей. Зменшіть поріг.")
    else:
        disp = ds[["faculty", "specialty", "n", "courses", "shrunk_quality",
                   "avg_quality", "low_rate", "weakest_question", "comment_count"]].copy()
        for c in ("shrunk_quality", "avg_quality"):
            disp[c] = disp[c].round(2)
        disp["low_rate"] = disp["low_rate"].round(1)
        disp["comment_count"] = disp["comment_count"].astype(int)
        disp = disp.rename(columns={
            "faculty": "Факультет", "specialty": "Спеціальність",
            "n": "Відповідей", "courses": "Курсів", "shrunk_quality": "Згладжена",
            "avg_quality": "Сира якість", "low_rate": "% ≤3",
            "weakest_question": "Слабке питання", "comment_count": "Корисних коментарів",
        })
        st.dataframe(
            disp.style.background_gradient(subset=["Згладжена", "Сира якість"],
                                           cmap="RdYlGn", vmin=4.0, vmax=5.0)
            .background_gradient(subset=["% ≤3"], cmap="YlOrRd", vmin=0, vmax=15)
            .format({"Згладжена": "{:.2f}", "Сира якість": "{:.2f}", "% ≤3": "{:.1f}%"}),
            width="stretch", height=520, hide_index=True,
        )
        download_csv(disp, "departments_ranking.csv", key="dl_depts")

# ── Single-department drill with dual comparison ─────────────────────────────
with tab_drill:
    pairs = (df[["faculty", "specialty"]].drop_duplicates()
             .sort_values(["faculty", "specialty"]))
    if len(pairs) == 0:
        st.warning("Немає даних за поточних фільтрів.")
        st.stop()
    labels = {f"{r.specialty}  ·  {r.faculty}": (r.faculty, r.specialty)
              for r in pairs.itertuples()}
    sel = st.selectbox("Виберіть спеціальність", list(labels.keys()))
    sel_fac, sel_spec = labels[sel]

    dept_df = df[(df["faculty"] == sel_fac) & (df["specialty"] == sel_spec)]
    fac_df = uni[uni["faculty"] == sel_fac]
    dept_teachers = teachers_scoped[(teachers_scoped["faculty"] == sel_fac)
                                    & (teachers_scoped["specialty"] == sel_spec)]

    dq = dept_df[SCORE_COLS].mean()
    m = st.columns(5)
    m[0].metric("Відповідей", len(dept_df))
    m[1].metric("Курсів", dept_df["course"].nunique())
    m[2].metric("Якість (без навант.)", f"{dept_df[[c for c in SCORE_COLS if c != 'Q01_workload']].values.mean():.2f}")
    m[3].metric("Оцінок ≤3", f"{(dept_df['avg_overall'] <= 3).mean() * 100:.1f}%")
    m[4].metric("Корисних коментарів", int(dept_df["comment_useful"].sum()))

    # Dual-baseline per-question comparison
    st.subheader("Профіль за темами: спеціальність vs факультет vs університет")
    st.caption("Стовпчики — середня спеціальності по кожному питанню; ромб — середнє по **факультету**, "
               "коло — середнє по **університету**. Так видно, чи спеціальність слабша/сильніша і за свій "
               "факультет, і за загальноуніверситетський рівень.")
    q_dept = dept_df[SCORE_COLS].mean().rename(QUESTION_LABELS)
    q_fac = fac_df[SCORE_COLS].mean().rename(QUESTION_LABELS)
    q_uni = uni[SCORE_COLS].mean().rename(QUESTION_LABELS)
    order = q_dept.sort_values().index.tolist()

    fig = go.Figure()
    fig.add_trace(go.Bar(x=q_dept[order].values, y=order, orientation="h",
                         name="Спеціальність", marker_color="#1f77b4"))
    fig.add_trace(go.Scatter(x=q_fac[order].values, y=order, mode="markers",
                             name="Факультет",
                             marker=dict(color="#ff7f0e", size=10, symbol="diamond")))
    fig.add_trace(go.Scatter(x=q_uni[order].values, y=order, mode="markers",
                             name="Університет",
                             marker=dict(color="black", size=8, symbol="circle")))
    fig.update_layout(xaxis=dict(range=[1, 5.4], title="Середня оцінка"), yaxis_title="",
                      height=440, legend=dict(orientation="h", yanchor="bottom", y=1.02),
                      margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, width="stretch")

    # Courses of the department
    st.subheader(f"Курси спеціальності — {sel_spec}")
    cs = course_summary(dept_df, min_n=5)
    if len(cs):
        cdisp = cs[["course", "n", "shrunk_quality", "low_score_rate", "weakest_question"]].copy()
        cdisp["shrunk_quality"] = cdisp["shrunk_quality"].round(2)
        cdisp["low_score_rate"] = (cdisp["low_score_rate"] * 100).round(1)
        cdisp = cdisp.rename(columns={
            "course": "Курс", "n": "n", "shrunk_quality": "Згладжена",
            "low_score_rate": "% ≤3", "weakest_question": "Слабке питання",
        })
        st.dataframe(
            cdisp.style.background_gradient(subset=["Згладжена"], cmap="RdYlGn", vmin=4.0, vmax=5.0)
            .format({"Згладжена": "{:.2f}", "% ≤3": "{:.1f}%"}),
            width="stretch", hide_index=True, height=min(360, 80 + 34 * len(cdisp)),
        )
    else:
        st.caption("Замало курсів з n ≥ 5 для таблиці.")

    # Top teachers of the department (both categories)
    st.subheader("Викладачі спеціальності")
    st.caption("Найвище оцінені викладачі спеціальності (n ≥ 5 на спеціальності), окремо лектори і практики.")
    tcol1, tcol2 = st.columns(2)

    def dept_leaderboard(container, role_name):
        top = top_teachers(dept_teachers, role_name, min_n=5, k=8)
        with container:
            st.markdown(f"**{role_name}и**")
            if len(top) == 0:
                st.caption("Немає викладачів з n ≥ 5.")
                return
            d = top[["teacher", "n", "shrunk", "avg"]].copy()
            d["shrunk"] = d["shrunk"].round(2)
            d["avg"] = d["avg"].round(2)
            d = d.rename(columns={"teacher": "Викладач", "n": "n",
                                  "shrunk": "Згладжена", "avg": "Сира"})
            st.dataframe(
                d.style.background_gradient(subset=["Згладжена"], cmap="RdYlGn", vmin=4.0, vmax=5.0)
                .format({"Згладжена": "{:.2f}", "Сира": "{:.2f}"}),
                width="stretch", hide_index=True,
            )

    dept_leaderboard(tcol1, "Лектор")
    dept_leaderboard(tcol2, "Практик")

    # Comments of the department
    st.subheader("Що писали студенти")
    render_comments(dept_df, key_prefix="dept")
