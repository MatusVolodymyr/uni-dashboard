import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
from src.loader import load, load_teachers
from src.metrics import (
    SCORE_COLS, Q01_COLS, Q03_COLS, Q05_COLS,
    QUESTION_LABELS, group_question_means, group_counts,
    university_question_means, group_summary, course_summary,
)
from src.charts import (
    stacked_score_bar, heatmap_faculty_question, scatter_risk, horizontal_bar_questions,
)
from src.access import access_control
from src.ui import download_csv
from src.report import build_summary_pdf

st.set_page_config(page_title="Огляд", page_icon="📊", layout="wide")
st.title("📊 Загальний огляд")
st.caption(
    "Оцінки сильно зміщені до 5 (≈80% — п'ятірки), тому сама лише середня оцінка "
    "малоінформативна. Дивіться на **розподіл**, **частку низьких оцінок (≤3)** "
    "та **відхилення від середнього**, а не на абсолютне число."
)

# Full dataset (for stable baselines) and access-scoped dataset
df_all = load()
teachers_all = load_teachers()
df_full, role, scope_faculty = access_control(df_all)
is_dean = scope_faculty is not None
# Hierarchy: Факультет → Спеціальність (Кафедра) → Курс.
# When a dean is scoped to one faculty, drill structure is by Кафедра.
group_col = "specialty" if is_dean else "faculty"
group_name = "Кафедра" if is_dean else "Факультет"
DEPT_LABEL = "Спеціальність"

# ── Page controls ────────────────────────────────────────────────────────────
block_opts = {"Всі блоки": SCORE_COLS, "Дисципліна": Q01_COLS, "Лектор": Q03_COLS, "Практик": Q05_COLS}
fc1, fc2, fc3 = st.columns(3)
with fc1:
    if not is_dean:
        faculties = ["Всі"] + sorted(df_full["faculty"].unique())
        sel_faculty = st.selectbox("Факультет", faculties)
    else:
        sel_faculty = scope_faculty
        st.selectbox("Факультет", [scope_faculty], disabled=True)
scope_df = df_full if sel_faculty in ("Всі", scope_faculty) else df_full[df_full["faculty"] == sel_faculty]
with fc2:
    specs = (["Всі"] + sorted(scope_df["specialty"].unique())) if sel_faculty != "Всі" else ["Всі"]
    sel_spec = st.selectbox(DEPT_LABEL, specs)
with fc3:
    sel_block = st.selectbox("Блок питань", list(block_opts.keys()))

# Apply filters (to both feedback and the canonical teacher table)
df = df_full.copy()
teachers_scoped = teachers_all if not is_dean else teachers_all[teachers_all["faculty"] == scope_faculty]
if sel_faculty != "Всі":
    df = df[df["faculty"] == sel_faculty]
    teachers_scoped = teachers_scoped[teachers_scoped["faculty"] == sel_faculty]
if sel_spec != "Всі":
    df = df[df["specialty"] == sel_spec]
    teachers_scoped = teachers_scoped[teachers_scoped["specialty"] == sel_spec]

active_cols = block_opts[sel_block]

# ── KPI cards ────────────────────────────────────────────────────────────────
all_scores = df[active_cols].values.flatten()
all_scores = all_scores[~pd.isna(all_scores)]
total_scores = len(all_scores)
avg = all_scores.mean() if total_scores else 0
low_rate = (all_scores <= 3).sum() / total_scores * 100 if total_scores else 0
top_rate = (all_scores == 5).sum() / total_scores * 100 if total_scores else 0
comment_rate = df["has_comment"].mean() * 100

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Відповідей", f"{len(df):,}")
k2.metric("Курсів", f"{df['course'].nunique():,}")
if is_dean:
    k3.metric("Кафедр", f"{df['specialty'].nunique():,}")
else:
    k3.metric("Факультетів", f"{df['faculty'].nunique():,}")
k4.metric("Середня оцінка", f"{avg:.2f}")
k5.metric("Оцінок ≤ 3", f"{low_rate:.1f}%")
k6.metric("З коментарем", f"{comment_rate:.1f}%")

st.divider()

# ── General data by faculty / department ─────────────────────────────────────
st.subheader(f"Загальні дані за {'спеціальностями' if is_dean else 'факультетами'}")
st.caption(
    "Зведена таблиця. **Лектори/Практики** показані у двох варіантах середньої: "
    "**за відповідями** (кожна відповідь студента важить однаково) та **за викладачами** "
    "(кожен викладач важить однаково, незалежно від розміру групи). «Якість» — середня по "
    "11 питаннях якості (без навантаження)."
)

gs = group_summary(df, group_col=group_col, teachers=teachers_scoped)
cols_order = [group_col, "n"]
if group_col == "faculty":
    cols_order += ["departments"]
cols_order += ["courses", "avg_quality",
               "lect_resp", "lect_tchr", "pract_resp", "pract_tchr",
               "low_rate", "comment_rate"]
gs_disp = gs[cols_order].copy()
rename = {
    group_col: (DEPT_LABEL if is_dean else group_name),
    "n": "Відповідей", "departments": "Кафедр", "courses": "Курсів",
    "avg_quality": "Якість",
    "lect_resp": "Лектори (за відп.)", "lect_tchr": "Лектори (за викл.)",
    "pract_resp": "Практики (за відп.)", "pract_tchr": "Практики (за викл.)",
    "low_rate": "% ≤3", "comment_rate": "% коментарів",
}
gs_disp = gs_disp.rename(columns=rename)
num_fmt = {c: "{:.2f}" for c in ["Якість", "Лектори (за відп.)", "Лектори (за викл.)",
                                  "Практики (за відп.)", "Практики (за викл.)"]}
num_fmt.update({"% ≤3": "{:.1f}%", "% коментарів": "{:.1f}%"})
st.dataframe(
    gs_disp.style.background_gradient(subset=["Якість"], cmap="RdYlGn", vmin=4.0, vmax=5.0)
    .background_gradient(subset=["% ≤3"], cmap="YlOrRd", vmin=0, vmax=15)
    .format(num_fmt),
    width="stretch",
    hide_index=True,
    height=min(560, 80 + 36 * len(gs_disp)),
)
exp1, exp2, _ = st.columns([1.1, 1.1, 4])
with exp1:
    download_csv(gs_disp, f"general_by_{group_col}.csv", label="⬇️ Дані (CSV)", key="dl_general")
with exp2:
    scope_label = f"Факультет: {scope_faculty}" if is_dean else "Університет (усі факультети)"
    if st.button("📄 Сформувати PDF-звіт", help="Зведений звіт із графіками та таблицями для друку / розсилки."):
        with st.spinner("Формуємо звіт…"):
            report_teachers = teachers_all if not is_dean else teachers_all[teachers_all["faculty"] == scope_faculty]
            uni_means = df_all[SCORE_COLS].mean()
            pdf_bytes = build_summary_pdf(df_full, report_teachers, scope_label,
                                          group_col, group_name, role, uni_means)
        st.session_state["overview_pdf"] = pdf_bytes
if st.session_state.get("overview_pdf"):
    st.download_button(
        "⬇️ Завантажити PDF-звіт",
        data=st.session_state["overview_pdf"],
        file_name=f"zvedenyi_zvit_{'kafedry' if is_dean else 'universytet'}.pdf",
        mime="application/pdf",
        key="dl_pdf",
    )

st.divider()

# ── Risk scatter: all courses at a glance ────────────────────────────────────
st.subheader("Огляд ризику: курси")
st.caption(
    "Кожна точка — курс. По горизонталі — кількість відповідей, по вертикалі — частка "
    "низьких оцінок (≤3); розмір точки теж відображає обсяг. **Найбільшої уваги "
    "потребують курси праворуч угорі**: багато відповідей і водночас багато негативу "
    "(це вже не випадковість). Точки внизу — благополучні курси; ліворуч — курси з малою "
    f"вибіркою. Колір — {group_name.lower()}."
)
scatter_min = st.slider("Мін. відповідей на курс (для графіка)", 5, 50, 10, 5, key="scatter_min")
scat = course_summary(df, min_n=scatter_min)
if len(scat):
    st.plotly_chart(
        scatter_risk(scat, color_col=group_col, color_label=group_name),
        width="stretch",
    )
else:
    st.info("Недостатньо курсів за поточного порогу — зменшіть мінімум відповідей.")

st.divider()

# ── Score distribution ───────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Розподіл оцінок")
    st.caption("Скільки відсотків усіх відповідей припало на кожну оцінку 1–5. "
               "Дозволяє побачити, чи висока середня — це стабільні 5, чи суміш 5 і кількох дуже низьких.")
    score_series = df[active_cols].melt(value_name="score")["score"].dropna().astype(int)
    dist = score_series.value_counts().reindex(range(1, 6), fill_value=0)
    total = dist.sum()

    import plotly.graph_objects as go
    fig_dist = go.Figure(go.Bar(
        x=[str(i) for i in range(1, 6)],
        y=(dist / total * 100).values,
        marker_color=["#d62728", "#ff7f0e", "#bcbd22", "#2ca02c", "#1f77b4"],
        text=[f"{v:.1f}%" for v in (dist / total * 100).values],
        textposition="outside",
    ))
    fig_dist.update_layout(
        xaxis_title="Оцінка",
        yaxis_title="% відповідей",
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_dist, width='stretch')

with col_right:
    st.subheader("Розподіл за блоками питань")
    st.caption("Порівняння трьох блоків опитування: оцінка дисципліни загалом, "
               "роботи лектора та роботи практика.")
    block_avgs = {
        "Дисципліна": df[Q01_COLS].values.flatten(),
        "Лектор": df[Q03_COLS].values.flatten(),
        "Практик": df[Q05_COLS].values.flatten(),
    }

    import plotly.express as px
    import numpy as np
    block_rows = []
    for block, vals in block_avgs.items():
        vals = vals[~np.isnan(vals)].astype(int)
        for score in range(1, 6):
            pct = (vals == score).sum() / len(vals) * 100 if len(vals) else 0
            block_rows.append({"Блок": block, "Оцінка": str(score), "Відсоток": pct})

    block_df = pd.DataFrame(block_rows)
    score_color_map = {
        "1": "#d62728", "2": "#ff7f0e", "3": "#bcbd22", "4": "#2ca02c", "5": "#1f77b4"
    }
    fig_blocks = px.bar(
        block_df, x="Блок", y="Відсоток", color="Оцінка",
        barmode="stack",
        color_discrete_map=score_color_map,
        height=320,
    )
    fig_blocks.update_layout(
        xaxis_title="",
        yaxis_title="% відповідей",
        legend_title="Оцінка",
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_blocks, width='stretch')

st.divider()

# ── Average per question topic ───────────────────────────────────────────────
st.subheader("Середні оцінки за темами питань")
st.caption(
    "Середня оцінка по кожному з 12 питань (за поточних фільтрів), від найслабшого до "
    "найсильнішого. Показує, які саме аспекти навчання просідають по всій вибірці — "
    "напр. «Навантаження» та «Матеріали Elearn» зазвичай нижчі за «Етику лектора»."
)
q_means = df[SCORE_COLS].mean()
st.plotly_chart(
    horizontal_bar_questions(q_means, QUESTION_LABELS, title=""),
    width="stretch",
)

st.divider()

# ── Heatmap (faculty or department) × question ───────────────────────────────
st.subheader(f"Heatmap: {group_name.lower()} × питання")

hcol1, hcol2 = st.columns([3, 1])
with hcol2:
    heat_mode = st.radio(
        "Режим кольору",
        ["Відхилення від середнього", "Абсолютна оцінка"],
        help="Через зміщення оцінок до 5 абсолютна шкала майже всюди «зелена». "
             "Режим відхилення центрує колір на середньому по університету для кожного "
             "питання, тож проблемні місця (синє = нижче, червоне = вище) одразу видно.",
    )
mode = "deviation" if heat_mode.startswith("Відхилення") else "absolute"

row_word = group_name.lower() + ("и" if not is_dean else "")  # "факультети" / "кафедра"
if mode == "deviation":
    st.caption(
        f"Рядки — {row_word} (n = кількість відповідей), колонки — 12 питань. "
        f"Число й колір у клітинці — **відхилення середньої оцінки за цим питанням "
        "від середньої по всьому університету** (синій / «−» = нижче, червоний / «+» = вище). "
        "Фактична середня — у підказці при наведенні."
    )
    st.info(
        "ℹ️ **Як рахується база порівняння.** Еталон для кожного питання — це "
        "**середня по всіх відповідях університету по конкретному питанню** (кожна відповідь важить однаково; "
        "це не середнє від середніх, тож великі підрозділи не перекошують "
        "базу). Еталон береться з **повного набору даних** і **не змінюється** від обраних "
        "фільтрів — тому, навіть звузивши вибірку, ви бачите відхилення саме від "
        "загальноуніверситетського рівня. ⚠️ Клітинка з малим n "
        "ненадійна: відхилення там може бути просто шумом."
    )
else:
    st.caption(
        f"Рядки — {row_word} (n = кількість відповідей), колонки — 12 питань. "
        "Колір і число — фактична середня оцінка 1–5. "
        "Малий n робить клітинку ненадійною — звертайте увагу на кількість відповідей."
    )

pivot = group_question_means(df, group_col)
# keep only selected block questions
block_label_cols = [QUESTION_LABELS[c] for c in active_cols if c in QUESTION_LABELS]
pivot = pivot[[c for c in block_label_cols if c in pivot.columns]]
pivot = pivot.sort_values(pivot.columns[0])

counts = group_counts(df, group_col)
# Baseline = response-weighted university average from the FULL (unscoped) dataset
baseline = university_question_means(df_all)
st.plotly_chart(
    heatmap_faculty_question(pivot, counts=counts, mode=mode, baseline=baseline, row_label=group_name),
    width='stretch',
)