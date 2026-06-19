import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.loader import load
from src.metrics import SCORE_COLS, Q01_COLS, Q03_COLS, Q05_COLS, QUESTION_LABELS
from src.ui import download_csv
from src.access import access_control

st.set_page_config(page_title="Факультети", page_icon="🏫", layout="wide")

df_full = load()
df_full, role, scope_faculty = access_control(df_full)
is_dean = scope_faculty is not None

# This page compares the units one level below the viewer's scope:
#   rector → faculties (baseline = university)
#   dean   → departments within their faculty (baseline = the faculty)
group_col = "specialty" if is_dean else "faculty"
unit = "Спеціальність" if is_dean else "Факультет"
unit_gen = "спеціальності" if is_dean else "факультету"   # "n відповідей <unit_gen>"
unit_loc = "спеціальностях" if is_dean else "факультетах"  # "по <unit_loc>"
unit_acc = "спеціальність" if is_dean else "факультет"     # "виберіть <unit_acc>"
baseline_name = "факультету" if is_dean else "університету"
child_col = "course" if is_dean else "specialty"
child_title = "Курси" if is_dean else "Спеціальності"
child_label = "Курс" if is_dean else "Спеціальність"

st.title(f"🏫 Аналіз за {'спеціальностями' if is_dean else 'факультетами'}")
st.caption(
    f"Порівняння {unit_loc}" + (f" факультету «{scope_faculty}»" if is_dean else " університету") +
    ". Завжди дивіться на кількість відповідей (n) поруч із середньою: "
    f"{unit.lower()} із малим n легко дає викривлену картину."
)

block_opts = {"Всі блоки": SCORE_COLS, "Дисципліна": Q01_COLS, "Лектор": Q03_COLS, "Практик": Q05_COLS}
fc1, fc2 = st.columns(2)
with fc1:
    sel_block = st.selectbox("Блок питань", list(block_opts.keys()))
with fc2:
    min_n = st.slider(f"Мін. відповідей ({unit.lower()})", 5, 200, 20, 5)

active_cols = block_opts[sel_block]

# ── Aggregation per unit ──────────────────────────────────────────────────────
agg = df_full.groupby(group_col).agg(
    n=("avg_overall", "count"),
    avg=("avg_overall", "mean"),
    comment_rate=("comment_useful", "mean"),
).reset_index()
agg = agg[agg["n"] >= min_n].sort_values("avg", ascending=True)
agg["label"] = agg.apply(lambda r: f"{r[group_col]}  (n={int(r['n']):,})", axis=1)

if len(agg) == 0:
    st.warning(f"Жодна {unit.lower()} не має ≥ {min_n} відповідей. Зменшіть поріг у фільтрах.")
    st.stop()

# height scales with number of bars; min keeps a single bar from looking lost
bar_h = max(260, len(agg) * 34)

# ── Average + volume ──────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader(f"Середня оцінка по {unit_loc}")
    st.caption("Колір: червоний < 4.0, помаранчевий < 4.4, зелений ≥ 4.4. "
               "n у підписі — кількість відповідей.")
    fig_avg = go.Figure(go.Bar(
        x=agg["avg"], y=agg["label"], orientation="h",
        marker_color=["#d62728" if v < 4.0 else "#ff7f0e" if v < 4.4 else "#2ca02c" for v in agg["avg"]],
        text=[f"{v:.2f}" for v in agg["avg"]], textposition="outside",
    ))
    fig_avg.update_layout(
        xaxis=dict(range=[1, 5.4], title="Середня оцінка"), yaxis_title="",
        height=bar_h, margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_avg, width="stretch")

with col2:
    st.subheader("Відповіді та коментарі")
    st.caption("Загальна кількість відповідей і скільки з них містять змістовний коментар.")
    fig_n = go.Figure()
    fig_n.add_trace(go.Bar(
        x=agg["n"], y=agg["label"], orientation="h",
        name="Відповідей", marker_color="#1f77b4",
    ))
    fig_n.add_trace(go.Bar(
        x=(agg["comment_rate"] * agg["n"]).round().astype(int), y=agg["label"], orientation="h",
        name="Зі змістовним коментарем", marker_color="#ff7f0e",
    ))
    fig_n.update_layout(
        barmode="overlay", xaxis_title="Кількість", yaxis_title="",
        height=bar_h, legend=dict(orientation="h", yanchor="bottom", y=1.0),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_n, width="stretch")

st.divider()

# ── Stacked score distribution ────────────────────────────────────────────────
st.subheader(f"Розподіл оцінок по {unit_loc}")
st.caption(f"Кожен стовпчик — 100% відповідей {unit_gen}, розкладені за оцінками 1–5. "
           "Червоні сегменти (1–2) показують реальну частку незадоволених — те, що ховається за високою середньою.")
rows = []
for u, lbl in zip(agg[group_col], agg["label"]):
    sub = df_full[df_full[group_col] == u][active_cols].values.flatten()
    sub = sub[~pd.isna(sub)].astype(int)
    total = len(sub)
    for sc in range(1, 6):
        rows.append({unit: lbl, "Оцінка": str(sc), "Відсоток": (sub == sc).sum() / total * 100 if total else 0})
stack_df = pd.DataFrame(rows)
color_map = {"1": "#d62728", "2": "#ff7f0e", "3": "#bcbd22", "4": "#2ca02c", "5": "#1f77b4"}
fig_stack = px.bar(
    stack_df, x="Відсоток", y=unit, color="Оцінка",
    barmode="stack", orientation="h", color_discrete_map=color_map, height=bar_h,
    category_orders={unit: agg["label"].tolist()},
)
fig_stack.update_layout(xaxis_title="% відповідей", yaxis_title="", legend_title="Оцінка",
                        margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig_stack, width="stretch")

st.divider()

# ── Drill-down into one unit ──────────────────────────────────────────────────
st.subheader(f"Деталі — {unit.lower()}")
sel_unit = st.selectbox(f"Виберіть {unit_acc}", sorted(df_full[group_col].unique()))
df_unit = df_full[df_full[group_col] == sel_unit]

q_unit = df_unit[active_cols].mean().rename(QUESTION_LABELS)
q_base = df_full[active_cols].mean().rename(QUESTION_LABELS)
cmp_df = pd.DataFrame({
    "Питання": q_unit.index, "Одиниця": q_unit.values, "Базис": q_base.values,
}).sort_values("Одиниця")

fig_cmp = go.Figure()
fig_cmp.add_trace(go.Bar(x=cmp_df["Одиниця"], y=cmp_df["Питання"], orientation="h",
                        name=sel_unit, marker_color="#1f77b4"))
fig_cmp.add_trace(go.Scatter(x=cmp_df["Базис"], y=cmp_df["Питання"], mode="markers",
                            name=f"Середнє по {baseline_name}",
                            marker=dict(color="white", size=9, symbol="diamond",
                                        line=dict(color="black", width=1))))
fig_cmp.update_layout(
    title=f"Порівняння із середнім по {baseline_name}: {sel_unit}",
    xaxis=dict(range=[1, 5.4], title="Середня оцінка"), yaxis_title="",
    height=420, legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(l=10, r=10, t=60, b=10),
)
st.plotly_chart(fig_cmp, width="stretch")

# Child breakdown table
st.subheader(f"{child_title} — {sel_unit}")
child_agg = df_unit.groupby(child_col).agg(
    n=("avg_overall", "count"),
    avg=("avg_overall", "mean"),
).reset_index().sort_values("avg", ascending=True)
child_disp = child_agg.rename(columns={child_col: child_label, "n": "Відповідей", "avg": "Середня оцінка"})
st.dataframe(
    child_disp.style.background_gradient(subset=["Середня оцінка"], cmap="RdYlGn", vmin=4.0, vmax=5.0)
    .format({"Середня оцінка": "{:.2f}"}),
    width="stretch", hide_index=True, height=min(500, 80 + 34 * len(child_disp)),
)
download_csv(child_disp, f"{child_col}_{sel_unit}.csv", key="dl_child")
