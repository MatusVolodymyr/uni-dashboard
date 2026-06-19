import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.loader import load
from src.metrics import THEME_LABELS
from src.ui import download_csv
from src.access import access_control

st.set_page_config(page_title="Коментарі", page_icon="💬", layout="wide")
st.title("💬 Аналіз коментарів")
st.caption(
    "Текстовий фідбек згруповано за **темами** (за ключовими словами) та **тональністю** "
    "(за середньою оцінкою відповіді). Тривіальні відповіді («так», «-», «все добре») "
    "відфільтровано — вони не несуть змісту. Найцінніше — негативні коментарі: вони пояснюють, "
    "*чому* виникають низькі оцінки."
)

df_full = load()
df_full, role, scope_faculty = access_control(df_full)
theme_cols = list(THEME_LABELS.keys())

# ── Page controls ─────────────────────────────────────────────────────────────
r1c1, r1c2, r1c3 = st.columns(3)
with r1c1:
    faculties = ["Всі"] + sorted(df_full["faculty"].unique())
    sel_faculty = st.selectbox("Факультет", faculties)
with r1c2:
    if sel_faculty != "Всі":
        courses = ["Всі"] + sorted(df_full[df_full["faculty"] == sel_faculty]["course"].unique())
    else:
        courses = ["Всі"]
    sel_course = st.selectbox("Курс", courses)
with r1c3:
    search = st.text_input("Пошук", placeholder="ключові слова...")

r2c1, r2c2, r2c3 = st.columns([2, 2, 1])
with r2c1:
    sentiments = st.multiselect(
        "Тональність", ["Негативний", "Нейтральний", "Позитивний"],
        default=["Негативний", "Нейтральний", "Позитивний"],
    )
with r2c2:
    sel_themes = st.multiselect(
        "Теми", [THEME_LABELS[c] for c in theme_cols],
        help="Коментар може належати кільком темам.",
    )
with r2c3:
    include_trivial = st.checkbox("Включати тривіальні", value=False,
                                  help="Показати також короткі беззмістовні відповіді.")

# ── Base filter ───────────────────────────────────────────────────────────────
base = df_full[df_full["has_comment"]].copy()
if not include_trivial:
    base = base[base["comment_useful"]]

df = base.copy()
if sel_faculty != "Всі":
    df = df[df["faculty"] == sel_faculty]
if sel_course != "Всі":
    df = df[df["course"] == sel_course]
if sentiments:
    df = df[df["sentiment"].isin(sentiments)]

# theme filter (OR across selected themes)
if sel_themes:
    label_to_col = {v: k for k, v in THEME_LABELS.items()}
    chosen_cols = [label_to_col[t] for t in sel_themes]
    mask = df[chosen_cols].any(axis=1)
    df = df[mask]

if search.strip():
    df = df[df["comment"].str.contains(search.strip(), case=False, na=False)]

# ── KPIs ──────────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Знайдено", f"{len(df):,}")
c2.metric("Змістовних усього", f"{int(base['comment_useful'].sum() if include_trivial else len(base)):,}")
if len(df):
    c3.metric("Середня оцінка", f"{df['avg_overall'].mean():.2f}")
    c4.metric("Курсів", f"{df['course'].nunique()}")

st.divider()

# ── Theme frequency by sentiment ──────────────────────────────────────────────
st.subheader("Теми коментарів за тональністю")
st.caption("Скільки коментарів торкається кожної теми, з розбивкою за тональністю. "
           "Червоні сегменти показують теми, що найчастіше згадуються у негативному контексті.")

theme_scope = base if sel_faculty == "Всі" else base[base["faculty"] == sel_faculty]
rows = []
for col in theme_cols:
    sub = theme_scope[theme_scope[col]]
    for sent in ["Негативний", "Нейтральний", "Позитивний"]:
        rows.append({
            "Тема": THEME_LABELS[col],
            "Тональність": sent,
            "Кількість": int((sub["sentiment"] == sent).sum()),
        })
theme_df = pd.DataFrame(rows)
totals = theme_df.groupby("Тема")["Кількість"].sum().sort_values()
theme_order = totals.index.tolist()

sent_colors = {"Негативний": "#d62728", "Нейтральний": "#bcbd22", "Позитивний": "#2ca02c"}
fig = go.Figure()
for sent in ["Негативний", "Нейтральний", "Позитивний"]:
    sub = theme_df[theme_df["Тональність"] == sent].set_index("Тема").reindex(theme_order)
    fig.add_trace(go.Bar(
        y=theme_order, x=sub["Кількість"], orientation="h",
        name=sent, marker_color=sent_colors[sent],
    ))
fig.update_layout(
    barmode="stack", height=380, xaxis_title="Кількість коментарів",
    legend_title="Тональність", margin=dict(l=10, r=10, t=10, b=10),
)
st.plotly_chart(fig, width='stretch')

st.divider()

# ── Comment table ─────────────────────────────────────────────────────────────
st.subheader("Коментарі")
SENTIMENT_ICON = {"Негативний": "🔴", "Нейтральний": "🟡", "Позитивний": "🟢"}

if len(df) == 0:
    st.info("Немає коментарів за заданими фільтрами.")
else:
    df = df.sort_values("avg_overall")  # negatives first
    page_size = st.selectbox("Коментарів на сторінці", [25, 50, 100], index=0)
    total_pages = max(1, (len(df) - 1) // page_size + 1)
    page = st.number_input("Сторінка", min_value=1, max_value=total_pages, value=1, step=1)
    page_df = df.iloc[(page - 1) * page_size: page * page_size]

    st.caption(f"Сторінка {page} з {total_pages} ({len(df):,} коментарів). "
               f"Сортовано від найнижчих оцінок.")

    # Readable list — full comment text always visible, with context.
    for _, row in page_df.iterrows():
        icon = SENTIMENT_ICON.get(row["sentiment"], "")
        teacher = row["lecturer"] or row["practitioner"] or "—"
        st.markdown(
            f"{icon} **{row['avg_overall']:.1f}** · _{row['course']}_ · "
            f"{row['faculty']} · {teacher}"
        )
        st.markdown(f"> {row['comment']}")
        st.divider()

    export_cols = ["faculty", "specialty", "course", "lecturer", "practitioner",
                   "sentiment", "avg_overall", "comment"]
    download_csv(df[export_cols], "comments.csv", key="dl_comments")
