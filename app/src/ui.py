"""Small Streamlit UI helpers shared across pages."""
import pandas as pd
import streamlit as st

SENTIMENT_ICON = {"Негативний": "🔴", "Нейтральний": "🟡", "Позитивний": "🟢"}


def download_csv(df: pd.DataFrame, filename: str, label: str = "⬇️ Завантажити CSV", key: str = None):
    """Render a download button that exports `df` as UTF-8 CSV (Excel-friendly)."""
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label=label,
        data=csv,
        file_name=filename,
        mime="text/csv",
        key=key,
    )


def render_comments(df_subset: pd.DataFrame, key_prefix: str, max_show: int = 50):
    """Show substantive (non-trivial) comments grouped by sentiment.

    Negatives are surfaced first — they are the actionable feedback. Expects the
    enriched columns from preprocessing: comment_useful, sentiment, avg_overall.
    """
    useful = df_subset[df_subset["comment_useful"]].copy()
    if len(useful) == 0:
        st.info("Немає змістовних коментарів (тривіальні «так / -» відфільтровані).")
        return

    counts = useful["sentiment"].value_counts()
    st.caption(
        f"Змістовних коментарів: **{len(useful)}** "
        f"(🔴 {counts.get('Негативний', 0)} · 🟡 {counts.get('Нейтральний', 0)} · "
        f"🟢 {counts.get('Позитивний', 0)}). Тривіальні відповіді відфільтровано."
    )

    options = ["🔴 Негативні", "🟡 Нейтральні", "🟢 Позитивні"]
    chosen = st.multiselect(
        "Показати тональність", options, default=["🔴 Негативні", "🟡 Нейтральні"],
        key=f"{key_prefix}_sent",
    )
    name_map = {"🔴 Негативні": "Негативний", "🟡 Нейтральні": "Нейтральний", "🟢 Позитивні": "Позитивний"}
    wanted = [name_map[c] for c in chosen]
    filtered = useful[useful["sentiment"].isin(wanted)] if wanted else useful

    # negatives first, then by score ascending
    filtered = filtered.sort_values("avg_overall")
    for _, row in filtered.head(max_show).iterrows():
        icon = SENTIMENT_ICON.get(row["sentiment"], "")
        st.markdown(f"{icon} **{row['avg_overall']:.1f}** — {row['comment']}")
    if len(filtered) > max_show:
        st.caption(f"Показано перші {max_show} із {len(filtered)}.")
