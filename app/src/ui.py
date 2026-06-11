"""Small Streamlit UI helpers shared across pages."""
import pandas as pd
import streamlit as st

SENTIMENT_ICON = {"Негативний": "🔴", "Нейтральний": "🟡", "Позитивний": "🟢"}


def explain_shrunk():
    """Expander explaining the Bayesian-shrunk ('Згладжена') average in plain language."""
    with st.expander("ℹ️ Що таке «Згладжена оцінка» і чому вона чесніша за звичайну середню"):
        st.markdown(
            """
**Проблема звичайної середньої.** Курс із 4 відповідями, де всі поставили 5,
отримує «ідеальні» **5.00** — хоча це майже напевно випадковість, а не доказ якості.
І навпаки: один незадоволений студент із групи на 3 особи може «потопити» курс.
За такого сильного зміщення оцінок до 5 малі вибірки дають найбільше хибних висновків.

**Що робить згладжена оцінка.** Вона «підтягує» оцінку курсу до загального
середнього по університету доти, доки не набереться достатньо відповідей. Формула:

```
згладжена = (n × середня_курсу + k × середнє_університету) / (n + k)
```

де **n** — кількість відповідей курсу, а **k ≈ 20** — «вага» загального середнього
(приблизно стільки відповідей потрібно, щоб курс почав «говорити сам за себе»).

- **Багато відповідей** (n ≫ k) → згладжена ≈ фактична середня курсу.
- **Мало відповідей** (n ≪ k) → згладжена ≈ середнє по університету (бо даним ще зарано довіряти).

**Навіщо.** Рейтинги стають чеснішими: випадкові кілька оцінок більше не виштовхують
курс ані в топ, ані в антитоп. У таблиці видно обидві колонки — **Згладжена**
(для рейтингу) і **Сира** (фактична середня) — щоб різниця була прозорою.
            """
        )


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
