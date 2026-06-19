"""Multi-page executive-summary PDF report.

Rendered with matplotlib (no browser/kaleido needed). matplotlib's default font
DejaVu Sans covers Ukrainian Cyrillic, so text renders correctly everywhere.

Pages: 1 Огляд · 2 Підрозділи + heatmap · 3 Курси ризику · 4 Викладачі ·
       5 Коментарі · 6 Якість даних.
"""
import io
import textwrap
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from .metrics import (
    SCORE_COLS, QUALITY_COLS, Q01_COLS, Q03_COLS, Q05_COLS, QUESTION_LABELS,
    THEME_LABELS, group_summary, course_summary, group_question_means, top_teachers,
)

SCORE_COLORS = ["#d62728", "#ff7f0e", "#bcbd22", "#2ca02c", "#1f77b4"]
SENT_COLORS = {"Негативний": "#d62728", "Нейтральний": "#bcbd22", "Позитивний": "#2ca02c"}
A4 = (8.27, 11.69)


# ── small helpers ────────────────────────────────────────────────────────────
def _title(fig, title, scope_label, role_label=None):
    fig.text(0.05, 0.965, title, fontsize=17, fontweight="bold")
    fig.text(0.05, 0.945, scope_label, fontsize=11, color="#555")
    if role_label:
        fig.text(0.05, 0.930, f"Роль: {role_label}    ·    Сформовано: {date.today():%d.%m.%Y}",
                 fontsize=8, color="#888")


def _table(ax, df, title=None, fontsize=7, first_col_w=0.34):
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=11, loc="left", pad=8, fontweight="bold")
    if len(df) == 0:
        ax.text(0.5, 0.9, "немає даних", ha="center", fontsize=8, color="#888")
        return
    ncols = len(df.columns)
    rest = (1 - first_col_w) / max(ncols - 1, 1)
    col_widths = [first_col_w] + [rest] * (ncols - 1)
    tbl = ax.table(cellText=df.values, colLabels=list(df.columns),
                   loc="upper center", cellLoc="center", colWidths=col_widths)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    tbl.scale(1, 1.32)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#e0e0e0")
        if c == 0:
            cell._loc = "left"
            cell.PAD = 0.03
        if r == 0:
            cell.set_facecolor("#1f77b4")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f5f7fa")
    return tbl


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _score_distribution_ax(ax, df):
    vals = df[SCORE_COLS].values.flatten()
    vals = vals[~np.isnan(vals)].astype(int)
    total = len(vals)
    pcts = [(vals == s).sum() / total * 100 if total else 0 for s in range(1, 6)]
    bars = ax.bar([str(s) for s in range(1, 6)], pcts, color=SCORE_COLORS)
    for b, p in zip(bars, pcts):
        ax.text(b.get_x() + b.get_width() / 2, p + 1, f"{p:.1f}%", ha="center", va="bottom", fontsize=7)
    ax.set_title("Розподіл оцінок (1–5)", fontsize=10, loc="left", fontweight="bold")
    ax.set_ylabel("% відповідей", fontsize=8)
    ax.set_ylim(0, max(pcts) * 1.18 if pcts else 1)
    ax.tick_params(labelsize=8)
    _despine(ax)


def _block_avg_ax(ax, df):
    blocks = [("Дисципліна", Q01_COLS), ("Лектор", Q03_COLS), ("Практик", Q05_COLS)]
    vals = [df[c].values.flatten() for _, c in blocks]
    means = [np.nanmean(v) for v in vals]
    bars = ax.bar([b for b, _ in blocks], means, color=["#1f77b4", "#2ca02c", "#ff7f0e"])
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.02, f"{m:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_title("Середня за блоками", fontsize=10, loc="left", fontweight="bold")
    ax.set_ylim(3.8, 5.05)
    ax.tick_params(labelsize=8)
    _despine(ax)


def _question_means_ax(ax, df):
    means = df[SCORE_COLS].mean().rename(QUESTION_LABELS).sort_values()
    colors = ["#d62728" if v < 4.4 else "#ff7f0e" if v < 4.6 else "#2ca02c" for v in means.values]
    ax.barh(range(len(means)), means.values, color=colors)
    ax.set_yticks(range(len(means)))
    ax.set_yticklabels(means.index, fontsize=7)
    for i, v in enumerate(means.values):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=6.5)
    ax.set_xlim(3.8, 5.1)
    ax.set_title("Середні за темами питань (від найслабшого)", fontsize=10, loc="left", fontweight="bold")
    ax.tick_params(labelsize=7)
    _despine(ax)


def _heatmap_ax(ax, pivot, baseline, max_rows=18):
    if len(pivot) > max_rows:
        pivot = pivot.sort_values(pivot.columns[0]).head(max_rows)
    dev = pivot.values - baseline.reindex(pivot.columns).values
    im = ax.imshow(dev, cmap="RdBu", vmin=-0.5, vmax=0.5, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=40, ha="right", fontsize=5)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(x)[:34] for x in pivot.index], fontsize=5)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:.2f}", ha="center", va="center", fontsize=4)
    ax.set_title("Відхилення від середнього по університету (синій нижче / червоний вище)",
                 fontsize=9, loc="left", fontweight="bold", pad=6)
    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.01)


def _scatter_ax(ax, courses):
    if len(courses) == 0:
        ax.text(0.5, 0.5, "Недостатньо курсів", ha="center", va="center")
        ax.axis("off")
        return
    att = courses["avg_quality"] < 4.4
    ax.scatter(courses.loc[~att, "n"], courses.loc[~att, "low_score_rate"] * 100,
               s=np.clip(courses.loc[~att, "n"], 10, 400), c="#1f77b4", alpha=0.5,
               edgecolors="none", label="Нормально")
    ax.scatter(courses.loc[att, "n"], courses.loc[att, "low_score_rate"] * 100,
               s=np.clip(courses.loc[att, "n"], 10, 400), c="#d62728", alpha=0.6,
               edgecolors="none", label="Потребує уваги (якість < 4.4)")
    for _, r in courses.sort_values("low_score_rate", ascending=False).head(5).iterrows():
        ax.annotate(str(r["course"])[:32], (r["n"], r["low_score_rate"] * 100),
                    fontsize=6, xytext=(4, 4), textcoords="offset points", color="#333")
    ax.set_xlabel("Кількість відповідей (n)", fontsize=8)
    ax.set_ylabel("Частка низьких оцінок ≤3, %", fontsize=8)
    ax.set_title("Огляд ризику: курси (праворуч угорі — найважливіші)",
                 fontsize=10, loc="left", fontweight="bold")
    ax.legend(fontsize=7, loc="upper right")
    ax.tick_params(labelsize=8)
    _despine(ax)


def _theme_sentiment_ax(ax, df):
    useful = df[df["comment_useful"]]
    theme_cols = list(THEME_LABELS.keys())
    rows = []
    for col in theme_cols:
        sub = useful[useful[col]]
        rows.append({"theme": THEME_LABELS[col],
                     **{s: int((sub["sentiment"] == s).sum())
                        for s in ("Негативний", "Нейтральний", "Позитивний")}})
    t = pd.DataFrame(rows)
    t["total"] = t[["Негативний", "Нейтральний", "Позитивний"]].sum(axis=1)
    t = t.sort_values("total")
    y = range(len(t))
    left = np.zeros(len(t))
    for s in ("Негативний", "Нейтральний", "Позитивний"):
        ax.barh(y, t[s].values, left=left, color=SENT_COLORS[s], label=s)
        left += t[s].values
    ax.set_yticks(list(y))
    ax.set_yticklabels(t["theme"], fontsize=7)
    ax.set_xlabel("Кількість коментарів", fontsize=8)
    ax.set_title("Теми коментарів за тональністю", fontsize=10, loc="left", fontweight="bold")
    ax.legend(fontsize=6.5, loc="lower right")
    ax.tick_params(labelsize=7)
    _despine(ax)


# ── main builder ─────────────────────────────────────────────────────────────
def build_summary_pdf(df, teachers, scope_label, group_col, group_name,
                      role_label, uni_means) -> bytes:
    df = df.copy()
    df["avg_quality"] = df[QUALITY_COLS].mean(axis=1)
    is_dept = group_col == "specialty"
    units_word = "спеціальностями" if is_dept else "факультетами"

    total = len(df)
    all_scores = df[SCORE_COLS].values.flatten()
    all_scores = all_scores[~np.isnan(all_scores)]
    avg = all_scores.mean()
    low_rate = (all_scores <= 3).mean() * 100
    comment_rate = df["comment_useful"].mean() * 100

    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        # ── Page 1: Огляд ─────────────────────────────────────────────────────
        fig = plt.figure(figsize=A4)
        _title(fig, "Зведений звіт — опитування студентів", scope_label, role_label)
        kpis = [
            ("Відповідей", f"{total:,}"),
            ("Курсів", f"{df['course'].nunique():,}"),
            (group_name + ("и" if group_name == "Факультет" else ""), f"{df[group_col].nunique():,}"),
            ("Середня", f"{avg:.2f}"),
            ("Оцінок ≤3", f"{low_rate:.1f}%"),
            ("З коментарем", f"{comment_rate:.1f}%"),
        ]
        for i, (lbl, val) in enumerate(kpis):
            x = 0.05 + i * 0.158
            fig.text(x, 0.895, val, fontsize=15, fontweight="bold")
            fig.text(x, 0.878, lbl, fontsize=7.5, color="#666")

        _score_distribution_ax(fig.add_axes([0.07, 0.60, 0.40, 0.20]), df)
        _block_avg_ax(fig.add_axes([0.57, 0.60, 0.37, 0.20]), df)
        _question_means_ax(fig.add_axes([0.30, 0.10, 0.62, 0.40]), df)
        pdf.savefig(fig); plt.close(fig)

        # ── Page 2: Підрозділи + heatmap ──────────────────────────────────────
        fig = plt.figure(figsize=A4)
        _title(fig, f"Дані за {units_word}", scope_label)
        gs = group_summary(df, group_col=group_col)
        cols = [group_col, "n", "courses", "avg_quality", "lect",
                "pract", "low_rate", "comment_rate"]
        t = gs[cols].head(24).copy()
        t[group_col] = t[group_col].astype(str).str.slice(0, 32)
        for c in ("avg_quality", "lect", "pract"):
            t[c] = pd.to_numeric(t[c], errors="coerce").map(lambda v: f"{v:.2f}" if pd.notna(v) else "—")
        t["low_rate"] = t["low_rate"].map(lambda v: f"{v:.1f}%")
        t["comment_rate"] = t["comment_rate"].map(lambda v: f"{v:.1f}%")
        t = t.rename(columns={
            group_col: group_name, "n": "Відп.", "courses": "Курс.", "avg_quality": "Якість",
            "lect": "Лектори", "pract": "Практики",
            "low_rate": "% ≤3", "comment_rate": "% ком.",
        })
        _table(fig.add_axes([0.03, 0.50, 0.94, 0.40]), t,
               title=f"Загальні дані за {units_word}", fontsize=6.5, first_col_w=0.30)
        pivot = group_question_means(df, group_col)
        _heatmap_ax(fig.add_axes([0.18, 0.06, 0.72, 0.36]), pivot, uni_means.rename(QUESTION_LABELS))
        pdf.savefig(fig); plt.close(fig)

        # ── Page 3: Курси ризику ──────────────────────────────────────────────
        fig = plt.figure(figsize=A4)
        _title(fig, "Курси, що потребують уваги", scope_label)
        courses = course_summary(df, min_n=10)
        _scatter_ax(fig.add_axes([0.09, 0.56, 0.84, 0.34]), courses)
        top = courses.sort_values("avg_quality").head(14)
        tc = top[["course", group_col, "n", "avg_quality", "low_score_rate", "weakest_question"]].copy()
        tc["course"] = tc["course"].astype(str).str.slice(0, 34)
        tc[group_col] = tc[group_col].astype(str).str.slice(0, 22)
        tc["avg_quality"] = tc["avg_quality"].map(lambda v: f"{v:.2f}")
        tc["low_score_rate"] = tc["low_score_rate"].map(lambda v: f"{v*100:.0f}%")
        tc = tc.rename(columns={"course": "Курс", group_col: group_name, "n": "n",
                                "avg_quality": "Якість", "low_score_rate": "% ≤3",
                                "weakest_question": "Слабке питання"})
        _table(fig.add_axes([0.03, 0.05, 0.94, 0.46]), tc,
               title="Топ-14 за середньою оцінкою якості (від найнижчої)", fontsize=6.5)
        pdf.savefig(fig); plt.close(fig)

        # ── Page 4: Викладачі ─────────────────────────────────────────────────
        fig = plt.figure(figsize=A4)
        _title(fig, "Найвище оцінені викладачі", scope_label)
        fig.text(0.05, 0.91, "Топ за середньою оцінкою, n ≥ 10. "
                 "Лектори (Q02) і практики (Q04) — окремо.", fontsize=8, color="#666")

        def tt_table(role):
            top = top_teachers(teachers, role, min_n=10, k=18)
            d = top[["teacher", "faculty", "n", "avg"]].copy()
            d["teacher"] = d["teacher"].astype(str).str.slice(0, 30)
            d["faculty"] = d["faculty"].astype(str).str.slice(0, 20)
            d["avg"] = d["avg"].map(lambda v: f"{v:.2f}")
            return d.rename(columns={"teacher": "Викладач", "faculty": "Факультет",
                                     "n": "n", "avg": "Середня"})
        _table(fig.add_axes([0.03, 0.06, 0.45, 0.82]), tt_table("Лектор"),
               title="Лектори", fontsize=6, first_col_w=0.46)
        _table(fig.add_axes([0.52, 0.06, 0.45, 0.82]), tt_table("Практик"),
               title="Практики", fontsize=6, first_col_w=0.46)
        pdf.savefig(fig); plt.close(fig)

        # ── Page 5: Коментарі ─────────────────────────────────────────────────
        fig = plt.figure(figsize=A4)
        _title(fig, "Зворотний зв'язок студентів", scope_label)
        _theme_sentiment_ax(fig.add_axes([0.28, 0.60, 0.64, 0.30]), df)
        ax = fig.add_axes([0.04, 0.05, 0.92, 0.48]); ax.axis("off")
        ax.set_title("Показові негативні коментарі (найнижчі оцінки)",
                     fontsize=11, loc="left", fontweight="bold")
        negs = df[df["comment_useful"] & (df["sentiment"] == "Негативний")].sort_values("avg_overall")
        y = 0.93
        if len(negs) == 0:
            ax.text(0.0, y, "Немає негативних змістовних коментарів у цій вибірці.", fontsize=8)
        for _, r in negs.head(10).iterrows():
            head = f"[{r['avg_overall']:.1f}] {str(r['course'])[:40]} — "
            wrapped = textwrap.fill(head + str(r["comment"]), width=120)
            ax.text(0.0, y, wrapped, fontsize=7, va="top", transform=ax.transAxes)
            y -= 0.018 * (wrapped.count("\n") + 1) + 0.012
            if y < 0.02:
                break
        pdf.savefig(fig); plt.close(fig)

        # ── Page 6: Якість даних ──────────────────────────────────────────────
        fig = plt.figure(figsize=A4)
        _title(fig, "Якість даних і застереження", scope_label)
        course_n = df.groupby("course").size()
        below20 = int((course_n < 20).sum())
        useful = int(df["comment_useful"].sum())
        trivial = int((df["has_comment"] & ~df["comment_useful"]).sum())
        stats = [
            f"Відповідей: {total:,}    ·    Курсів: {df['course'].nunique():,}    ·    "
            f"{group_name}и: {df[group_col].nunique():,}",
            f"Курсів з n < 20: {below20} з {len(course_n)} "
            f"({below20/len(course_n)*100:.0f}%) — приховані у порівняннях за замовчуванням.",
            f"Коментарі: {useful:,} змістовних, {trivial:,} тривіальних (відфільтровано).",
        ]
        for i, s in enumerate(stats):
            fig.text(0.05, 0.90 - i * 0.03, s, fontsize=9, color="#333")

        ax = fig.add_axes([0.04, 0.06, 0.92, 0.70]); ax.axis("off")
        ax.set_title("Відомі обмеження", fontsize=11, loc="left", fontweight="bold")
        limits = [
            "Оцінки сильно зміщені до 5 (≈80% — п'ятірки): середня малоінформативна, "
            "тому використовуємо розподіл, частку ≤3 і відхилення від середнього.",
            "Імена викладачів студенти вписували вручну — автоматично дедупльовані "
            "(консервативно: прізвище + ініціали); рідкісні неоднозначні написання можуть лишатися окремо.",
            "Багато курсів і викладачів мають малу вибірку; у рейтингах показуються лише "
            "ті, хто має достатню кількість відповідей (поріг n).",
            "Питання «Навантаження» — це калібрування (важко/легко), не якість; "
            "виключене з оцінки якості.",
            "Тональність коментарів визначається за середньою оцінкою відповіді (проксі), не NLP.",
            "Поле «Група» порожнє; студенти анонімні; немає дат/семестрів — тренди в часі поки недоступні.",
        ]
        yy = 0.92
        for lim in limits:
            wrapped = "•  " + textwrap.fill(lim, width=110, subsequent_indent="   ")
            ax.text(0.0, yy, wrapped, fontsize=8, va="top", transform=ax.transAxes)
            yy -= 0.022 * (wrapped.count("\n") + 1) + 0.012
        pdf.savefig(fig); plt.close(fig)

    buf.seek(0)
    return buf.getvalue()
