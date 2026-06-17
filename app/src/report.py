"""Executive-summary PDF report.

Rendered with matplotlib (no browser/kaleido needed). matplotlib's default font
DejaVu Sans covers Ukrainian Cyrillic, so text renders correctly everywhere.
"""
import io
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from .metrics import SCORE_COLS, QUALITY_COLS, group_summary, course_summary

SCORE_COLORS = ["#d62728", "#ff7f0e", "#bcbd22", "#2ca02c", "#1f77b4"]


def _table(ax, df: pd.DataFrame, title: str = None, fontsize: int = 7, first_col_w: float = 0.34):
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=11, loc="left", pad=8, fontweight="bold")
    ncols = len(df.columns)
    rest = (1 - first_col_w) / max(ncols - 1, 1)
    col_widths = [first_col_w] + [rest] * (ncols - 1)
    tbl = ax.table(cellText=df.values, colLabels=list(df.columns),
                   loc="upper center", cellLoc="center", colWidths=col_widths)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    tbl.scale(1, 1.35)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#e0e0e0")
        if c == 0:  # first column: left-aligned with a little padding
            cell._loc = "left"
            cell.PAD = 0.03
        if r == 0:
            cell.set_facecolor("#1f77b4")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f5f7fa")
    return tbl


def _score_distribution_ax(ax, df: pd.DataFrame):
    vals = df[SCORE_COLS].values.flatten()
    vals = vals[~np.isnan(vals)].astype(int)
    total = len(vals)
    pcts = [(vals == s).sum() / total * 100 if total else 0 for s in range(1, 6)]
    bars = ax.bar([str(s) for s in range(1, 6)], pcts, color=SCORE_COLORS)
    for b, p in zip(bars, pcts):
        ax.text(b.get_x() + b.get_width() / 2, p + 1, f"{p:.1f}%",
                ha="center", va="bottom", fontsize=8)
    ax.set_title("Розподіл оцінок (1–5)", fontsize=11, loc="left", fontweight="bold")
    ax.set_ylabel("% відповідей", fontsize=8)
    ax.set_ylim(0, max(pcts) * 1.18 if pcts else 1)
    ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _scatter_ax(ax, courses: pd.DataFrame):
    if len(courses) == 0:
        ax.text(0.5, 0.5, "Недостатньо курсів", ha="center", va="center")
        ax.axis("off")
        return
    attention = courses["avg_quality"] < 4.4
    ax.scatter(courses.loc[~attention, "n"], courses.loc[~attention, "low_score_rate"] * 100,
               s=np.clip(courses.loc[~attention, "n"], 10, 400), c="#1f77b4",
               alpha=0.5, edgecolors="none", label="Нормально")
    ax.scatter(courses.loc[attention, "n"], courses.loc[attention, "low_score_rate"] * 100,
               s=np.clip(courses.loc[attention, "n"], 10, 400), c="#d62728",
               alpha=0.6, edgecolors="none", label="Потребує уваги (якість < 4.4)")
    # annotate the 5 highest-negativity courses with enough responses
    worst = courses.sort_values("low_score_rate", ascending=False).head(5)
    for _, r in worst.iterrows():
        ax.annotate(str(r["course"])[:32], (r["n"], r["low_score_rate"] * 100),
                    fontsize=6, xytext=(4, 4), textcoords="offset points", color="#333")
    ax.set_xlabel("Кількість відповідей (n)", fontsize=8)
    ax.set_ylabel("Частка низьких оцінок ≤3, %", fontsize=8)
    ax.set_title("Огляд ризику: курси (праворуч угорі — найважливіші)",
                 fontsize=11, loc="left", fontweight="bold")
    ax.legend(fontsize=7, loc="upper right")
    ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def build_summary_pdf(df: pd.DataFrame, scope_label: str, group_col: str,
                      group_name: str, role_label: str) -> bytes:
    """Build a 2-page executive summary PDF and return its bytes."""
    df = df.copy()
    df["avg_quality"] = df[QUALITY_COLS].mean(axis=1)

    total = len(df)
    all_scores = df[SCORE_COLS].values.flatten()
    all_scores = all_scores[~np.isnan(all_scores)]
    avg = all_scores.mean()
    low_rate = (all_scores <= 3).mean() * 100
    comment_rate = df["comment_useful"].mean() * 100
    units = df[group_col].nunique()

    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        # ── Page 1: overview ──────────────────────────────────────────────────
        fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
        fig.text(0.05, 0.965, "Зведений звіт — опитування студентів",
                 fontsize=17, fontweight="bold")
        fig.text(0.05, 0.945, scope_label, fontsize=11, color="#555")
        fig.text(0.05, 0.930, f"Роль: {role_label}    ·    Сформовано: {date.today():%d.%m.%Y}",
                 fontsize=8, color="#888")

        kpis = [
            ("Відповідей", f"{total:,}"),
            ("Курсів", f"{df['course'].nunique():,}"),
            (group_name + ("и" if group_name == "Факультет" else ""), f"{units:,}"),
            ("Середня", f"{avg:.2f}"),
            ("Оцінок ≤3", f"{low_rate:.1f}%"),
            ("З коментарем", f"{comment_rate:.1f}%"),
        ]
        for i, (lbl, val) in enumerate(kpis):
            x = 0.05 + i * 0.158
            fig.text(x, 0.895, val, fontsize=15, fontweight="bold")
            fig.text(x, 0.878, lbl, fontsize=7.5, color="#666")

        ax_dist = fig.add_axes([0.08, 0.60, 0.84, 0.20])
        _score_distribution_ax(ax_dist, df)

        # general data table
        gs = group_summary(df, group_col=group_col)
        cols = [group_col, "n", "courses", "avg_quality", "lect_resp", "pract_resp", "low_rate", "comment_rate"]
        t = gs[cols].copy()
        t[group_col] = t[group_col].astype(str).str.slice(0, 38)
        for c in ("avg_quality", "lect_resp", "pract_resp"):
            t[c] = t[c].map(lambda v: f"{v:.2f}")
        t["low_rate"] = t["low_rate"].map(lambda v: f"{v:.1f}%")
        t["comment_rate"] = t["comment_rate"].map(lambda v: f"{v:.1f}%")
        t = t.rename(columns={
            group_col: group_name, "n": "Відп.", "courses": "Курсів",
            "avg_quality": "Якість", "lect_resp": "Лектори", "pract_resp": "Практики",
            "low_rate": "% ≤3", "comment_rate": "% комент.",
        })
        ax_tbl = fig.add_axes([0.04, 0.04, 0.92, 0.50])
        _table(ax_tbl, t, title=f"Загальні дані за {'кафедрами' if group_col == 'specialty' else 'факультетами'}",
               fontsize=6.5)
        pdf.savefig(fig)
        plt.close(fig)

        # ── Page 2: risk ──────────────────────────────────────────────────────
        fig2 = plt.figure(figsize=(8.27, 11.69))
        fig2.text(0.05, 0.965, "Курси, що потребують уваги", fontsize=17, fontweight="bold")
        fig2.text(0.05, 0.945, scope_label, fontsize=11, color="#555")

        courses = course_summary(df, min_n=10)
        ax_sc = fig2.add_axes([0.09, 0.56, 0.84, 0.34])
        _scatter_ax(ax_sc, courses)

        top = courses.sort_values("shrunk_quality").head(12).copy()
        tc = top[["course", group_col, "n", "shrunk_quality", "low_score_rate", "weakest_question"]].copy()
        tc["course"] = tc["course"].astype(str).str.slice(0, 34)
        tc[group_col] = tc[group_col].astype(str).str.slice(0, 22)
        tc["shrunk_quality"] = tc["shrunk_quality"].map(lambda v: f"{v:.2f}")
        tc["low_score_rate"] = tc["low_score_rate"].map(lambda v: f"{v*100:.0f}%")
        tc = tc.rename(columns={
            "course": "Курс", group_col: group_name, "n": "n",
            "shrunk_quality": "Згладж.", "low_score_rate": "% ≤3",
            "weakest_question": "Слабке питання",
        })
        ax_tc = fig2.add_axes([0.03, 0.04, 0.94, 0.46])
        _table(ax_tc, tc, title="Топ-12 за згладженою оцінкою (від найнижчої)", fontsize=6.5)
        pdf.savefig(fig2)
        plt.close(fig2)

    buf.seek(0)
    return buf.getvalue()
