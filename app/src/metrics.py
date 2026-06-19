"""Aggregation helpers shared across pages."""
import pandas as pd

SCORE_COLS = [
    "Q01_profession", "Q01_elearn", "Q01_workload", "Q01_criteria", "Q01_skills",
    "Q03_clarity", "Q03_questions", "Q03_ethics", "Q03_engagement",
    "Q05_explanation", "Q05_consultation", "Q05_fairness",
]

Q01_COLS = [c for c in SCORE_COLS if c.startswith("Q01_")]
Q03_COLS = [c for c in SCORE_COLS if c.startswith("Q03_")]
Q05_COLS = [c for c in SCORE_COLS if c.startswith("Q05_")]

QUESTION_LABELS = {
    "Q01_profession":   "Зв'язок з професією",
    "Q01_elearn":       "Матеріали Elearn",
    "Q01_workload":     "Навантаження",
    "Q01_criteria":     "Прозорість оцінювання",
    "Q01_skills":       "Практичні навички",
    "Q03_clarity":      "Пояснення лектора",
    "Q03_questions":    "Відповіді на запитання",
    "Q03_ethics":       "Етика лектора",
    "Q03_engagement":   "Залучення аудиторії",
    "Q05_explanation":  "Пояснення практика",
    "Q05_consultation": "Консультація практика",
    "Q05_fairness":     "Справедливість оцінювання",
}

BLOCK_LABELS = {"Q01": "Дисципліна", "Q03": "Лектор", "Q05": "Практик"}

# "Навантаження" is a calibration metric (heavy vs light), not a quality metric,
# so it is excluded from the quality average used for rankings.
QUALITY_COLS = [c for c in SCORE_COLS if c != "Q01_workload"]

# Theme columns produced in preprocessing → human labels
THEME_LABELS = {
    "theme_workload":      "Навантаження",
    "theme_materials":     "Матеріали / Elearn",
    "theme_grading":       "Оцінювання",
    "theme_communication": "Комунікація / етика",
    "theme_practical":     "Практичні заняття",
    "theme_organization":  "Організація",
    "theme_teaching":      "Якість викладання",
}


def score_dist(series: pd.Series) -> dict[int, int]:
    counts = series.value_counts().reindex(range(1, 6), fill_value=0)
    return counts.to_dict()


def low_score_rate(series: pd.Series) -> float:
    valid = series.dropna()
    if len(valid) == 0:
        return 0.0
    return (valid <= 3).sum() / len(valid)


def top_box_rate(series: pd.Series) -> float:
    valid = series.dropna()
    if len(valid) == 0:
        return 0.0
    return (valid == 5).sum() / len(valid)


def course_summary(df: pd.DataFrame, min_n: int = 20) -> pd.DataFrame:
    key = ["faculty", "specialty", "course"]
    df = df.copy()
    df["avg_quality"] = df[QUALITY_COLS].mean(axis=1)

    grp = df.groupby(key)
    agg = grp.agg(
        n=("avg_overall", "count"),
        avg_overall=("avg_overall", "mean"),
        avg_quality=("avg_quality", "mean"),
        avg_workload=("Q01_workload", "mean"),
        avg_discipline=("avg_discipline", "mean"),
        avg_lecturer=("avg_lecturer", "mean"),
        avg_practitioner=("avg_practitioner", "mean"),
        comment_count=("comment_useful", "sum"),
        lecturer=("lecturer", lambda s: s.mode().iloc[0] if len(s) else ""),
        practitioner=("practitioner", lambda s: s.mode().iloc[0] if len(s) else ""),
    ).reset_index()

    # low score rate (per-response avg_overall <= 3)
    low_rows = df[df["avg_overall"] <= 3].groupby(key).size().reset_index(name="low_n")
    agg = agg.merge(low_rows, on=key, how="left")
    agg["low_n"] = agg["low_n"].fillna(0)
    agg["low_score_rate"] = agg["low_n"] / agg["n"]

    # Weakest question per course (quality questions only — workload isn't "bad")
    q_means = df.groupby(key)[QUALITY_COLS].mean()
    agg["weakest_question"] = q_means.idxmin(axis=1).map(QUESTION_LABELS).values
    agg["weakest_score"] = q_means.min(axis=1).values

    agg = agg[agg["n"] >= min_n].copy()
    return agg.sort_values("avg_quality", ascending=True)


def faculty_question_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("faculty")[SCORE_COLS].mean().rename(columns=QUESTION_LABELS)


def group_question_means(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Per-question means grouped by any column (faculty or specialty/department)."""
    return df.groupby(group_col)[SCORE_COLS].mean().rename(columns=QUESTION_LABELS)


def group_counts(df: pd.DataFrame, group_col: str) -> pd.Series:
    return df.groupby(group_col).size()


def _teacher_weighted(teachers: pd.DataFrame, group_col: str, role: str, score_col: str) -> pd.Series:
    """Average of per-teacher averages within each group (every teacher counts once).

    Uses the canonical teacher long-table so co-teachers are counted individually.
    """
    sub = teachers[teachers["role"] == role]
    per_teacher = sub.groupby([group_col, "teacher"])[score_col].mean()
    return per_teacher.groupby(level=0).mean()


def group_summary(df: pd.DataFrame, group_col: str = "faculty",
                  teachers: pd.DataFrame = None) -> pd.DataFrame:
    """General summary table grouped by faculty (or department).

    Provides BOTH lecturer/practitioner averages:
      *_resp  = response-weighted (each student response counts once)
      *_tchr  = teacher-weighted  (each canonical teacher counts once) — needs `teachers`
    """
    df = df.copy()
    df["avg_quality"] = df[QUALITY_COLS].mean(axis=1)

    g = df.groupby(group_col)
    out = g.agg(
        n=("avg_overall", "count"),
        courses=("course", "nunique"),
        avg_quality=("avg_quality", "mean"),
        lect_resp=("avg_lecturer", "mean"),
        pract_resp=("avg_practitioner", "mean"),
        comment_rate=("comment_useful", "mean"),
    )
    if group_col == "faculty":
        out["departments"] = g["specialty"].nunique()

    if teachers is not None:
        out["lect_tchr"] = _teacher_weighted(teachers, group_col, "Лектор", "avg_lecturer")
        out["pract_tchr"] = _teacher_weighted(teachers, group_col, "Практик", "avg_practitioner")
    else:
        out["lect_tchr"] = pd.NA
        out["pract_tchr"] = pd.NA

    low = df[df["avg_overall"] <= 3].groupby(group_col).size()
    out["low_rate"] = (low / out["n"] * 100).fillna(0)
    out["comment_rate"] = out["comment_rate"] * 100

    return out.reset_index().sort_values("avg_quality")


# Role → (block average column, that block's question columns)
ROLE_CONFIG = {
    "Лектор": ("avg_lecturer", Q03_COLS),
    "Практик": ("avg_practitioner", Q05_COLS),
}


def teacher_summary(teachers: pd.DataFrame, role: str, min_n: int = 10) -> pd.DataFrame:
    """Per-canonical-teacher stats for one role (Лектор / Практик).

    Returns n, average, low-score rate, faculties, courses, useful-comment count,
    and the per-question averages for that block. Sorted best-first by average.
    """
    score_col, qcols = ROLE_CONFIG[role]
    sub = teachers[teachers["role"] == role]
    if len(sub) == 0:
        return pd.DataFrame()

    g = sub.groupby("teacher")
    out = g.agg(
        n=(score_col, "count"),
        avg=(score_col, "mean"),
        courses=("course", "nunique"),
        faculties=("faculty", "nunique"),
        faculty=("faculty", lambda s: s.mode().iloc[0] if len(s) else ""),
        comment_count=("comment_useful", "sum"),
    )
    # per-question means for this block
    qmeans = g[qcols].mean()
    out = out.join(qmeans)

    low = sub[sub[score_col] <= 3].groupby("teacher").size()
    out["low_rate"] = (low / out["n"] * 100).fillna(0)

    out = out[out["n"] >= min_n].reset_index()
    return out.sort_values("avg", ascending=False)


def top_teachers(teachers: pd.DataFrame, role: str, min_n: int = 10, k: int = 10) -> pd.DataFrame:
    """Top-k teachers of a role by average score (n ≥ min_n)."""
    return teacher_summary(teachers, role, min_n=min_n).head(k)


def teacher_question_ranking(teachers: pd.DataFrame, role: str, question_col: str,
                             min_n: int = 10) -> pd.DataFrame:
    """Rank teachers of a role by a SINGLE question topic (by average, n ≥ min_n)."""
    sub = teachers[teachers["role"] == role]
    if len(sub) == 0:
        return pd.DataFrame()
    g = sub.groupby("teacher")
    out = g.agg(
        n=(question_col, "count"),
        q_avg=(question_col, "mean"),
        courses=("course", "nunique"),
        faculty=("faculty", lambda s: s.mode().iloc[0] if len(s) else ""),
    )
    out["low_rate"] = (sub[sub[question_col] <= 3].groupby("teacher").size() / out["n"] * 100).fillna(0)
    out = out[out["n"] >= min_n].reset_index()
    return out.sort_values("q_avg", ascending=False)


def department_summary(df: pd.DataFrame, min_n: int = 20) -> pd.DataFrame:
    """Per-department (faculty + specialty) summary, sorted worst-first by avg quality.

    Keyed by (faculty, specialty) so identically named departments in different
    faculties stay distinct.
    """
    key = ["faculty", "specialty"]
    df = df.copy()
    df["avg_quality"] = df[QUALITY_COLS].mean(axis=1)

    g = df.groupby(key)
    out = g.agg(
        n=("avg_overall", "count"),
        courses=("course", "nunique"),
        avg_quality=("avg_quality", "mean"),
        lect_resp=("avg_lecturer", "mean"),
        pract_resp=("avg_practitioner", "mean"),
        comment_count=("comment_useful", "sum"),
    ).reset_index()

    low = df[df["avg_overall"] <= 3].groupby(key).size().reset_index(name="low_n")
    out = out.merge(low, on=key, how="left")
    out["low_n"] = out["low_n"].fillna(0)
    out["low_rate"] = out["low_n"] / out["n"] * 100

    qm = df.groupby(key)[QUALITY_COLS].mean()
    out["weakest_question"] = qm.idxmin(axis=1).map(QUESTION_LABELS).values

    out = out[out["n"] >= min_n].copy()
    return out.sort_values("avg_quality")


def faculty_counts(df: pd.DataFrame) -> pd.Series:
    """Number of responses per faculty (for n-display on the heatmap)."""
    return df.groupby("faculty").size()


def university_question_means(df: pd.DataFrame) -> pd.Series:
    """Response-weighted university average per question (index = question labels).

    This is the *true* average over all individual responses (each response counts
    once), NOT the mean of faculty means. Use this as the deviation baseline so the
    comparison is fair and stable regardless of how many faculties are shown.
    """
    return df[SCORE_COLS].mean().rename(QUESTION_LABELS)


def score_distribution_long(df: pd.DataFrame, group_col: str = None) -> pd.DataFrame:
    """Return melted score distribution for stacked bar charts."""
    all_scores = df[SCORE_COLS].melt(var_name="question", value_name="score")
    if group_col:
        all_scores[group_col] = df[group_col].repeat(len(SCORE_COLS)).values
    all_scores["score"] = pd.to_numeric(all_scores["score"], errors="coerce")
    return all_scores.dropna(subset=["score"])
