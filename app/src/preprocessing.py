"""
One-time preprocessing: clean raw CSV → feedback.parquet
Run: python -m app.src.preprocessing
"""
import re
import pandas as pd
from pathlib import Path

from .names import build_canonical_map, canonical_names

# Single source of truth: the CSV at the project root (the file you edit).
RAW_CSV = Path(__file__).parent.parent.parent / "_.csv"
PARQUET = Path(__file__).parent.parent / "data" / "feedback.parquet"
TEACHERS_PARQUET = Path(__file__).parent.parent / "data" / "teachers.parquet"

SCORE_COLS = {
    "Q01_profession":   "Q01_Дисципліна в цілому->Оцініть, як допоміг Вам курс дисципліни краще зрозуміти майбутню професію.",
    "Q01_elearn":       "Q01_Дисципліна в цілому->Оцініть зручність, повноту та якість навчальних матеріалів з дисципліни на порталі Elearn.",
    "Q01_workload":     "Q01_Дисципліна в цілому->Наскільки навантаження за курсом (кількість завдань та їх складність) є посильним та адекватним кількості годин?",
    "Q01_criteria":     "Q01_Дисципліна в цілому->Наскільки чіткими та зрозумілими були критерії оцінювання та дотримувалася прозорість при виставленні балів?",
    "Q01_skills":       "Q01_Дисципліна в цілому->Оцініть задоволеність практичними навичками, здобутими під час вивчення дисципліни?",
    "Q03_clarity":      "Q03_Робота лектора->Наскільки цікаво, структуровано та зрозуміло лектор пояснював теоретичний матеріал?",
    "Q03_questions":    "Q03_Робота лектора->Оцініть готовність лектора фахово відповідати на запитання та наводити практичні приклади.",
    "Q03_ethics":       "Q03_Робота лектора->Оцініть, як дотримувався викладач професійної етики та поваги у спілкуванні зі студентами?",
    "Q03_engagement":   "Q03_Робота лектора->Оцініть, як залучав викладач аудиторію до активної роботи, використовуючи сучасні методи навчання (дискусії, онлайн-платформи, відеоматеріали)?",
    "Q05_explanation":  "Q05_Оцінка практика->Наскільки зрозуміло викладач пояснював алгоритм виконання завдань та допомагав у разі труднощів?",
    "Q05_consultation": "Q05_Оцінка практика->Оцініть, чи можна було отримати від викладача допомогу або консультацію, якщо матеріал був незрозумілим?",
    "Q05_fairness":     "Q05_Оцінка практика->Наскільки прозоро, справедливо та вчасно коментувалися та оцінювалися ваші практичні (семінарські, лабораторні) роботи?",
}

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

BLOCK_LABELS = {
    "Q01": "Дисципліна",
    "Q03": "Лектор",
    "Q05": "Практик",
}


def _clean_text(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"[☑✓✔️☑✓✔️`]+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# Comments that carry no analytical signal
TRIVIAL_PHRASES = {
    "", "-", "--", "---", ".", "..", "...", "?", "!", "ні", "нема", "немає",
    "так", "ок", "ok", "все", "все добре", "добре", "все супер", "супер",
    "відмінно", "чудово", "дякую", "дякую за все", "все чудово", "все ок",
    "все відмінно", "все класно", "класно", "норм", "нормально", "все гаразд",
    "+", "++", "все влаштовує", "задоволена", "задоволений", "все подобається",
}


def _is_trivial(text: str) -> bool:
    if not text:
        return True
    t = text.lower().strip()
    t_clean = re.sub(r"[.,!?;:)(\"'\s]+", " ", t).strip()
    if len(t_clean) < 4:
        return True
    if t_clean in TRIVIAL_PHRASES:
        return True
    # only digits / punctuation / emoji
    if not re.search(r"[а-яіїєґa-z]{4,}", t):
        return True
    return False


# Keyword-based theme tagging (Ukrainian stems). A comment may match several themes.
THEMES = {
    "theme_workload":     ["навантаж", "перевантаж", "забагато завдан", "обсяг завдан",
                            "дуже багато завдан", "часу не вистача", "не встига", "дедлайн"],
    "theme_materials":    ["elearn", "ілерн", "матеріал", "презентац", "конспект",
                            "підручник", "відеоматеріал", "лекці на порталі", "запис лекці"],
    "theme_grading":      ["оцінюван", "критері", "справедлив", "несправедлив",
                            "прозор", "завищ", "занижен", "необ'єктивн", "необєктивн", "бали"],
    "theme_communication":["спілкуван", "груб", "хамств", "повага", "неповаг", "етик",
                            "ставлен", "зневаж", "крич", "принижу", "конфлікт", "токсичн"],
    "theme_practical":    ["практичн", "лаборатор", "семінар", "практик"],
    "theme_organization": ["організац", "розклад", "запізн", "пропуска", "не проводи",
                           "не з'явля", "не зявля", "скасов", "переноси"],
    "theme_teaching":     ["поясн", "цікав", "нудно", "незрозуміл", "зрозуміл",
                           "структур", "монотон", "читав з листа", "просто диктув", "не вчить"],
}


def _tag_themes(text: str) -> dict[str, bool]:
    t = text.lower()
    return {theme: any(kw in t for kw in kws) for theme, kws in THEMES.items()}


def run():
    df = pd.read_csv(RAW_CSV, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    # Rename identifier columns
    df = df.rename(columns={
        "Відповідь":              "response_id",
        "Факультет":              "faculty",
        "Спеціальність (Кафедра)": "specialty",
        "Курс":                   "course_raw",
        "Повна назва":            "student_anon",
        "Q02_Прізвище лектора":   "lecturer",
        "Q04_Прізвище практика":  "practitioner",
        "Q06_Особлива думка":     "comment",
    })

    # Clean text fields
    for col in ("faculty", "specialty", "course_raw", "lecturer", "practitioner"):
        df[col] = df[col].apply(_clean_text)

    df["course"] = df["course_raw"]
    df.drop(columns=["course_raw", "Група", "student_anon"], errors="ignore", inplace=True)

    # Rename and coerce score columns to numeric
    rename_map = {v: k for k, v in SCORE_COLS.items()}
    df = df.rename(columns=rename_map)

    score_keys = list(SCORE_COLS.keys())
    for col in score_keys:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Derived aggregate columns
    q01_cols = [c for c in score_keys if c.startswith("Q01_")]
    q03_cols = [c for c in score_keys if c.startswith("Q03_")]
    q05_cols = [c for c in score_keys if c.startswith("Q05_")]

    df["avg_discipline"] = df[q01_cols].mean(axis=1)
    df["avg_lecturer"] = df[q03_cols].mean(axis=1)
    df["avg_practitioner"] = df[q05_cols].mean(axis=1)
    df["avg_overall"] = df[score_keys].mean(axis=1)

    df["comment"] = df["comment"].fillna("").astype(str).str.strip()
    df["has_comment"] = df["comment"].str.len() > 0

    # ── Comment enrichment ────────────────────────────────────────────────────
    df["comment_trivial"] = df["comment"].apply(_is_trivial)
    # substantive = has a comment AND not trivial
    df["comment_useful"] = df["has_comment"] & ~df["comment_trivial"]

    # Sentiment derived from the response's overall score (proxy, no NLP needed)
    def _sentiment(avg):
        if avg <= 3.5:
            return "Негативний"
        if avg >= 4.5:
            return "Позитивний"
        return "Нейтральний"
    df["sentiment"] = df["avg_overall"].apply(_sentiment)

    # Theme tags (only meaningful for useful comments)
    theme_cols = list(THEMES.keys())
    tags = df["comment"].apply(_tag_themes).apply(pd.Series)
    for col in theme_cols:
        df[col] = tags[col] & df["comment_useful"]

    # ── Teacher name canonicalization ─────────────────────────────────────────
    # Build one corpus-wide map from both lecturer and practitioner columns,
    # then resolve each cell to a deduplicated list of canonical names.
    all_cells = pd.concat([df["lecturer"], df["practitioner"]]).tolist()
    cmap = build_canonical_map(all_cells)

    lec_lists = df["lecturer"].apply(lambda c: canonical_names(c, cmap))
    prac_lists = df["practitioner"].apply(lambda c: canonical_names(c, cmap))
    # Cleaned display strings on the feedback table (co-teachers joined)
    df["lecturer"] = lec_lists.apply(lambda lst: "; ".join(lst))
    df["practitioner"] = prac_lists.apply(lambda lst: "; ".join(lst))

    df.to_parquet(PARQUET, index=False)
    print(f"Saved {len(df):,} rows → {PARQUET}")

    # ── Teacher long-table (one row per response × canonical teacher) ─────────
    base_cols = (["response_id", "faculty", "specialty", "course"] + score_keys +
                 ["avg_lecturer", "avg_practitioner", "avg_overall",
                  "comment", "comment_useful", "sentiment"])
    parts = []
    for role, lists in (("Лектор", lec_lists), ("Практик", prac_lists)):
        sub = df[base_cols].copy()
        sub["teacher"] = lists.values
        sub = sub.explode("teacher")
        sub = sub[sub["teacher"].notna() & (sub["teacher"].astype(str).str.len() > 0)]
        sub["role"] = role
        parts.append(sub)
    teachers = pd.concat(parts, ignore_index=True)
    teachers.to_parquet(TEACHERS_PARQUET, index=False)
    print(f"Saved {len(teachers):,} teacher-rows ({teachers['teacher'].nunique():,} identities) → {TEACHERS_PARQUET}")
    return df


if __name__ == "__main__":
    run()
