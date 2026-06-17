"""Reusable Plotly chart builders."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

SCORE_COLORS = {
    1: "#d62728",
    2: "#ff7f0e",
    3: "#bcbd22",
    4: "#2ca02c",
    5: "#1f77b4",
}

PALETTE = px.colors.qualitative.Set2


def stacked_score_bar(
    df: pd.DataFrame,
    group_col: str,
    score_cols: list[str],
    title: str = "",
    top_n: int = None,
) -> go.Figure:
    """Stacked 1-5 bar chart grouped by group_col."""
    melted = df.melt(id_vars=[group_col], value_vars=score_cols, value_name="score")
    melted["score"] = pd.to_numeric(melted["score"], errors="coerce")
    melted = melted.dropna(subset=["score"])
    melted["score"] = melted["score"].astype(int)

    dist = (
        melted.groupby([group_col, "score"])
        .size()
        .reset_index(name="count")
    )
    totals = dist.groupby(group_col)["count"].transform("sum")
    dist["pct"] = dist["count"] / totals * 100

    if top_n:
        order = (
            dist[dist["score"].isin([1, 2, 3])]
            .groupby(group_col)["pct"].sum()
            .sort_values(ascending=False)
            .head(top_n)
            .index.tolist()
        )
        dist = dist[dist[group_col].isin(order)]
    else:
        order = dist[group_col].unique().tolist()

    fig = go.Figure()
    for score in range(1, 6):
        subset = dist[dist["score"] == score]
        fig.add_trace(go.Bar(
            name=str(score),
            x=subset[group_col],
            y=subset["pct"],
            marker_color=SCORE_COLORS[score],
            hovertemplate=f"Оцінка {score}: %{{y:.1f}}%<extra></extra>",
        ))

    fig.update_layout(
        barmode="stack",
        title=title,
        xaxis_title="",
        yaxis_title="% відповідей",
        legend_title="Оцінка",
        height=420,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def heatmap_faculty_question(
    pivot: pd.DataFrame,
    counts: pd.Series = None,
    mode: str = "deviation",
    baseline: pd.Series = None,
    row_label: str = "Факультет",
) -> go.Figure:
    """Faculty × question heatmap.

    mode="deviation": color = faculty mean minus the university baseline for that
        question (diverging), so problem spots stand out despite the positive skew.
    mode="absolute":  color = raw 1-5 mean.
    `counts` (responses per faculty) adds n to the y-axis labels and hover.
    `baseline` (per-question reference means, index = question labels) is the value
        deviation is measured against. Pass the response-weighted university average
        from the FULL dataset; if omitted, falls back to the (less correct) mean of
        the visible faculty means.
    """
    means = pivot.values
    mean_text = [[f"{v:.2f}" for v in row] for row in means]  # absolute mean, for hover

    if mode == "deviation":
        if baseline is not None:
            ref = baseline.reindex(pivot.columns).values
        else:
            ref = pivot.mean(axis=0).values  # fallback: mean of visible faculty means
        z = means - ref
        cell_text = [[f"{v:+.2f}" for v in row] for row in z]  # signed Δ in the cell
        colorscale = "RdBu"
        zmid, zmin, zmax = 0, -0.5, 0.5
        color_label = "Відхилення"
    else:
        z = means
        cell_text = mean_text
        colorscale = "RdYlGn"
        zmid, zmin, zmax = None, 1, 5
        color_label = "Середня"

    y_labels = pivot.index.tolist()
    if counts is not None:
        y_labels = [f"{f}  (n={int(counts.get(f, 0)):,})" for f in pivot.index]

    customdata = [[mean_text[i][j] for j in range(len(pivot.columns))] for i in range(len(pivot.index))]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=pivot.columns.tolist(),
        y=y_labels,
        colorscale=colorscale,
        zmid=zmid,
        zmin=zmin,
        zmax=zmax,
        customdata=customdata,
        text=cell_text,
        texttemplate="%{text}",
        textfont=dict(size=10),
        colorbar=dict(title=color_label),
        hovertemplate=(
            f"{row_label}: %{{y}}<br>Питання: %{{x}}<br>"
            "Середня: %{customdata}<br>Δ від середнього: %{z:+.2f}<extra></extra>"
        ),
    ))
    title = ("Heatmap: відхилення від середнього по університету"
             if mode == "deviation" else f"Середні оцінки: {row_label.lower()} × питання")
    fig.update_layout(
        title=title,
        xaxis=dict(tickangle=-40, tickfont=dict(size=11)),
        yaxis=dict(tickfont=dict(size=11)),
        height=560,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def horizontal_bar_questions(row: pd.Series, labels: dict, title: str = "") -> go.Figure:
    """Horizontal bar for a single course/teacher question profile."""
    data = {labels.get(k, k): v for k, v in row.items() if k in labels}
    df = pd.DataFrame({"question": list(data.keys()), "score": list(data.values())})
    df = df.sort_values("score")

    colors = [
        "#d62728" if s < 3.5 else "#ff7f0e" if s < 4.0
        else "#bcbd22" if s < 4.5 else "#2ca02c"
        for s in df["score"]
    ]

    fig = go.Figure(go.Bar(
        x=df["score"],
        y=df["question"],
        orientation="h",
        marker_color=colors,
        text=[f"{s:.2f}" for s in df["score"]],
        textposition="outside",
    ))
    fig.update_layout(
        title=title,
        xaxis=dict(range=[1, 5.4], title="Середня оцінка"),
        yaxis_title="",
        height=380,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def score_distribution_bar(scores: pd.Series, title: str = "") -> go.Figure:
    dist = scores.value_counts().reindex(range(1, 6), fill_value=0)
    total = dist.sum()
    pcts = dist / total * 100

    fig = go.Figure(go.Bar(
        x=[str(i) for i in range(1, 6)],
        y=pcts.values,
        marker_color=[SCORE_COLORS[i] for i in range(1, 6)],
        text=[f"{p:.1f}%" for p in pcts.values],
        textposition="outside",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Оцінка",
        yaxis_title="% відповідей",
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def scatter_risk(
    df: pd.DataFrame,
    color_col: str = "faculty",
    color_label: str = "Факультет",
    title: str = "Курси: обсяг відповідей vs. частка негативних оцінок",
) -> go.Figure:
    fig = px.scatter(
        df,
        x="n",
        y="low_score_rate",
        size="n",
        color=color_col,
        hover_data={"course": True, "lecturer": True, "avg_overall": ":.2f", "n": True},
        labels={
            "n": "Кількість відповідей",
            "low_score_rate": "Частка низьких оцінок (≤3)",
            color_col: color_label,
        },
        title=title,
        height=480,
    )
    fig.update_traces(marker=dict(opacity=0.7))
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    return fig


def treemap_courses(df: pd.DataFrame) -> go.Figure:
    agg = df.groupby(["faculty", "specialty", "course"]).size().reset_index(name="n")
    fig = px.treemap(
        agg,
        path=["faculty", "specialty", "course"],
        values="n",
        title="Обсяг відповідей: факультет → спеціальність → курс",
        height=550,
    )
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    return fig
