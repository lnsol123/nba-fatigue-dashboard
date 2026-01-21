# app.py
# NBA Fatigue Dashboard (minimal, portfolio-ready)
# Data inputs:
#   - data_processed/nba_2324_fatigue_matchups.csv  (required)
#   - data_processed/team_games_2324_with_fatigue.csv (optional)

import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------
# Page config (minimal)
# -----------------------------
st.set_page_config(
    page_title="NBA Fatigue Dashboard",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.5rem; padding-bottom: 1.5rem; }
      [data-testid="stSidebar"] { padding-top: 1.0rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Helpers
# -----------------------------
NBA_TEAMS = sorted([
    "Atlanta Hawks", "Boston Celtics", "Brooklyn Nets", "Charlotte Hornets", "Chicago Bulls",
    "Cleveland Cavaliers", "Dallas Mavericks", "Denver Nuggets", "Detroit Pistons",
    "Golden State Warriors", "Houston Rockets", "Indiana Pacers", "LA Clippers",
    "Los Angeles Lakers", "Memphis Grizzlies", "Miami Heat", "Milwaukee Bucks",
    "Minnesota Timberwolves", "New Orleans Pelicans", "New York Knicks",
    "Oklahoma City Thunder", "Orlando Magic", "Philadelphia 76ers", "Phoenix Suns",
    "Portland Trail Blazers", "Sacramento Kings", "San Antonio Spurs", "Toronto Raptors",
    "Utah Jazz", "Washington Wizards"
])


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    # case-insensitive fallback
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def to_datetime_safe(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def numeric_safe(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def add_simple_trendline(fig: go.Figure, x: np.ndarray, y: np.ndarray, name: str) -> go.Figure:
    """
    Robust linear trendline without statsmodels.
    Avoids crashes from NaN/inf, too-few points, constant x, or numerical issues.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x2, y2 = x[mask], y[mask]

    if len(x2) < 8:
        return fig

    if np.unique(x2).size < 2 or np.ptp(x2) < 1e-9:
        return fig

    try:
        # Center/scale for stability
        xm = x2.mean()
        xs = x2.std()
        if xs < 1e-12:
            return fig
        xz = (x2 - xm) / xs

        m, b = np.polyfit(xz, y2, 1)

        x_line = np.linspace(x2.min(), x2.max(), 120)
        x_line_z = (x_line - xm) / xs
        y_line = m * x_line_z + b

        fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines", name=name))
        return fig
    except Exception:
        return fig


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def filter_to_nba_teams(df: pd.DataFrame, home_col: str, away_col: str) -> pd.DataFrame:
    return df[df[home_col].isin(NBA_TEAMS) & df[away_col].isin(NBA_TEAMS)].copy()


def compute_win_pct_by_bin(df: pd.DataFrame, x_col: str, win_col: str, bin_size: float) -> pd.DataFrame:
    d = df[[x_col, win_col]].copy()
    d[x_col] = numeric_safe(d[x_col])
    d[win_col] = numeric_safe(d[win_col])
    d = d.dropna(subset=[x_col, win_col])

    if d.empty:
        return pd.DataFrame(columns=["fatigue_bin", "home_win_pct", "games"])

    # Bin edges like Tableau-style: center bins at multiples
    # We'll compute bin label = floor(x/bin_size)*bin_size
    d["fatigue_bin"] = np.floor(d[x_col] / bin_size) * bin_size
    out = (
        d.groupby("fatigue_bin", as_index=False)
         .agg(home_win_pct=(win_col, "mean"), games=(win_col, "size"))
         .sort_values("fatigue_bin")
    )
    return out


def fatigue_advantage_category(df: pd.DataFrame, fatigue_diff_col: str) -> pd.Series:
    x = numeric_safe(df[fatigue_diff_col])
    # fatigue_diff = home - away
    # negative => home less fatigued
    return np.select(
        [x < 0, x > 0],
        ["Home fatigue advantage", "Away fatigue advantage"],
        default="Even"
    )


# -----------------------------
# Sidebar (data paths + filters)
# -----------------------------
st.sidebar.header("Data")

default_matchups = "data_processed/nba_2324_fatigue_matchups.csv"
default_team_games = "data_processed/team_games_2324_with_fatigue.csv"

matchups_path = st.sidebar.text_input(
    "Matchups CSV path", value=default_matchups)
team_games_path = st.sidebar.text_input(
    "Team-games CSV path (optional)", value=default_team_games)

if not Path(matchups_path).exists():
    st.error(f"Matchups file not found: {matchups_path}")
    st.stop()

matchups = load_csv(matchups_path)

# Identify core columns (robust to naming)
col_game_id = pick_col(matchups, ["GAME_ID", "Game Id", "game_id"])
col_game_date = pick_col(matchups, ["GAME_DATE", "Game Date", "game_date"])
col_home_team = pick_col(matchups, ["home_team", "Home Team"])
col_away_team = pick_col(matchups, ["away_team", "Away Team"])
col_home_pts = pick_col(matchups, ["home_pts", "Home Pts", "Home Points"])
col_away_pts = pick_col(matchups, ["away_pts", "Away Pts", "Away Points"])
col_home_win = pick_col(matchups, ["home_win", "Home Win"])
col_point_diff = pick_col(
    matchups, ["point_diff", "Point Diff", "Point Margin"])
col_fatigue_diff = pick_col(matchups, ["fatigue_diff", "Fatigue Diff"])
col_rest_diff = pick_col(matchups, ["rest_diff", "Rest Diff"])
col_travel_diff = pick_col(matchups, ["travel_diff", "Travel Diff"])

missing_required = [name for name, col in {
    "GAME_DATE": col_game_date,
    "home_team": col_home_team,
    "away_team": col_away_team,
    "home_win": col_home_win,
    "point_diff": col_point_diff,
    "fatigue_diff": col_fatigue_diff,
}.items() if col is None]

if missing_required:
    st.error("Your matchups CSV is missing required columns: " +
             ", ".join(missing_required))
    st.stop()

# Normalize types
df = matchups.copy()
df[col_game_date] = to_datetime_safe(df[col_game_date])
df[col_home_win] = numeric_safe(df[col_home_win])
df[col_point_diff] = numeric_safe(df[col_point_diff])
df[col_fatigue_diff] = numeric_safe(df[col_fatigue_diff])

# Filter out non-NBA teams (fix for the weird teams)
df = filter_to_nba_teams(df, col_home_team, col_away_team)

# Add derived fields
df["Fatigue Advantage"] = fatigue_advantage_category(df, col_fatigue_diff)

# Date range filter
st.sidebar.header("Filters")

min_date = df[col_game_date].min()
max_date = df[col_game_date].max()
if pd.isna(min_date) or pd.isna(max_date):
    # If parsing failed, fall back to no date filter
    date_range = None
    st.sidebar.warning(
        "Game Date could not be parsed. Date filters are disabled.")
else:
    date_range = st.sidebar.date_input(
        "Game date range",
        value=(min_date.date(), max_date.date()),
        min_value=min_date.date(),
        max_value=max_date.date(),
    )

team_mode = st.sidebar.selectbox(
    "Team filter", ["All games", "Home team", "Away team"])

team_selected = None
if team_mode != "All games":
    team_selected = st.sidebar.selectbox(
        "Select team", ["(Choose)"] + NBA_TEAMS)
    if team_selected == "(Choose)":
        team_selected = None

bin_size = st.sidebar.selectbox("Fatigue bin size", [0.25, 0.5, 1.0], index=1)
top_n = st.sidebar.slider("Top N teams (rankings)",
                          min_value=5, max_value=30, value=30, step=1)

# Apply filters
f = df.copy()
if date_range is not None:
    start_d, end_d = date_range
    f = f[(f[col_game_date].dt.date >= start_d)
          & (f[col_game_date].dt.date <= end_d)]

if team_selected is not None and team_mode == "Home team":
    f = f[f[col_home_team] == team_selected]
elif team_selected is not None and team_mode == "Away team":
    f = f[f[col_away_team] == team_selected]

# -----------------------------
# Header
# -----------------------------
st.title("NBA Fatigue Dashboard")
st.caption("Data source: nba_api LeagueGameFinder (season 2023–24)")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric("Games", f"{len(f):,}")
with kpi2:
    st.metric("Home win rate",
              f"{(f[col_home_win].mean() * 100):.1f}%" if len(f) else "—")
with kpi3:
    st.metric("Avg fatigue diff (home − away)",
              f"{f[col_fatigue_diff].mean():.3f}" if len(f) else "—")
with kpi4:
    st.metric("Avg point diff (home − away)",
              f"{f[col_point_diff].mean():.2f}" if len(f) else "—")

st.divider()

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Win% vs Fatigue (Binned)",
    "Point Margin vs Fatigue (Scatter)",
    "Team Rankings (Fatigue)",
    "Data Table"
])

# -----------------------------
# Tab 1: Win% by fatigue difference (binned)
# -----------------------------
with tab1:
    if len(f) == 0:
        st.info("No games match your filters.")
    else:
        binned = compute_win_pct_by_bin(
            f, x_col=col_fatigue_diff, win_col=col_home_win, bin_size=float(
                bin_size)
        )

        # Plot
        fig = px.bar(
            binned,
            x="fatigue_bin",
            y="home_win_pct",
            hover_data={"games": True,
                        "home_win_pct": ":.3f", "fatigue_bin": True},
        )
        fig.update_layout(
            xaxis_title=f"Fatigue Diff (binned, size={bin_size})",
            yaxis_title="Home Win %",
            yaxis_tickformat=".0%",
            showlegend=False,
            height=480,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        # Reference line at overall win %
        overall = f[col_home_win].mean()
        fig.add_hline(y=overall, line_width=1, line_dash="dot")

        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Interpretation: Fatigue Diff = Home fatigue score − Away fatigue score. "
            "Negative values indicate the home team is less fatigued than the away team."
        )

# -----------------------------
# Tab 2: Scatter (point diff vs fatigue diff) + safe trendlines
# -----------------------------
with tab2:
    if len(f) == 0:
        st.info("No games match your filters.")
    else:
        # Keep only finite numeric rows
        f2 = f.copy()
        f2[col_fatigue_diff] = numeric_safe(f2[col_fatigue_diff])
        f2[col_point_diff] = numeric_safe(f2[col_point_diff])
        f2 = f2.replace(
            [np.inf, -np.inf], np.nan).dropna(subset=[col_fatigue_diff, col_point_diff])

        if len(f2) == 0:
            st.info("No valid numeric rows after cleaning.")
        else:
            fig = px.scatter(
                f2,
                x=col_fatigue_diff,
                y=col_point_diff,
                color="Fatigue Advantage",
                hover_data={
                    col_game_id: True if col_game_id else False,
                    col_game_date: True,
                    col_home_team: True,
                    col_away_team: True,
                    col_home_pts: True if col_home_pts else False,
                    col_away_pts: True if col_away_pts else False,
                    col_fatigue_diff: ":.3f",
                    col_point_diff: ":.1f",
                },
            )
            fig.update_layout(
                xaxis_title="Fatigue Diff (home − away)",
                yaxis_title="Point Diff (home − away)",
                height=560,
                margin=dict(l=10, r=10, t=40, b=10),
            )

            # Add safe trendlines per category (optional)
            add_trends = st.checkbox("Show trendlines", value=True)
            if add_trends:
                for cat in ["Home fatigue advantage", "Away fatigue advantage", "Even"]:
                    sub = f2[f2["Fatigue Advantage"] == cat]
                    if len(sub) >= 12 and np.unique(sub[col_fatigue_diff]).size >= 2:
                        fig = add_simple_trendline(
                            fig,
                            sub[col_fatigue_diff].to_numpy(),
                            sub[col_point_diff].to_numpy(),
                            name=f"Trend: {cat}",
                        )

            st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Tab 3: Team Rankings (Fatigue)
# -----------------------------
with tab3:
    if len(f) == 0:
        st.info("No games match your filters.")
    else:
        rank_mode = st.radio(
            "Ranking by",
            ["Away team fatigue", "Home team fatigue"],
            horizontal=True,
        )

        # If matchups CSV does not have per-side fatigue scores, we compute approximate per-side fatigue
        # using fatigue_diff + assumption about baseline is not valid, so we prefer the team_games file if present.
        # However, your matchups file (from your pipeline) typically includes home_fatigue_score & away_fatigue_score.
        col_home_fatigue = pick_col(
            df, ["home_fatigue_score", "Home Fatigue Score"])
        col_away_fatigue = pick_col(
            df, ["away_fatigue_score", "Away Fatigue Score"])

        if rank_mode == "Away team fatigue" and col_away_fatigue is not None:
            tmp = f.copy()
            tmp[col_away_fatigue] = numeric_safe(tmp[col_away_fatigue])
            tmp = tmp.dropna(subset=[col_away_fatigue])
            rank = (
                tmp.groupby(col_away_team, as_index=False)
                .agg(Avg_Fatigue=(col_away_fatigue, "mean"), Games=(col_away_team, "size"))
                .sort_values("Avg_Fatigue", ascending=False)
                .head(top_n)
            )
            fig = px.bar(
                rank.sort_values("Avg_Fatigue"),
                x="Avg_Fatigue",
                y=col_away_team,
                orientation="h",
                hover_data={"Games": True, "Avg_Fatigue": ":.3f"},
            )
            fig.update_layout(
                xaxis_title="Avg Away Fatigue Score",
                yaxis_title="Away Team",
                height=700,
                margin=dict(l=10, r=10, t=40, b=10),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        elif rank_mode == "Home team fatigue" and col_home_fatigue is not None:
            tmp = f.copy()
            tmp[col_home_fatigue] = numeric_safe(tmp[col_home_fatigue])
            tmp = tmp.dropna(subset=[col_home_fatigue])
            rank = (
                tmp.groupby(col_home_team, as_index=False)
                .agg(Avg_Fatigue=(col_home_fatigue, "mean"), Games=(col_home_team, "size"))
                .sort_values("Avg_Fatigue", ascending=False)
                .head(top_n)
            )
            fig = px.bar(
                rank.sort_values("Avg_Fatigue"),
                x="Avg_Fatigue",
                y=col_home_team,
                orientation="h",
                hover_data={"Games": True, "Avg_Fatigue": ":.3f"},
            )
            fig.update_layout(
                xaxis_title="Avg Home Fatigue Score",
                yaxis_title="Home Team",
                height=700,
                margin=dict(l=10, r=10, t=40, b=10),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            # Fallback: use team_games file if provided
            if team_games_path and Path(team_games_path).exists():
                tg = load_csv(team_games_path)
                tg_team = pick_col(tg, ["TEAM_NAME", "Team Name"])
                tg_date = pick_col(tg, ["GAME_DATE", "Game Date"])
                tg_fat = pick_col(tg, ["fatigue_score", "Fatigue Score"])

                if tg_team and tg_date and tg_fat:
                    tg[tg_date] = to_datetime_safe(tg[tg_date])
                    tg[tg_fat] = numeric_safe(tg[tg_fat])
                    tg = tg.dropna(subset=[tg_team, tg_date, tg_fat])
                    tg = tg[tg[tg_team].isin(NBA_TEAMS)]

                    # Apply same date + team filters
                    if date_range is not None:
                        start_d, end_d = date_range
                        tg = tg[(tg[tg_date].dt.date >= start_d)
                                & (tg[tg_date].dt.date <= end_d)]
                    if team_selected is not None:
                        tg = tg[tg[tg_team] == team_selected]

                    rank = (
                        tg.groupby(tg_team, as_index=False)
                          .agg(Avg_Fatigue=(tg_fat, "mean"), Games=(tg_team, "size"))
                          .sort_values("Avg_Fatigue", ascending=False)
                          .head(top_n)
                    )
                    fig = px.bar(
                        rank.sort_values("Avg_Fatigue"),
                        x="Avg_Fatigue",
                        y=tg_team,
                        orientation="h",
                        hover_data={"Games": True, "Avg_Fatigue": ":.3f"},
                    )
                    fig.update_layout(
                        xaxis_title="Avg Fatigue Score",
                        yaxis_title="Team",
                        height=700,
                        margin=dict(l=10, r=10, t=40, b=10),
                        showlegend=False,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(
                        "Rankings require either home/away fatigue score columns in matchups CSV "
                        "or a valid team_games file containing TEAM_NAME, GAME_DATE, fatigue_score."
                    )
            else:
                st.warning(
                    "Rankings require either home/away fatigue score columns in matchups CSV "
                    "or a valid team_games CSV path."
                )

# -----------------------------
# Tab 4: Data Table
# -----------------------------
with tab4:
    st.subheader("Filtered games (matchups)")
    show_cols = [c for c in [
        col_game_id, col_game_date, col_home_team, col_away_team,
        col_home_pts, col_away_pts, col_home_win,
        col_point_diff, col_rest_diff, col_travel_diff, col_fatigue_diff
    ] if c is not None]

    out = f[show_cols + ["Fatigue Advantage"]].copy()
    # Sort newest first if possible
    if col_game_date is not None:
        out = out.sort_values(col_game_date, ascending=False)

    st.dataframe(out, use_container_width=True, height=520)

st.caption(
    "Fatigue score definition (pipeline): fatigue_score = 2*(back-to-back flag) + travel_miles/1000. "
    "This is intentionally simple and explainable for a portfolio demo."
)
