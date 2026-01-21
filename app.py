# app.py
# NBA Fatigue Dashboard (portfolio-ready, minimalist, WHITE UI)
# Fixes:
# - Forces ALL fonts to dark (Streamlit + Plotly) so nothing “disappears”
# - Sidebar widgets are light with dark text (no black boxes)
# - Restores KPIs visibility
# - No Data paths exposed, no Data Table tab, no page icon
# - Team rankings from team_games CSV, NBA-only filter
# - Robust trendline (won't crash)

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# =============================
# Page config (no icon, wide)
# =============================
st.set_page_config(
    page_title="NBA Fatigue Dashboard",
    layout="wide",
)


# =============================
# White, professional UI CSS
# IMPORTANT: Do NOT set global "* { color: ... }" because it breaks Streamlit widgets/KPIs.
# We set targeted styles only.
# =============================
st.markdown(
    """
<style>
/* ---- App background ---- */
.stApp {
  background: #ffffff;
}

/* ---- Main text defaults ---- */
h1, h2, h3 { color: #111827 !important; font-weight: 700; }
p, span, label, div { color: #111827; }

/* ---- Sidebar ---- */
section[data-testid="stSidebar"]{
  background: #ffffff;
  border-right: 1px solid #e5e7eb;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div {
  color: #111827 !important;
}

/* ---- Inputs: make them light (not black boxes) ---- */
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea {
  background: #ffffff !important;
  color: #111827 !important;
  border: 1px solid #e5e7eb !important;
  border-radius: 10px !important;
}

section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
  background: #ffffff !important;
  border: 1px solid #e5e7eb !important;
  border-radius: 10px !important;
}
section[data-testid="stSidebar"] div[data-baseweb="select"] span {
  color: #111827 !important;
}

/* Date input container */
section[data-testid="stSidebar"] div[data-testid="stDateInput"] > div {
  background: #ffffff !important;
  border-radius: 10px !important;
}
section[data-testid="stSidebar"] div[data-testid="stDateInput"] input {
  color: #111827 !important;
}

/* Slider label/value */
section[data-testid="stSidebar"] div[data-testid="stSlider"] * {
  color: #111827 !important;
}

/* ---- Tabs ---- */
button[data-baseweb="tab"]{
  background: #ffffff !important;
  color: #374151 !important;
  border-bottom: 2px solid transparent !important;
}
button[data-baseweb="tab"][aria-selected="true"]{
  color: #111827 !important;
  border-bottom: 2px solid #2563eb !important;
}

/* ---- Metrics (KPIs) ---- */
div[data-testid="stMetric"] * {
  color: #111827 !important;
}

/* ---- Remove Streamlit chrome ---- */
header, footer { visibility: hidden; }
</style>
""",
    unsafe_allow_html=True,
)


# =============================
# Palette
# =============================
PLOT_TEMPLATE = "plotly_white"
TEXT_DARK = "#111827"
GRID_COLOR = "rgba(0,0,0,0.08)"

COLOR_HOME_ADV = "#2563EB"   # blue
COLOR_AWAY_ADV = "#F59E0B"   # orange
COLOR_NEUTRAL = "#9CA3AF"    # gray
COLOR_BAR = "#3B82F6"        # bar blue


# =============================
# Repo data files (hidden)
# =============================
MATCHUPS_PATH = "data_processed/nba_2324_fatigue_matchups.csv"
TEAM_GAMES_PATH = "data_processed/team_games_2324_with_fatigue.csv"

NBA_TEAMS = {
    "Atlanta Hawks", "Boston Celtics", "Brooklyn Nets", "Charlotte Hornets", "Chicago Bulls",
    "Cleveland Cavaliers", "Dallas Mavericks", "Denver Nuggets", "Detroit Pistons",
    "Golden State Warriors", "Houston Rockets", "Indiana Pacers", "LA Clippers", "Los Angeles Lakers",
    "Memphis Grizzlies", "Miami Heat", "Milwaukee Bucks", "Minnesota Timberwolves",
    "New Orleans Pelicans", "New York Knicks", "Oklahoma City Thunder", "Orlando Magic",
    "Philadelphia 76ers", "Phoenix Suns", "Portland Trail Blazers", "Sacramento Kings",
    "San Antonio Spurs", "Toronto Raptors", "Utah Jazz", "Washington Wizards"
}


# =============================
# Helpers
# =============================
def resolve_path(p: str) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return (Path(__file__).parent / path).resolve()


def load_csv(path: str) -> pd.DataFrame:
    p = resolve_path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    return pd.read_csv(p)


def pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def coerce_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def keep_nba_teams(df: pd.DataFrame, team_col: Optional[str]) -> pd.DataFrame:
    if not team_col or team_col not in df.columns:
        return df
    out = df.copy()
    out[team_col] = out[team_col].astype(str).str.strip()
    return out[out[team_col].isin(NBA_TEAMS)].copy()


def safe_trendline_xy(x: pd.Series, y: pd.Series) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    x = safe_numeric(x)
    y = safe_numeric(y)
    mask = np.isfinite(x.values) & np.isfinite(y.values)
    x2, y2 = x[mask], y[mask]
    if len(x2) < 2 or x2.nunique() < 2:
        return None
    try:
        m, b = np.polyfit(x2.values, y2.values, 1)
    except Exception:
        return None
    x_line = np.array([x2.min(), x2.max()])
    y_line = m * x_line + b
    return x_line, y_line


def apply_plotly_white(fig: go.Figure, title: str, x: str, y: str) -> go.Figure:
    """Hard-force dark axis titles/ticks so Streamlit theme can’t turn them white."""
    fig.update_layout(
        template=PLOT_TEMPLATE,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color=TEXT_DARK),
        title=dict(text=title, x=0, font=dict(color=TEXT_DARK, size=20)),
        margin=dict(l=20, r=20, t=70, b=60),
        legend=dict(
            title=None,
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(color=TEXT_DARK),
        ),
    )
    fig.update_xaxes(
        title=dict(text=x, font=dict(color=TEXT_DARK)),
        tickfont=dict(color=TEXT_DARK),
        showgrid=True,
        gridcolor=GRID_COLOR,
        zeroline=False,
    )
    fig.update_yaxes(
        title=dict(text=y, font=dict(color=TEXT_DARK)),
        tickfont=dict(color=TEXT_DARK),
        showgrid=True,
        gridcolor=GRID_COLOR,
        zeroline=False,
    )
    fig.update_traces(hoverlabel=dict(bgcolor="white", font_color=TEXT_DARK))
    return fig


# =============================
# Load data (cached)
# =============================
@st.cache_data(show_spinner=False)
def get_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_csv(MATCHUPS_PATH), load_csv(TEAM_GAMES_PATH)


try:
    df_match, df_team = get_data()
except Exception as e:
    st.error(f"App could not start: {e}")
    st.stop()


# =============================
# Detect columns (matchups)
# =============================
col_date = pick_col(df_match, ["Game Date", "game_date", "date"])
col_home_team = pick_col(df_match, ["Home Team", "home_team", "HOME_TEAM"])
col_away_team = pick_col(df_match, ["Away Team", "away_team", "AWAY_TEAM"])
col_home_pts = pick_col(
    df_match, ["Home Pts", "home_pts", "HOME_PTS", "home_points"])
col_away_pts = pick_col(
    df_match, ["Away Pts", "away_pts", "AWAY_PTS", "away_points"])
col_point_diff = pick_col(
    df_match, ["Point Diff", "point_diff", "point_margin", "margin"])
col_fatigue_diff = pick_col(
    df_match, ["Fatigue Diff", "fatigue_diff", "Fatigue Difference"])
col_home_win = pick_col(
    df_match, ["Home Win", "home_win", "home_wl", "HOME_WIN", "Home WL"])

df = df_match.copy()
if col_date:
    df[col_date] = coerce_datetime(df[col_date])

if col_point_diff is None and col_home_pts and col_away_pts:
    df["Point Diff (Home-Away)"] = safe_numeric(df[col_home_pts]
                                                ) - safe_numeric(df[col_away_pts])
    col_point_diff = "Point Diff (Home-Away)"

if col_fatigue_diff:
    df[col_fatigue_diff] = safe_numeric(df[col_fatigue_diff])

if col_home_win:
    df[col_home_win] = (safe_numeric(df[col_home_win]) > 0).astype(int)


# =============================
# Detect columns (team_games)
# =============================
col_team_team = pick_col(df_team, ["TEAM_NAME", "team", "team_name", "Team"])
col_team_date = pick_col(df_team, ["GAME_DATE", "game_date", "date"])
col_team_fatigue = pick_col(
    df_team, ["fatigue_score", "Fatigue Score", "FATIGUE_SCORE"])


# =============================
# Header
# =============================
st.title("NBA Fatigue Dashboard")
st.caption("Exploring how rest and travel relate to game outcomes (2023–24).")


# =============================
# Sidebar filters
# =============================
with st.sidebar:
    st.subheader("Filters")

    date_range = None
    if col_date and df[col_date].notna().any():
        dmin = df[col_date].min().date()
        dmax = df[col_date].max().date()
        date_range = st.date_input("Game date range", value=(dmin, dmax))

    team_filter = "All games"
    if col_home_team and col_away_team:
        teams = sorted(
            set(
                df[col_home_team].dropna().astype(str).str.strip().tolist()
                + df[col_away_team].dropna().astype(str).str.strip().tolist()
            )
        )
        teams = [t for t in teams if t in NBA_TEAMS] or teams
        team_filter = st.selectbox(
            "Team filter", ["All games"] + teams, index=0)

    bin_size = st.selectbox("Fatigue bin size", [0.25, 0.5, 1.0], index=1)
    top_n = st.slider("Top N teams (rankings)", min_value=10,
                      max_value=30, value=30, step=1)


# =============================
# Apply filters
# =============================
fdf = df.copy()

if date_range and col_date:
    start, end = date_range
    start = pd.to_datetime(start)
    end = pd.to_datetime(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    fdf = fdf[(fdf[col_date] >= start) & (fdf[col_date] <= end)].copy()

if team_filter != "All games" and col_home_team and col_away_team:
    fdf = fdf[(fdf[col_home_team] == team_filter) | (
        fdf[col_away_team] == team_filter)].copy()


# =============================
# KPIs (visible + dark)
# =============================
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Games", f"{len(fdf):,}")
with c2:
    st.metric(
        "Avg fatigue diff (home–away)",
        f"{fdf[col_fatigue_diff].mean():.2f}" if col_fatigue_diff and fdf[col_fatigue_diff].notna(
        ).any() else "—",
    )
with c3:
    st.metric(
        "Avg point diff (home–away)",
        f"{fdf[col_point_diff].mean():.2f}" if col_point_diff and fdf[col_point_diff].notna(
        ).any() else "—",
    )

st.divider()


# =============================
# Tabs (NO Data Table)
# =============================
tab1, tab2, tab3 = st.tabs(
    ["Win% vs Fatigue (Binned)", "Point Margin vs Fatigue (Scatter)",
     "Team Rankings (Fatigue)"]
)


# =============================
# Tab 1: Win% by fatigue bins
# =============================
with tab1:
    st.subheader("Win% by Fatigue Differential (Binned)")

    if col_fatigue_diff is None or col_home_win is None:
        st.warning(
            "This chart requires fatigue_diff and a home-win indicator in the matchups CSV.")
    else:
        temp = fdf[[col_fatigue_diff, col_home_win]].dropna().copy()
        b = float(bin_size)
        temp["Fatigue Bin"] = (
            np.floor(temp[col_fatigue_diff] / b) * b).round(2)

        agg = (
            temp.groupby("Fatigue Bin", as_index=False)
            .agg(WinPct=(col_home_win, "mean"), Games=(col_home_win, "size"))
            .sort_values("Fatigue Bin")
        )

        fig = px.bar(
            agg,
            x="Fatigue Bin",
            y="WinPct",
            text=agg["WinPct"].map(lambda v: f"{v:.0%}"),
            hover_data={"Games": True, "Fatigue Bin": True, "WinPct": ":.2%"},
        )
        fig.update_traces(marker_color=COLOR_BAR,
                          textposition="outside", cliponaxis=False)
        fig.update_yaxes(tickformat=".0%")

        avg = float(temp[col_home_win].mean()) if len(temp) else np.nan
        if np.isfinite(avg):
            fig.add_hline(y=avg, line_width=1, line_dash="dot",
                          line_color="rgba(17,24,39,0.35)")

        fig = apply_plotly_white(
            fig, "Home Win% by Fatigue Differential", "Fatigue diff (binned)", "Home win%")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Negative fatigue differential means the home team is less fatigued than the away team.")


# =============================
# Tab 2: Scatter with differentiation + trend
# =============================
with tab2:
    st.subheader("Point Margin vs Fatigue Differential")

    if col_fatigue_diff is None or col_point_diff is None:
        st.warning(
            "This chart requires fatigue_diff and point differential in the matchups CSV.")
    else:
        temp = fdf[[col_fatigue_diff, col_point_diff]].dropna().copy()

        def cat(v: float) -> str:
            if v < 0:
                return "Home advantage"
            if v > 0:
                return "Away advantage"
            return "Even"

        temp["Advantage"] = temp[col_fatigue_diff].apply(cat)

        fig = px.scatter(
            temp,
            x=col_fatigue_diff,
            y=col_point_diff,
            color="Advantage",
            opacity=0.75,
            color_discrete_map={
                "Home advantage": COLOR_HOME_ADV,
                "Away advantage": COLOR_AWAY_ADV,
                "Even": COLOR_NEUTRAL,
            },
        )
        fig.update_traces(marker=dict(size=7))

        tl = safe_trendline_xy(temp[col_fatigue_diff], temp[col_point_diff])
        if tl is not None:
            x_line, y_line = tl
            fig.add_trace(
                go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode="lines",
                    line=dict(color="rgba(17,24,39,0.55)", width=2),
                    name="Trend",
                )
            )

        fig = apply_plotly_white(fig, "Point Margin vs Fatigue Differential",
                                 "Fatigue diff (home–away)", "Point diff (home–away)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Positive point differential means the home team won by that margin.")


# =============================
# Tab 3: Team Rankings (team_games CSV, NBA only)
# =============================
with tab3:
    st.subheader("Team Rankings: Most Fatigued Situations")

    if col_team_team is None or col_team_fatigue is None:
        st.warning(
            "Team rankings require TEAM_NAME and fatigue_score columns in team_games CSV.")
    else:
        tdf = df_team.copy()

        if col_team_date:
            tdf[col_team_date] = coerce_datetime(tdf[col_team_date])
            if date_range:
                start, end = date_range
                start = pd.to_datetime(start)
                end = pd.to_datetime(
                    end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                tdf = tdf[(tdf[col_team_date] >= start) &
                          (tdf[col_team_date] <= end)].copy()

        tdf = keep_nba_teams(tdf, col_team_team)
        tdf[col_team_fatigue] = safe_numeric(tdf[col_team_fatigue])
        tdf = tdf[tdf[col_team_fatigue].notna()].copy()

        if tdf.empty:
            st.warning(
                "No NBA teams remain after filtering. Verify team names in team_games CSV.")
        else:
            rank = (
                tdf.groupby(col_team_team, as_index=False)
                .agg(Avg_Fatigue=(col_team_fatigue, "mean"), Games=(col_team_fatigue, "size"))
                .sort_values("Avg_Fatigue", ascending=False)
                .head(int(top_n))
            )

            rank_plot = rank.sort_values("Avg_Fatigue", ascending=True)

            fig = px.bar(
                rank_plot,
                x="Avg_Fatigue",
                y=col_team_team,
                orientation="h",
                text=rank_plot["Avg_Fatigue"].map(lambda v: f"{v:.2f}"),
                hover_data={"Games": True, "Avg_Fatigue": ":.2f"},
            )
            fig.update_traces(marker_color=COLOR_BAR,
                              textposition="outside", cliponaxis=False)
            fig = apply_plotly_white(
                fig, f"Top {int(top_n)} — Avg Fatigue Score", "Avg fatigue score", "Team")

            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Ranking source: team_games CSV. Filtered to official NBA team names.")
            st.caption(
                "Fatigue score definition (pipeline): fatigue_score = 2*(back-to-back flag) + travel_miles/1000.")
