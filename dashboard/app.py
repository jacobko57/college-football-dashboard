import streamlit as st
import pandas as pd

from model import forecast_team

from plots import (
    plot_top_n_highest_pct,
    plot_nil_spending_vs,
    plot_avg_spending_by_conf,
)

# ----------------------------------------------------
# Page Configuration
# ----------------------------------------------------

st.set_page_config(
    page_title="College Football & NIL Analytics",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------
# Load Data
# ----------------------------------------------------

DATA_DIR = "./final_datasets"

@st.cache_data
def load_data():

    team_stats = pd.read_csv(f"{DATA_DIR}/cfb_data.csv")
    team_spending = pd.read_csv(f"{DATA_DIR}/conference_spending.csv")

    if "Unnamed: 0" in team_stats.columns:
        team_stats.drop(columns=["Unnamed: 0"], inplace=True)

    return team_stats, team_spending

team_stats, team_spending = load_data()

# ----------------------------------------------------
# Sidebar
# ----------------------------------------------------

st.sidebar.title("Filters")

year_range = st.sidebar.slider(
    "Season",
    int(team_stats.year.min()),
    int(team_stats.year.max()),
    (
        int(team_stats.year.min()),
        int(team_stats.year.max())
    )
)

filtered_df = team_stats.copy()

filtered_df = filtered_df[
    (filtered_df.year >= year_range[0]) &
    (filtered_df.year <= year_range[1])
]

if "conference" in filtered_df.columns and conference:
    filtered_df = filtered_df[
        filtered_df.conference.isin(conference)
    ]

scatter_variable = st.sidebar.selectbox(
    "Compare NIL Spending Against",
    [
        c for c in filtered_df.columns
        if c not in ["team", "year", "conference"]
    ]
)

# ----------------------------------------------------
# Header
# ----------------------------------------------------

st.title("🏈 College Football & NIL Dashboard")

st.markdown("""
### Has NIL changed college football?

This dashboard investigates whether **Name, Image, and Likeness (NIL) spending**
is associated with football success and whether increased financial investment
is changing competitive balance across Division I football.

Move from left to right through the dashboard to explore the story.
""")

# ----------------------------------------------------
# Executive Summary
# ----------------------------------------------------

st.divider()

st.header("Executive Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Teams",
        filtered_df.team.nunique()
    )

with col2:

    st.metric(
        "Seasons",
        filtered_df.year.nunique()
    )

with col3:

    avg_win_pct = (
        filtered_df.win.sum()
        /
        (filtered_df.win.sum() + filtered_df.loss.sum())
    )

    st.metric(
        "Average Win %",
        f"{avg_win_pct:.1%}"
    )

with col4:

    st.metric(
        "Games",
        filtered_df.win.sum() + filtered_df.loss.sum()
    )

st.info("""
The dashboard is organized around three questions:

1. Where is NIL money concentrated?
2. Is spending associated with winning?
3. What does this mean for future performance?
""")

# ----------------------------------------------------
# Tabs
# ----------------------------------------------------

story_tab, team_tab, forecast_tab = st.tabs(
    [
        "📖 NIL Story",
        "🏈 Team Analysis",
        "🔮 Future Outlook"
    ]
)

########################################################
# STORY TAB
########################################################

with story_tab:

    st.header("1. NIL Spending Across Conferences")

    left, right = st.columns([2,1])

    with left:

        plot_avg_spending_by_conf()

    with right:

        st.success("""
**Key Question**

Which conferences possess the greatest financial advantage?

Large spending differences may indicate widening resource gaps between
Power conferences and the rest of Division I football.
""")

    st.divider()

    st.header("2. Does NIL Spending Translate to Success?")

    left, right = st.columns([3,1])

    with left:

        plot_nil_spending_vs(scatter_variable)

    with right:

        st.info(f"""
This visualization compares NIL spending with **{scatter_variable}**.

Look for:

• positive trend

• clusters

• outliers

Remember that correlation does not imply causation.
""")

    st.divider()

    st.header("3. Competitive Balance")

    left, right = st.columns([2,1])

    with left:

        default_index = sorted(team_spending.Conference.unique().tolist()).index("Big Ten")

        conference = st.selectbox(
            "Select Conference",
            sorted(team_spending.Conference.unique()),
            index=default_index
        )

        plot_top_n_highest_pct(conference=conference)

    with right:

        st.info("""
**Key Insight**

If a small number of conferences dominate the top-performing programs,
NIL resources may reinforce existing competitive advantages rather than
create new ones.
""")

    st.divider()

    with st.expander("View Underlying Data"):

        st.dataframe(
            filtered_df,
            use_container_width=True
        )

########################################################
# TEAM TAB
########################################################

with team_tab:

    st.header("Team Performance")

    team = st.selectbox(
        "Select Team",
        sorted(filtered_df.team.unique())
    )

    team_df = filtered_df[
        filtered_df.team == team
    ]

    st.subheader(team)

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Winning %",
            f"{team_df.win.sum() / (team_df.win.sum() + team_df.loss.sum()):.1%}"
        )

    with c2:

        st.metric(
            "Points/Game",
            f"{team_df.points_per_game.mean():.1f}"
        )

    with c3:

        st.metric(
            "Yards Allowed/Play",
            f"{team_df.yards_play_allowed.mean():.2f}"
        )

    with c4:

        st.metric(
            "Seasons",
            team_df.year.nunique()
        )

    st.divider()

    st.markdown("""
### Interpretation

These metrics summarize historical performance over the selected seasons.
Compare multiple teams using the sidebar filters to understand how sustained
success aligns with conference affiliation and NIL spending.
""")

########################################################
# FORECAST TAB
########################################################

with forecast_tab:

    st.header("Future Performance Projection")

    st.markdown("""
Historical team performance is used to forecast future outcomes.

These predictions are based on previous seasons and **do not account for**
future coaching changes, transfer portal activity, injuries, or future NIL
investments.
""")

    col1, col2 = st.columns([2,1])

    with col1:

        forecast_team_name = st.selectbox(
            "Team",
            sorted(filtered_df.team.unique()),
            key="forecast"
        )

        forecast_year = st.slider(
            "Forecast Through",
            2025,
            2035,
            2028
        )

        forecast_team(
            forecast_team_name,
            last_year=forecast_year
        )

    with col2:

        st.info("""
### How to Read the Forecast

The model projects expected future performance using historical trends.

Forecasts become increasingly uncertain farther into the future and should be
interpreted as projections rather than guaranteed outcomes.
""")