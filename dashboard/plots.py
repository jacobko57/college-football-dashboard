import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from pathlib import Path
from plotly.subplots import make_subplots
import plotly.graph_objects as go

BASE_DIR = Path(__file__).parent

team_conference = pd.read_csv(BASE_DIR / "final_datasets" / "team_conference.csv")
team_stats = pd.read_csv(BASE_DIR / "final_datasets" / "team_stats_clean.csv")
team_spending = pd.read_csv(BASE_DIR / "final_datasets" / "all_school_fees.csv")
conference_spending = pd.read_csv(BASE_DIR / "final_datasets" / "conference_spending.csv")

team_stats["winning_percentage"] = team_stats["win"] / (team_stats["win"] + team_stats["loss"])

# -------------------------------------------------------
# Global Visualization Theme
# -------------------------------------------------------

PRIMARY_COLOR = "#1f77b4"
ACCENT_COLOR = "#ff7f0e"

CONFERENCE_COLORS = {

    "SEC": "#E03A3E",          # Red
    "Big Ten": "#1D3557",      # Navy
    "ACC": "#2A9D8F",          # Teal
    "Big 12": "#F4A261",       # Orange
    "Pac-12": "#6A4C93",       # Purple
    "American": "#457B9D",     # Blue
    "Mountain West": "#8D99AE",
    "Sun Belt": "#FFB703",
    "MAC": "#90BE6D",
    "Conference USA": "#577590",
    "Independent": "#999999",
    "Unknown": "#CCCCCC"

}

# Plot top N teams with highest winning percentages + conference counts
def plot_top_n_highest_pct(
    conference="Big Ten",
    team_stats=team_stats,
    team_conference=team_conference,
    n=20
):

    """
    Displays:
    1. Top N teams by average winning percentage
    2. Conference representation among those teams

    Designed for storytelling:
    - Ranked performance
    - Conference concentration
    """
    # ------------------------------------------------
    # Calculate average winning percentage
    # ------------------------------------------------
    average_winning_percent = (
        team_stats
        .groupby("team")["winning_percentage"]
        .mean()
        .reset_index()
        .sort_values(
            "winning_percentage",
            ascending=False
        )
    )

    # Select top teams
    top_n_teams = (
        average_winning_percent
        .head(n)
        .copy()
    )

    # ------------------------------------------------
    # Add conference information
    # ------------------------------------------------
    top_n_teams = top_n_teams.merge(
        team_conference,
        left_on="team",
        right_on="Team",
        how="left"
    )

    top_n_teams["Conference"] = (
        top_n_teams["Conference"]
        .fillna("Unknown")
    )


    top_n_teams["color"] = (
        top_n_teams["Conference"]
        .where(top_n_teams["Conference"] == conference)
        .map(CONFERENCE_COLORS)
        .fillna("gray")
    )

    # ------------------------------------------------
    # Conference representation
    # ------------------------------------------------
    conference_counts = (
        top_n_teams["Conference"]
        .value_counts()
        .reset_index()
    )

    conference_counts.columns = [
        "Conference",
        "Teams"
    ]

    conference_counts["Color"] = (
        conference_counts["Conference"]
        .map(CONFERENCE_COLORS)
        .fillna("#CCCCCC")
    )

    conference_counts["Color"] = (
        conference_counts["Conference"]
        .where(conference_counts["Conference"] == conference)
        .map(CONFERENCE_COLORS)
        .fillna("gray")
    )

    conference_counts = (
        conference_counts
        .sort_values(
            "Teams",
            ascending=True
        )
    )

    # ------------------------------------------------
    # Create subplot layout
    # ------------------------------------------------
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.65,0.35],
        subplot_titles=(
            f"Top {n} Programs by Average Winning Percentage",
            "Conference Representation"
        )
    )

    # ------------------------------------------------
    # LEFT CHART
    # ------------------------------------------------
    # Highlight top 5
    top_n_teams["Highlight"] = [
        "Top 5"
        if i < 5
        else "Other"
        for i in range(len(top_n_teams))
    ]

    fig.add_trace(
        go.Bar(
            x=top_n_teams["winning_percentage"],
            y=top_n_teams["team"],
            orientation="h",
            marker_color=top_n_teams["color"],
            text=[
                f"{x:.1%}"
                for x in top_n_teams["winning_percentage"]
            ],
            textposition="outside",
            hovertemplate=
            "<b>%{y}</b><br>" +
            "Winning Percentage: %{x:.1%}" +
            "<extra></extra>"
        ),
        row=1,
        col=1
    )

    # ------------------------------------------------
    # RIGHT CHART
    # ------------------------------------------------
    fig.add_trace(
        go.Bar(
            x=conference_counts["Teams"],
            y=conference_counts["Conference"],
            orientation="h",
            marker_color=conference_counts["Color"],
            text=conference_counts["Teams"],
            textposition="outside",
            hovertemplate=
            "<b>%{y}</b><br>" +
            "Teams: %{x}" +
            "<extra></extra>"
        ),
        row=1,
        col=2
    )

    fig.update_yaxes(
        autorange="reversed",
        row=1,
        col=1
    )

    # ------------------------------------------------
    # Layout
    # ------------------------------------------------
    fig.update_layout(
        template="plotly_white",
        height=650,
        showlegend=False,
        title={
            "text":
            "Elite College Football Programs and Conference Dominance",
            "x":0.05
        },

        margin=dict(
            l=20,
            r=20,
            t=90,
            b=20
        )
    )

    # ------------------------------------------------
    # Axis formatting
    # ------------------------------------------------
    fig.update_xaxes(
        showticklabels=False,
        title="Average Winning Percentage",
        row=1,
        col=1
    )

    fig.update_xaxes(
        showticklabels=False,title="Number of Programs",
        row=1,
        col=2
    )

    # Remove unnecessary gridlines
    fig.update_xaxes(
        showgrid=False
    )

    fig.update_yaxes(
        showgrid=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# Plot scatter plot of NIL spending vs. chosen variable
def plot_nil_spending_vs(
    var,
    team_stats=team_stats,
    team_spending=team_spending
):

    """
    Displays relationship between NIL spending and
    a selected football performance metric.

    Parameters:
        var:
            Performance variable to compare against NIL

        team_stats:
            Football statistics dataframe

        team_spending:
            NIL spending dataframe
    """

    # ----------------------------------------
    # Use most recent season
    # ----------------------------------------
    latest_year = team_stats["year"].max()

    recent_stats = team_stats[
        team_stats["year"] == latest_year
    ].copy()

    # ----------------------------------------
    # Merge NIL + performance data
    # ----------------------------------------
    merged_df = pd.merge(
        team_spending,
        recent_stats,
        left_on="School",
        right_on="team",
        how="inner"
    )

    if var not in merged_df.columns:
        st.error(
            f"{var} is not available."
        )
        return

    # ----------------------------------------
    # Calculate winning percentage
    # ----------------------------------------
    merged_df["winning_percentage"] = (
        merged_df["win"]
        /
        (
            merged_df["win"]
            +
            merged_df["loss"]
        )
    )

    # ----------------------------------------
    # Plot
    # ----------------------------------------
    fig = px.scatter(
        merged_df,
        x="Available 2026 ($)",
        y=var,
        size="winning_percentage",
        color="Conference"
        if "Conference" in merged_df.columns
        else None,
        color_discrete_map=CONFERENCE_COLORS,
        hover_name="team",
        hover_data={
            "Available 2026 ($)": ":$,.0f",
            var: ":.2f",
            "winning_percentage": ":.1%"
        },
        size_max=35
    )

    # ----------------------------------------
    # Styling
    # ----------------------------------------
    fig.update_layout(
        template="plotly_white",
        height=550,
        title={
            "text":
            f"NIL Spending vs {var}<br>"
            f"<sup>{latest_year} season</sup>",
            "x":0.05,
            "xanchor":"left"
        },

        xaxis_title="NIL Spending",

        yaxis_title=var,

        legend_title="Conference",

        margin=dict(
            l=40,
            r=40,
            t=80,
            b=40
        )
    )

    # ----------------------------------------
    # Axis formatting
    # ----------------------------------------
    fig.update_xaxes(
        showticklabels=False
    )

    # Remove visual clutter
    fig.update_traces(
        marker=dict(
            opacity=0.75,
            line=dict(
                width=1,
                color="white"
            )
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# Plot Average Spending by Conference
def plot_avg_spending_by_conf(
    conference_spending=conference_spending,
    CONFERENCE_COLORS=CONFERENCE_COLORS
):

    """
    Shows conferences with above-average revenue.

    Story:
    Which conferences have a financial advantage?
    """
    df = conference_spending.copy()

    # ----------------------------------------
    # Clean data
    # ----------------------------------------
    df["Avr Revenue per School *"] = pd.to_numeric(
        df["Avr Revenue per School *"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["Avr Revenue per School *"]
    )

    # ----------------------------------------
    # Calculate average
    # ----------------------------------------
    national_average = (
        df["Avr Revenue per School *"]
        .mean()
    )

    # ----------------------------------------
    # Keep only above-average conferences
    # ----------------------------------------
    df = df[
        df["Avr Revenue per School *"] 
        > national_average
    ]

    # ----------------------------------------
    # Sort highest -> lowest
    # ----------------------------------------
    df = df.sort_values(
        "Avr Revenue per School *",
        ascending=False
    )

    # ----------------------------------------
    # Plot
    # ----------------------------------------
    fig = go.Figure()
    for _, row in df.iterrows():
        conference = row["Conference"]
        fig.add_trace(
            go.Bar(
                x=[
                    row["Avr Revenue per School *"]
                ],
                y=[
                    conference
                ],
                orientation="h",
                name=conference,
                marker_color=CONFERENCE_COLORS.get(
                    conference,
                    "#CCCCCC"
                ),
                text=[
                    f"${row['Avr Revenue per School *']/1_000_000:.1f}M"
                ],
                textposition="outside",
                hovertemplate=
                "<b>%{y}</b><br>" +
                "Average Revenue: %{text}" +
                "<extra></extra>"
            )
        )

    # ----------------------------------------
    # Average line
    # ----------------------------------------
    fig.add_vline(
        x=national_average,
        line_dash="dash",
        line_color="gray",
        annotation_text="National Average",
        annotation_position="top"
    )

    # ----------------------------------------
    # Layout
    # ----------------------------------------
    fig.update_layout(
        template="plotly_white",
        height=500,
        title={
            "text":
            "Conferences Operating Above Average Financial Levels<br>"
            "<sup>Only conferences exceeding the national average are shown</sup>",
            "x":0.05
        },

        xaxis_title="Average Revenue Per School",
        yaxis_title="",
        showlegend=True,
        legend_title="Conference",
        margin=dict(
            l=20,
            r=100,
            t=90,
            b=30
        )
    )

    # ----------------------------------------
    # Force highest at top
    # ----------------------------------------
    fig.update_yaxes(
        autorange="reversed"
    )

    fig.update_xaxes(
        showticklabels=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# Plot scatter plot of Big Ten winning percentage vs. spending
def plot_big_ten_spending_vs_wins(
    team_stats=team_stats,
    team_spending=team_spending,
):
    """
    Displays the relationship between NIL spending and
    winning percentage for Big Ten football teams.
    """

    # ----------------------------------------
    # Most recent season
    # ----------------------------------------
    latest_year = team_stats["year"].max()

    recent_stats = (
        team_stats[team_stats["year"] == latest_year]
        .copy()
    )

    # ----------------------------------------
    # Merge data
    # ----------------------------------------
    merged_df = pd.merge(
        team_spending,
        recent_stats,
        left_on="School",
        right_on="team",
        how="inner"
    )

    # ----------------------------------------
    # Winning %
    # ----------------------------------------
    merged_df["winning_percentage"] = (
        merged_df["win"] /
        (merged_df["win"] + merged_df["loss"])
    )

    # ----------------------------------------
    # Keep only Big Ten
    # ----------------------------------------
    merged_df = merged_df[
        merged_df["Conference"] == "Big Ten"
    ].copy()

    # ----------------------------------------
    # Correlation
    # ----------------------------------------
    corr = merged_df["Available 2026 ($)"].corr(
        merged_df["winning_percentage"]
    )

    # ----------------------------------------
    # Scatter plot
    # ----------------------------------------
    fig = px.scatter(
        merged_df,
        x="Available 2026 ($)",
        y="winning_percentage",
        trendline="ols",
        color_discrete_sequence=["#1f77b4"],
        hover_name="team",
        hover_data={
            "Available 2026 ($)": ":$,.0f",
            "winning_percentage": ":.1%"
        }
    )

    # ----------------------------------------
    # Style
    # ----------------------------------------
    fig.update_traces(
        textposition="top center",
        marker=dict(
            size=12,
            opacity=0.8,
            line=dict(
                width=1,
                color="white"
            )
        )
    )

    fig.update_layout(
        template="plotly_white",
        height=600,
        title=(
            f"Big Ten NIL Spending vs Winning Percentage"
            f"<br><sup>Correlation = {corr:.2f}</sup>"
        ),
        xaxis_title="NIL Spending",
        yaxis_title="Winning Percentage",
        showlegend=False,
        margin=dict(
            l=40,
            r=40,
            t=80,
            b=40
        )
    )

    fig.update_xaxes(
        showgrid=False,
        showticklabels=False
    )

    fig.update_yaxes(
        showgrid=False,
        showticklabels=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# Plot spending resources for all teams
def plot_all_schools_awards_2025_vs_2026(
    team_spending=team_spending,
):
    """
    Compares each school's total 2025 awards with its
    available NIL resources for 2026.

    A log scale is used to make differences between schools
    with very different resource levels easier to see.
    """

    # ----------------------------------------
    # Prepare data
    # ----------------------------------------
    plot_df = team_spending[
        [
            "School",
            "Conference",
            "Total Value ($) 2025 Awards",
            "Available 2026 ($)",
            "% Increase",
        ]
    ].dropna().copy()

    # Remove schools with zero/negative values because
    # log scales cannot display them.
    plot_df = plot_df[
        (plot_df["Total Value ($) 2025 Awards"] > 0)
        & (plot_df["Available 2026 ($)"] > 0)
    ].copy()

    # ----------------------------------------
    # Scatter plot
    # ----------------------------------------
    fig = px.scatter(
        plot_df,
        x="Total Value ($) 2025 Awards",
        y="Available 2026 ($)",
        hover_name="School",
        hover_data={
            "Conference": True,
            "Total Value ($) 2025 Awards": ":$,.0f",
            "Available 2026 ($)": ":$,.0f",
            "% Increase": ":.0f%",
        },
        color_discrete_sequence=["#1f77b4"],
    )

    # ----------------------------------------
    # Style
    # ----------------------------------------
    fig.update_traces(
        marker=dict(
            size=11,
            opacity=0.75,
            line=dict(
                width=1,
                color="white",
            ),
        ),
        selector=dict(mode="markers"),
    )

    fig.update_layout(
        template="plotly_white",
        height=650,
        title=(
            "School NIL Resources: 2025 vs 2026"
            "<br><sup>"
            "Schools farther toward the top-right have substantially more resources"
            "</sup>"
        ),
        xaxis_title="Total Value of 2025 Awards",
        yaxis_title="Available Resources in 2026",
        showlegend=False,
        margin=dict(
            l=50,
            r=40,
            t=90,
            b=50,
        ),
    )

    # ----------------------------------------
    # Log scales
    # ----------------------------------------
    fig.update_xaxes(
        type="log",
        showgrid=False,
        showticklabels=False,
    )

    fig.update_yaxes(
        type="log",
        showgrid=False,
        showticklabels=False,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

# Plot winning percentages of all teams
def plot_best_vs_worst_programs(
    team_stats=team_stats,
    n=10
):
    """
    Displays the highest- and lowest-performing college football
    programs based on average winning percentage.

    Designed to emphasize the gap between elite and struggling
    programs across the entire dataset.
    """

    # ------------------------------------------------
    # Calculate average winning percentage
    # ------------------------------------------------
    average_winning_percent = (
        team_stats
        .groupby("team")["winning_percentage"]
        .mean()
        .reset_index()
        .sort_values(
            "winning_percentage",
            ascending=False
        )
    )

    # ------------------------------------------------
    # Select top and bottom programs
    # ------------------------------------------------
    top_teams = (
        average_winning_percent
        .head(n)
        .sort_values(
            "winning_percentage",
            ascending=False
        )
        .copy()
    )

    bottom_teams = (
        average_winning_percent
        .tail(n)
        .sort_values(
            "winning_percentage",
            ascending=True
        )
        .copy()
    )

    # ------------------------------------------------
    # Identify absolute highest / lowest
    # ------------------------------------------------
    highest_team = average_winning_percent.iloc[0]
    lowest_team = average_winning_percent.iloc[-1]

    # ------------------------------------------------
    # Calculate gap
    # ------------------------------------------------
    gap = (
        highest_team["winning_percentage"]
        - lowest_team["winning_percentage"]
    )

    # ------------------------------------------------
    # Colors
    # ------------------------------------------------
    top_colors = [
        "#2ca02c"
        if team == highest_team["team"]
        else "#B0B0B0"
        for team in top_teams["team"]
    ]

    bottom_colors = [
        "#d62728"
        if team == lowest_team["team"]
        else "#B0B0B0"
        for team in bottom_teams["team"]
    ]

    # ------------------------------------------------
    # Labels
    # Only highlight absolute highest / lowest
    # ------------------------------------------------
    top_text = [
        (
            f"<b>{team}</b><br>{pct:.1%}"
            if team == highest_team["team"]
            else f"<b>{team}</b>"
        )
        for team, pct in zip(
            top_teams["team"],
            top_teams["winning_percentage"]
        )
    ]

    bottom_text = [
        (
            f"<b>{team}</b><br>{pct:.1%}"
            if team == lowest_team["team"]
            else f"<b>{team}</b>"
        )
        for team, pct in zip(
            bottom_teams["team"],
            bottom_teams["winning_percentage"]
        )
    ]

    # ------------------------------------------------
    # Create subplot layout
    # ------------------------------------------------
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.5, 0.5],
        horizontal_spacing=0.12,
        subplot_titles=(
            f"Top {n} Programs",
            f"Bottom {n} Programs"
        )
    )

    # ------------------------------------------------
    # TOP PROGRAMS
    # Highest → Lowest
    # ------------------------------------------------
    fig.add_trace(
        go.Bar(
            x=top_teams["winning_percentage"],
            y=top_teams["team"],
            orientation="h",
            marker_color=top_colors,
            text=top_text,
            textposition="outside",
            cliponaxis=False,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Average Winning Percentage: %{x:.1%}"
                "<extra></extra>"
            )
        ),
        row=1,
        col=1
    )

    # ------------------------------------------------
    # BOTTOM PROGRAMS
    # Lowest → Highest
    # ------------------------------------------------
    fig.add_trace(
        go.Bar(
            x=bottom_teams["winning_percentage"],
            y=bottom_teams["team"],
            orientation="h",
            marker_color=bottom_colors,
            text=bottom_text,
            textposition="outside",
            cliponaxis=False,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Average Winning Percentage: %{x:.1%}"
                "<extra></extra>"
            )
        ),
        row=1,
        col=2
    )

    # ------------------------------------------------
    # Layout
    # ------------------------------------------------
    fig.update_layout(
        template="plotly_white",
        height=650,
        showlegend=False,

        title={
            "text": (
                "The Gap Between College Football's Best and Worst"
                f"<br><sup>"
                f"{highest_team['team']} leads at "
                f"{highest_team['winning_percentage']:.1%}"
                f" &nbsp;&nbsp;|&nbsp;&nbsp; "
                f"{lowest_team['team']} trails at "
                f"{lowest_team['winning_percentage']:.1%}"
                f" &nbsp;&nbsp;|&nbsp;&nbsp; "
                f"Gap: {gap:.1%} points"
                f"</sup>"
            ),
            "x": 0.05
        },

        margin=dict(
            l=20,
            r=40,
            t=110,
            b=30
        )
    )

    # ------------------------------------------------
    # Y-axis ordering
    # ------------------------------------------------
    # Top: highest → lowest
    fig.update_yaxes(
        autorange="reversed",
        showgrid=False,
        showticklabels=False,
        row=1,
        col=1
    )

    # Bottom: lowest → highest
    fig.update_yaxes(
        autorange="reversed",
        showgrid=False,
        showticklabels=False,
        row=1,
        col=2
    )

    # ------------------------------------------------
    # X-axis formatting
    # ------------------------------------------------
    fig.update_xaxes(
        range=[0, 1],
        showticklabels=False,
        showgrid=False,
        row=1,
        col=1
    )

    fig.update_xaxes(
        range=[0, 1],
        showticklabels=False,
        showgrid=False,
        row=1,
        col=2
    )

    # ------------------------------------------------
    # Plot
    # ------------------------------------------------
    st.plotly_chart(
        fig,
        use_container_width=True
    )
