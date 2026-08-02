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