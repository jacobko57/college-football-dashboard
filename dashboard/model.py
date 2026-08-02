
from turtle import st

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import plotly.graph_objects as go

encoder = joblib.load("../label_encoder.pkl")

team_stats_clean = pd.read_csv('../final_datasets/team_stats_clean.csv')

feature_columns = [
    c for c in team_stats_clean.columns
    if c not in ["team_id", "team", "year"]
]

NUM_FEATURES = 40
MAX_SEQUENCE_LENGTH = 3
NUM_TEAMS = len(encoder.classes_)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

class FootballLSTM(nn.Module):

    def __init__(
        self,
        input_size,
        num_teams,
        embedding_dim=8,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        output_size=1
    ):
        super().__init__()

        self.team_embedding = nn.Embedding(
            num_teams,
            embedding_dim
        )


        self.lstm = nn.LSTM(
            input_size=input_size + embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )


        self.fc = nn.Sequential(
            nn.Linear(hidden_size,64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64,1),
            nn.Sigmoid()
        )


    def forward(self, x, team_id):

        # team_id shape: (batch,)
        embedding = self.team_embedding(team_id)


        # repeat embedding across sequence length
        embedding = embedding.unsqueeze(1).repeat(
            1,
            x.size(1),
            1
        )


        # concatenate team identity + statistics
        x = torch.cat(
            [
                x,
                embedding
            ],
            dim=2
        )


        _, (hidden, _) = self.lstm(x)


        return self.fc(hidden[-1])

model = FootballLSTM(
        input_size=NUM_FEATURES,
        num_teams=NUM_TEAMS,
        hidden_size=32,
        num_layers=1,
        dropout=0.0,
        output_size=1
    ).to(device)

model.load_state_dict(
    torch.load("../best_football_lstm.pt", map_location=device)
)

model.to(device)


def add_future_year(team, year, df, noise_factor=0.5):

    recent = (
        df[df['team'] == team]
        .sort_values('year')
        .tail(3)
    )

    stat_columns = recent.drop(columns=['team','year']).columns

    future = {
        'team': team,
        'year': year
    }


    for col in stat_columns:

        mean = recent[col].mean()

        # Estimate natural variation
        std = recent[col].std()

        # If there is not enough history
        if np.isnan(std):
            std = 0


        # Add random variation
        future[col] = np.random.normal(
            loc=mean,
            scale=std * noise_factor
        )


    return pd.DataFrame([future])

def engineer_features(df):

    df = df.sort_values(
        ['team','year']
    )

    df['avg_winning_percentage_last_3_years'] = (
        df.groupby('team')['winning_percentage']
        .transform(
            lambda x:
            x.shift(1)
             .rolling(3, min_periods=1)
             .mean()
        )
    )


    df['change_in_winning_percentage_last_3_years'] = (
        df.groupby('team')['winning_percentage']
        .transform(
            lambda x:
            x.shift(1).diff(3)
        )
    )


    return df

def create_prediction_sequence(
    df,
    team_id,
    seq_length=MAX_SEQUENCE_LENGTH,
    feature_columns=None
):

    team_data = (
        df[df["team_id"] == team_id]
        .sort_values("year")
    )


    if len(team_data) < seq_length:
        raise ValueError(
            "Not enough seasons to create sequence"
        )


    # print(f"Team Stats\n{team_data[feature_columns]}")

    # Use the exact same features as training
    X = team_data[
        feature_columns
    ].values[-seq_length:]

    # print(f"Sequences:\n{X}")

    X = np.array(
        X,
        dtype=np.float32
    )


    # Add batch dimension
    X = np.expand_dims(
        X,
        axis=0
    )


    return X


def plot_winning_percentage(
    df,
    team_name,
    forecast_start_year=2024
):
    """
    Historical vs forecast winning percentage.

    Story:
    How has this program performed historically,
    and what trajectory does the model expect?
    """

    # ----------------------------------------
    # Split historical and forecast
    # ----------------------------------------
    historical = df[
        df["year"] < forecast_start_year
    ]

    forecast = df[
        df["year"] >= forecast_start_year
    ]

    # ----------------------------------------
    # Calculate trend direction
    # ----------------------------------------
    if len(forecast) > 1:
        change = (
            forecast.winning_percentage.iloc[-1]
            -
            forecast.winning_percentage.iloc[0]
        )
    else:
        change = 0

    if change > 0.05:
        trend_message = "📈 Improving"
    elif change < -0.05:
        trend_message = "📉 Declining"
    else:
        trend_message = "➡️ Stable"

    # ----------------------------------------
    # Create figure
    # ----------------------------------------
    fig = go.Figure()

    # ----------------------------------------
    # Historical performance
    # ----------------------------------------
    fig.add_trace(
        go.Scatter(
            x=historical["year"],
            y=historical["winning_percentage"],
            mode="lines+markers",
            name="Historical",
            line=dict(
                color="#1D3557",
                width=3
            ),
            marker=dict(
                size=8
            ),
            hovertemplate=
            "<b>%{x}</b><br>" +
            "Winning %: %{y:.1%}" +
            "<extra></extra>"
        )
    )

    # ----------------------------------------
    # Forecast
    # ----------------------------------------
    fig.add_trace(
        go.Scatter(
            x=forecast["year"],
            y=forecast["winning_percentage"],
            mode="lines+markers",
            name="Forecast",
            line=dict(
                color="#E76F51",
                width=3,
                dash="dash"
            ),
            marker=dict(
                size=9
            ),
            hovertemplate=
            "<b>%{x}</b><br>" +
            "Predicted Winning %: %{y:.1%}" +
            "<extra></extra>"
        )
    )

    # ----------------------------------------
    # Forecast divider
    # ----------------------------------------
    fig.add_vline(
        x=forecast_start_year,
        line_dash="dot",
        line_color="gray",
        annotation_text="Forecast Begins",
        annotation_position="top"
    )

    # ----------------------------------------
    # Add reference line
    # ----------------------------------------
    fig.add_hline(
        y=0.5,
        line_dash="dash",
        line_color="gray",
        annotation_text="50% Winning Rate",
        annotation_position="right"
    )

    # ----------------------------------------
    # Layout
    # ----------------------------------------
    fig.update_layout(
        template="plotly_white",
        height=550,
        title={
            "text":
            f"{team_name}: Winning Percentage Outlook<br>"
            f"<sup>{trend_message}</sup>",
            "x":0.05
        },
        xaxis_title="Season",
        yaxis_title="Winning Percentage",
        yaxis=dict(
            range=[0,1],
            tickformat=".0%"
        ),
        legend_title="",
        margin=dict(
            l=40,
            r=40,
            t=90,
            b=40
        )
    )

    # Remove clutter
    fig.update_xaxes(
        showgrid=False
    )

    fig.update_yaxes(
        showticklabels=False,
        showgrid=False,
        gridcolor="#eeeeee"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


def forecast_team(team, last_year=2026):

    forecast_df = team_stats_clean.copy()

    # team_id = forecast_df[forecast_df["team"] == team]["team_id"].iloc[0]
    team_id = encoder.transform(
        [team]
    )[0]

    future_predictions = []

    years = [i for i in range(2024,last_year+1)]

    for year in years:

        # 1. Create future feature row
        future_row = add_future_year(
            team,
            year,
            forecast_df,
            noise_factor=1,
        )

        future_row['team'] = team
        future_row['team_id'] = team_id

        # 2. Add it temporarily
        forecast_df = pd.concat(
            [
                forecast_df,
                future_row
            ],
            ignore_index=True
        )

        # print(forecast_df["year"])

        # 3. Recalculate engineered features
        forecast_df = engineer_features(
            forecast_df
        )


        # 4. Build the LSTM sequence
        X_future = create_prediction_sequence(
            forecast_df,
            team_id,
            seq_length=MAX_SEQUENCE_LENGTH,
            feature_columns=feature_columns
        )


        X_future = torch.tensor(
            X_future,
            dtype=torch.float32
        ).to(device)


        team_tensor = torch.tensor(
            [team_id],
            dtype=torch.long
        ).to(device)

        # 5. Predict
        with torch.no_grad():

            pred = model(
                X_future,
                team_tensor
            )


        pred = pred.item()


        future_predictions.append(
            {
                "year": year,
                "winning_percentage": pred
            }
        )

        # 6. Replace placeholder with prediction
        forecast_df.loc[
            (forecast_df.team == team) &
            (forecast_df.year == year),
            "winning_percentage"
        ] = pred

    winning_forecasted = (
        forecast_df[
            forecast_df.team==team
        ][
            ['year','winning_percentage']
        ]
    )

    print(winning_forecasted)
    plot_winning_percentage(winning_forecasted, team)
