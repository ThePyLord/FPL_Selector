import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from utils import Database, Timer

# Constants
BUDGET = 100  # FPL budget in millions
POSITION_LIMITS = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
MAX_PER_TEAM = 3


# Step 1: Data Collection
def load_historical_data(seasons):
    base_url = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{}/gws/merged_gw.csv"
    dfs = []
    for season in seasons:
        df = pd.read_csv(base_url.format(season))
        df["season"] = season
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def load_fixtures(season):
    url = f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season}/fixtures.csv"
    return pd.read_csv(url)


# Step 2: Feature Engineering
def engineer_features(df, window=3):
    df = df.sort_values(["name", "kickoff_time"])
    stats = ["total_points", "minutes", "goals_scored", "assists", "clean_sheets"]
    for stat in stats:
        df[f"{stat}_rolling_{window}"] = (
            df.groupby("name")[stat]
            .rolling(window, min_periods=1)
            .mean()
            .reset_index(drop=True)
        )

    # Fix opponent team mapping
    team_map = {i + 1: team for i, team in enumerate(df["team"].unique())}
    df["opp_team_name"] = df["opponent_team"].map(team_map)
    df["opponent_strength"] = df.groupby("opp_team_name")["goals_conceded"].transform(
        lambda x: x.rolling(window, min_periods=1).mean()
    )
    return df


# Step 3: Model Training
def train_models(df):
    models = {}
    for pos in ["GK", "DEF", "MID", "FWD"]:
        pos_df = df[df["position"] == pos].dropna()
        features = [
            col
            for col in pos_df.columns
            if "rolling" in col or col == "opponent_strength"
        ]
        X = pos_df[features]
        y = pos_df["total_points"]

        tscv = TimeSeriesSplit(n_splits=5)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            print(f"{pos} MSE: {mean_squared_error(y_test, y_pred):.2f}")
        models[pos] = model
    return models


# Step 4: Predict Points
def predict_next_gameweek(df, models, fixtures, next_gw):
    if next_gw > 38 or next_gw < 1:
        print(f'That gameweek does not exist {next_gw}')
        return
    db = Database()
    latest_df = df[df["GW"] < next_gw].groupby("name").tail(3)  # Last 3 gameweeks
    latest_df = engineer_features(latest_df)
    next_fixtures: pd.DataFrame = fixtures[fixtures["event"] == next_gw]
    team_df = db.get_data("SELECT id, name FROM teams")
    predictions = []
    for pos, model in models.items():
        pos_df = latest_df[latest_df["position"] == pos].copy()
        # print(pos_df.columns)
        nxt = next_fixtures.drop(columns=['stats'])
        # map the team_h and team_a to the team names in team
        nxt = nxt.merge(
            team_df,
            left_on="team_h",
            right_on="id",
            how="left",
            suffixes=("", "_h_name"),
        ).rename(columns={"name": "team_h_name"})
        nxt = nxt.merge(
            team_df,
            left_on="team_a",
            right_on="id",
            how="left",
            suffixes=("", "_a_name"),
        ).rename(columns={"name": "team_a_name"})
        next_fixtures = nxt[
            ["team_h", "team_a", "team_h_name", "team_a_name", "event"]
        ].copy()
        # print(nxt[nxt.columns[-10:].tolist()].head())
        pos_df = pos_df.merge(
            next_fixtures[["team_h_name", "team_a_name"]],
            left_on="team",
            right_on="team_h_name",
            how="left",
        )
        print(pos_df.columns)
        pos_df["opp_team_name"] = pos_df["team_a"].where(
            pos_df["team"] == pos_df["team_h"], pos_df["team_h"]
        )
        pos_df["opponent_strength"] = pos_df.groupby("opp_team_name")[
            "goals_conceded"
        ].transform(lambda x: x.mean())

        features = [
            col
            for col in pos_df.columns
            if "rolling" in col or col == "opponent_strength"
        ]
        X = pos_df[features].dropna()
        pos_df["predicted_points"] = model.predict(X)
        predictions.append(
            pos_df[["name", "position", "team", "value", "predicted_points"]]
        )
    return pd.concat(predictions)


# Step 5: Team Selection
def select_team(predictions):
    predictions["value"] = predictions["value"] / 10  # Convert to millions
    predictions["points_per_cost"] = (
        predictions["predicted_points"] / predictions["value"]
    )
    sorted_players = predictions.sort_values("points_per_cost", ascending=False)

    team = []
    spent = 0
    positions_filled = {pos: 0 for pos in POSITION_LIMITS}
    team_count = {}

    for _, player in sorted_players.iterrows():
        if spent + player["value"] > BUDGET:
            continue
        if positions_filled[player["position"]] >= POSITION_LIMITS[player["position"]]:
            continue
        if team_count.get(player["team"], 0) >= MAX_PER_TEAM:
            continue
        team.append(player)
        spent += player["value"]
        positions_filled[player["position"]] += 1
        team_count[player["team"]] = team_count.get(player["team"], 0) + 1
        if len(team) == 15:
            break
    return pd.DataFrame(team)


# Main Execution
if __name__ == "__main__":
    # Load and prepare data
    seasons = ["2021-22", "2022-23", "2023-24"]
    data = load_historical_data(seasons)
    fixtures = load_fixtures("2024-25")
    data = engineer_features(data)

    # Train models
    models = train_models(data)

    # Predict and select team for next gameweek
    next_gw = (data["GW"].max() - 1) % 39
    print(f'Predicting for: {next_gw}')
    predictions = predict_next_gameweek(data, models, fixtures, next_gw)
    print(f'Predicting for: {predictions.shape}')
    selected_team = select_team(predictions)

    # Output results
    print(f"\nSuggested Team for Gameweek {next_gw}:")
    print(selected_team[["name", "position", "team", "value", "predicted_points"]])
    print(f"Total Cost: {selected_team['value'].sum():.1f}M")
    print(f"Predicted Points: {selected_team['predicted_points'].sum():.1f}")
