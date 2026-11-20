"""Simple linear regression, following the style of MSE 446 Tutorials 4 and 5."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. Load the core data. Add new numeric features to this list when available.
root = Path(__file__).resolve().parents[1]
data = pd.read_excel(root / "Data/raw/Ballon_dOr_2015_2025.xlsx", sheet_name="TrainingData")
numeric_features = [
    "league_appearances", "league_minutes_played", "league_goals",
    "league_assists", "league_yellow_cards", "league_red_cards",
]
target = "target_vote_share"


def make_features(frame):
    """Keep numeric stats and turn position/league words into 0/1 columns."""
    positions = pd.get_dummies(frame["position"], prefix="position", dtype=int)
    # A player who changed leagues gets an indicator for each league.
    leagues = frame["league"].str.replace("; ", ";", regex=False).str.get_dummies(sep=";")
    return pd.concat([frame[numeric_features], positions, leagues.add_prefix("league_")], axis=1)


# 2. Keep entire award years together; the 2020 award is absent from the data.
train = data[data["year"] <= 2023]
test = data[data["year"] >= 2024]

# Defender is the reference position. Use the training columns for every split.
X_train = make_features(train).drop(columns="position_Defender")
y_train = train[target]

# 3. Fit ordinary linear regression. The core data has no missing values.
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Compare predictions against ground truth and a training-mean baseline.
metrics = []
prediction_tables = []
for split, frame in [("train", train), ("test", test)]:
    X = make_features(frame).reindex(columns=X_train.columns, fill_value=0)
    y = frame[target]
    predicted = model.predict(X)
    baseline = np.full(len(frame), y_train.mean())

    for name, values in [("linear_regression", predicted), ("mean_baseline", baseline)]:
        metrics.append({
            "split": split,
            "model": name,
            "MAE": mean_absolute_error(y, values),
            "RMSE": mean_squared_error(y, values) ** 0.5,
            "R2": r2_score(y, values),
        })

    results = frame[["year", "player", target]].rename(columns={target: "actual_vote_share"}).copy()
    results["split"] = split
    results["predicted_vote_share"] = predicted
    results["baseline_vote_share"] = baseline
    results["actual_rank"] = results.groupby("year")["actual_vote_share"].rank(ascending=False, method="min")
    results["predicted_rank"] = results.groupby("year")["predicted_vote_share"].rank(ascending=False, method="min")
    prediction_tables.append(results)

# 5. Save results and the model with its input column order.
reports = root / "reports"
reports.mkdir(exist_ok=True)
(root / "models").mkdir(exist_ok=True)

metrics = pd.DataFrame(metrics)
metrics.to_json(reports / "metrics.json", orient="records", indent=2)
predictions = pd.concat(prediction_tables).sort_values(["year", "predicted_rank"])
predictions.to_csv(reports / "predictions.csv", index=False)
pd.DataFrame({"feature": X_train.columns, "coefficient": model.coef_}).to_csv(
    reports / "coefficients.csv", index=False,
)
joblib.dump(
    {"model": model, "feature_columns": X_train.columns.tolist()},
    root / "models/linear_regression.joblib",
)

print(f"Rows: {len(train)} train, {len(test)} test")
print(metrics.round(4).to_string(index=False))
print(f"Intercept: {model.intercept_:.4f}")
print("Saved metrics, predictions, and coefficients to reports/; model to models/.")
