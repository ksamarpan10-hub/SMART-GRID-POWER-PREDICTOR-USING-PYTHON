"""
Smart Grid Power Predictor
=============================
Forecasts hourly POWER DEMAND and POWER GENERATION for a smart grid using
weather + time-based features.

Pipeline:
  1. Load data (smart_grid_data.csv — run data_generator.py first, or point
     LOAD_PATH at your own dataset with matching columns)
  2. Feature engineering: cyclical time encoding + lag/rolling features
  3. Time-based train/test split (never shuffle time series data!)
  4. Train & compare 3 models per target: Linear Regression, Random Forest,
     Gradient Boosting — automatically keep the best one
  5. Evaluate with MAE, RMSE, R^2, MAPE
  6. Save plots to outputs/ and trained models to models/
  7. Demonstrate a real 24-hour-ahead recursive forecast

Run:
    python data_generator.py        # creates smart_grid_data.csv
    python smart_grid_predictor.py  # trains, evaluates, forecasts
"""

import os
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

LOAD_PATH = "smart_grid_data.csv"
OUTPUT_DIR = "outputs"
MODEL_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

DEMAND_FEATURES = [
    "hour_sin", "hour_cos", "month_sin", "month_cos", "dow_sin", "dow_cos",
    "is_weekend", "temperature_C",
    "demand_lag_1h", "demand_lag_24h", "demand_roll_24h_mean",
]

GENERATION_FEATURES = [
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "solar_irradiance_Wm2", "wind_speed_ms", "cloud_cover",
    "generation_lag_1h", "generation_lag_24h", "generation_roll_24h_mean",
]


# =====================================================================
# 1. LOAD
# =====================================================================
def load_data(path=LOAD_PATH):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


# =====================================================================
# 2. FEATURE ENGINEERING
# =====================================================================
def engineer_features(df):
    df = df.copy()

    # Cyclical encoding so e.g. hour 23 and hour 0 are treated as adjacent
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Lag / rolling features — always shifted so the current-hour target
    # never leaks into its own features
    df["demand_lag_1h"] = df["demand_MW"].shift(1)
    df["demand_lag_24h"] = df["demand_MW"].shift(24)
    df["demand_roll_24h_mean"] = df["demand_MW"].shift(1).rolling(24).mean()

    df["generation_lag_1h"] = df["total_generation_MW"].shift(1)
    df["generation_lag_24h"] = df["total_generation_MW"].shift(24)
    df["generation_roll_24h_mean"] = df["total_generation_MW"].shift(1).rolling(24).mean()

    return df.dropna().reset_index(drop=True)


def time_based_split(df, test_fraction=0.15):
    """Time series must be split chronologically, never shuffled randomly."""
    split_idx = int(len(df) * (1 - test_fraction))
    return df.iloc[:split_idx].reset_index(drop=True), df.iloc[split_idx:].reset_index(drop=True)


# =====================================================================
# 3. TRAIN + EVALUATE
# =====================================================================
def train_and_evaluate(train, test, features, target, label):
    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target]

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=14, min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=250, max_depth=4, learning_rate=0.05, random_state=42
        ),
    }

    results, predictions = {}, {}
    print(f"\n--- {label} model comparison ---")
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        mape = np.mean(np.abs((y_test - preds) / y_test)) * 100
        results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape, "model": model}
        predictions[name] = preds
        print(f"  {name:18s} | MAE {mae:6.2f} MW | RMSE {rmse:6.2f} MW | R2 {r2:.4f} | MAPE {mape:5.2f}%")

    best_name = max(results, key=lambda k: results[k]["R2"])
    print(f"  -> Best: {best_name}")
    return results, predictions, best_name, y_test


# =====================================================================
# 4. VISUALIZATION
# =====================================================================
def plot_sample_week(df, path):
    week = df.iloc[24 * 100: 24 * 107]  # an arbitrary representative week
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)

    axes[0].plot(week["timestamp"], week["demand_MW"], color="#d62728", label="Demand")
    axes[0].plot(week["timestamp"], week["total_generation_MW"], color="#1f77b4", label="Generation")
    axes[0].set_ylabel("MW")
    axes[0].set_title("Demand vs. Generation — sample week")
    axes[0].legend()

    axes[1].plot(week["timestamp"], week["solar_generation_MW"], color="#ff7f0e", label="Solar")
    axes[1].plot(week["timestamp"], week["wind_generation_MW"], color="#2ca02c", label="Wind")
    axes[1].plot(week["timestamp"], week["conventional_generation_MW"], color="#7f7f7f", label="Conventional")
    axes[1].set_ylabel("MW")
    axes[1].set_title("Generation mix")
    axes[1].legend()

    axes[2].plot(week["timestamp"], week["temperature_C"], color="#9467bd", label="Temperature (C)")
    ax2 = axes[2].twinx()
    ax2.plot(week["timestamp"], week["solar_irradiance_Wm2"], color="#e377c2", alpha=0.6, label="Solar irradiance")
    axes[2].set_ylabel("Temperature (C)")
    ax2.set_ylabel("Irradiance (W/m2)")
    axes[2].set_title("Weather drivers")
    axes[2].legend(loc="upper left")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_correlation_heatmap(df, path):
    cols = ["demand_MW", "total_generation_MW", "temperature_C", "solar_irradiance_Wm2",
            "wind_speed_ms", "cloud_cover", "hour", "is_weekend"]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_model_comparison(demand_results, generation_results, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, results, title in [
        (axes[0], demand_results, "Demand — model comparison (R2)"),
        (axes[1], generation_results, "Generation — model comparison (R2)"),
    ]:
        names = list(results.keys())
        r2s = [results[n]["R2"] for n in names]
        bars = ax.bar(names, r2s, color=["#8c8c8c", "#1f77b4", "#2ca02c"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("R2 score")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=15)
        for b, v in zip(bars, r2s):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center")
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_actual_vs_predicted(test, y_test, preds, best_name, label, unit_label, path):
    fig, axes = plt.subplots(2, 1, figsize=(13, 8))

    show_n = min(24 * 10, len(test))  # last ~10 days of the test period
    ts = test["timestamp"].iloc[-show_n:]
    axes[0].plot(ts, y_test.iloc[-show_n:], label="Actual", color="black", linewidth=1.5)
    axes[0].plot(ts, preds[-show_n:], label=f"Predicted ({best_name})", color="#d62728", linewidth=1.2, alpha=0.85)
    axes[0].set_title(f"{label}: Actual vs Predicted (last {show_n} test hours)")
    axes[0].set_ylabel(unit_label)
    axes[0].legend()

    axes[1].scatter(y_test, preds, alpha=0.3, s=10, color="#1f77b4")
    lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
    axes[1].plot(lims, lims, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel(f"Actual {unit_label}")
    axes[1].set_ylabel(f"Predicted {unit_label}")
    axes[1].set_title("Predicted vs Actual (full test set)")

    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_feature_importance(demand_model, generation_model, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, model, features, title in [
        (axes[0], demand_model, DEMAND_FEATURES, "Demand model — feature importance"),
        (axes[1], generation_model, GENERATION_FEATURES, "Generation model — feature importance"),
    ]:
        if hasattr(model, "feature_importances_"):
            importances = pd.Series(model.feature_importances_, index=features).sort_values()
            importances.plot(kind="barh", ax=ax, color="#1f77b4")
        else:
            ax.text(0.5, 0.5, "Best model has no\nfeature_importances_", ha="center", va="center")
        ax.set_title(title)
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_forecast(forecast_df, path):
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    axes[0].plot(forecast_df["timestamp"], forecast_df["actual_demand_MW"], label="Actual", color="black")
    axes[0].plot(forecast_df["timestamp"], forecast_df["predicted_demand_MW"], label="Forecast",
                 color="#d62728", linestyle="--", marker="o", markersize=3)
    axes[0].set_title("24-hour-ahead recursive forecast — Demand")
    axes[0].set_ylabel("MW")
    axes[0].legend()

    axes[1].plot(forecast_df["timestamp"], forecast_df["actual_generation_MW"], label="Actual", color="black")
    axes[1].plot(forecast_df["timestamp"], forecast_df["predicted_generation_MW"], label="Forecast",
                 color="#1f77b4", linestyle="--", marker="o", markersize=3)
    axes[1].set_title("24-hour-ahead recursive forecast — Generation")
    axes[1].set_ylabel("MW")
    axes[1].legend()
    plt.xticks(rotation=30)

    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# =====================================================================
# 5. RECURSIVE MULTI-STEP FORECAST DEMO
# =====================================================================
def _cyclical(hour, month, dow):
    return {
        "hour_sin": np.sin(2 * np.pi * hour / 24), "hour_cos": np.cos(2 * np.pi * hour / 24),
        "month_sin": np.sin(2 * np.pi * month / 12), "month_cos": np.cos(2 * np.pi * month / 12),
        "dow_sin": np.sin(2 * np.pi * dow / 7), "dow_cos": np.cos(2 * np.pi * dow / 7),
    }


def forecast_next_hours(df, demand_model, generation_model, horizon=24, start_idx=None):
    """
    Recursively forecasts `horizon` hours ahead: each predicted value becomes
    the lag feature for the next step, exactly as it would in production.

    NOTE: weather columns (temperature, solar irradiance, wind, cloud cover)
    for the forecast window are taken from the dataset as a stand-in for a
    weather-forecast API feed, which is what you would use in a real
    deployment. Only the *lag/rolling power features* are genuinely
    recursive/predicted here.
    """
    if start_idx is None:
        start_idx = len(df) - horizon

    demand_hist = list(df["demand_MW"].iloc[:start_idx])
    gen_hist = list(df["total_generation_MW"].iloc[:start_idx])
    future = df.iloc[start_idx:start_idx + horizon].reset_index(drop=True)

    rows = []
    for i in range(len(future)):
        r = future.iloc[i]
        cyc = _cyclical(r["hour"], r["month"], r["day_of_week"])

        d_feat = pd.DataFrame([{
            **cyc, "is_weekend": r["is_weekend"], "temperature_C": r["temperature_C"],
            "demand_lag_1h": demand_hist[-1], "demand_lag_24h": demand_hist[-24],
            "demand_roll_24h_mean": np.mean(demand_hist[-24:]),
        }])[DEMAND_FEATURES]

        g_feat = pd.DataFrame([{
            **cyc, "solar_irradiance_Wm2": r["solar_irradiance_Wm2"],
            "wind_speed_ms": r["wind_speed_ms"], "cloud_cover": r["cloud_cover"],
            "generation_lag_1h": gen_hist[-1], "generation_lag_24h": gen_hist[-24],
            "generation_roll_24h_mean": np.mean(gen_hist[-24:]),
        }])[GENERATION_FEATURES]

        d_pred = float(demand_model.predict(d_feat)[0])
        g_pred = float(generation_model.predict(g_feat)[0])

        demand_hist.append(d_pred)
        gen_hist.append(g_pred)

        rows.append({
            "timestamp": r["timestamp"],
            "predicted_demand_MW": round(d_pred, 2),
            "actual_demand_MW": r["demand_MW"],
            "predicted_generation_MW": round(g_pred, 2),
            "actual_generation_MW": r["total_generation_MW"],
        })

    return pd.DataFrame(rows)


# =====================================================================
# MAIN
# =====================================================================
def main():
    print("Loading data...")
    raw = load_data()
    df = engineer_features(raw)
    print(f"{len(df)} usable rows after feature engineering (from {len(raw)} raw rows)")

    train, test = time_based_split(df, test_fraction=0.15)
    print(f"Train: {len(train)} rows | Test: {len(test)} rows (chronological split, no shuffling)")

    plot_sample_week(raw, f"{OUTPUT_DIR}/01_sample_week_overview.png")
    plot_correlation_heatmap(df, f"{OUTPUT_DIR}/02_correlation_heatmap.png")

    # ---- Demand ----
    demand_results, demand_preds, demand_best, y_demand_test = train_and_evaluate(
        train, test, DEMAND_FEATURES, "demand_MW", "DEMAND"
    )
    demand_best_model = demand_results[demand_best]["model"]

    # ---- Generation ----
    gen_results, gen_preds, gen_best, y_gen_test = train_and_evaluate(
        train, test, GENERATION_FEATURES, "total_generation_MW", "GENERATION"
    )
    gen_best_model = gen_results[gen_best]["model"]

    plot_model_comparison(demand_results, gen_results, f"{OUTPUT_DIR}/03_model_comparison.png")
    plot_actual_vs_predicted(test, y_demand_test, demand_preds[demand_best], demand_best,
                              "Demand", "MW", f"{OUTPUT_DIR}/04_demand_actual_vs_predicted.png")
    plot_actual_vs_predicted(test, y_gen_test, gen_preds[gen_best], gen_best,
                              "Generation", "MW", f"{OUTPUT_DIR}/05_generation_actual_vs_predicted.png")
    plot_feature_importance(demand_best_model, gen_best_model, f"{OUTPUT_DIR}/06_feature_importance.png")

    # ---- Save best models ----
    joblib.dump(demand_best_model, f"{MODEL_DIR}/demand_model.pkl")
    joblib.dump(gen_best_model, f"{MODEL_DIR}/generation_model.pkl")
    print(f"\nSaved best models -> {MODEL_DIR}/demand_model.pkl, {MODEL_DIR}/generation_model.pkl")

    # ---- 24h-ahead recursive forecast demo ----
    print("\n--- 24-hour-ahead recursive forecast demo (held-out final day) ---")
    forecast_df = forecast_next_hours(df, demand_best_model, gen_best_model, horizon=24)
    forecast_df.to_csv(f"{OUTPUT_DIR}/forecast_next_24h.csv", index=False)
    plot_forecast(forecast_df, f"{OUTPUT_DIR}/07_next_24h_forecast.png")

    fc_demand_mae = mean_absolute_error(forecast_df["actual_demand_MW"], forecast_df["predicted_demand_MW"])
    fc_gen_mae = mean_absolute_error(forecast_df["actual_generation_MW"], forecast_df["predicted_generation_MW"])
    print(f"Forecast MAE — Demand: {fc_demand_mae:.2f} MW | Generation: {fc_gen_mae:.2f} MW")
    print(forecast_df.to_string(index=False))

    print(f"\nAll plots saved to ./{OUTPUT_DIR}/  |  Trained models saved to ./{MODEL_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
