"""
Smart Grid Power Predictor — Synthetic Data Generator
=======================================================
Generates one year of realistic HOURLY smart grid data:
  - Weather drivers: temperature, solar irradiance, wind speed, cloud cover
  - Time features: hour, day of week, month, weekend flag
  - Power DEMAND (load), shaped by time-of-day peaks, weekday/weekend
    behaviour, and heating/cooling effects from temperature
  - Power GENERATION, split into solar, wind, and conventional
    (dispatched to balance whatever demand renewables don't cover)

Swap this out for a real utility / SCADA / smart-meter dataset later —
just make sure the column names in smart_grid_predictor.py match, or
edit the FEATURE lists there.

Run directly to write smart_grid_data.csv:
    python data_generator.py
"""

import numpy as np
import pandas as pd


def generate_smart_grid_data(start_date="2025-01-01", periods_days=365, seed=42):
    rng = np.random.default_rng(seed)

    timestamps = pd.date_range(start=start_date, periods=periods_days * 24, freq="h")
    n = len(timestamps)

    hour = timestamps.hour.values
    day_of_year = timestamps.dayofyear.values
    day_of_week = timestamps.dayofweek.values  # 0 = Monday
    month = timestamps.month.values
    is_weekend = (day_of_week >= 5).astype(int)

    # ---------------------------------------------------------------
    # WEATHER
    # ---------------------------------------------------------------
    # Seasonal + daily temperature cycle (Northern-hemisphere shape;
    # flip the sign inside sin() if you want Southern-hemisphere seasons)
    seasonal_temp = 15 + 12 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    daily_temp_swing = 5 * np.sin(2 * np.pi * (hour - 9) / 24)
    temperature = seasonal_temp + daily_temp_swing + rng.normal(0, 1.5, n)

    # Solar irradiance (W/m^2): zero at night, bell-shaped through the day,
    # seasonal amplitude, random cloud attenuation
    day_length_factor = 0.55 + 0.45 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    daylight_shape = np.clip(np.sin(np.pi * np.clip((hour - 6) / 12, 0, 1)), 0, 1)
    cloud_cover = np.clip(rng.beta(2, 5, n), 0, 1)  # mostly clear, occasional cloudy
    solar_irradiance = 1000 * day_length_factor * daylight_shape * (1 - 0.85 * cloud_cover)
    solar_irradiance = np.clip(solar_irradiance + rng.normal(0, 10, n), 0, None)

    # Wind speed (m/s): smoothed random walk with a mild winter bias
    wind_base = 6 + 2 * np.sin(2 * np.pi * (day_of_year - 30) / 365)
    wind_noise = pd.Series(rng.normal(0, 1, n)).rolling(6, min_periods=1).mean().values
    wind_speed = np.clip(wind_base + wind_noise * 2.2, 0, None)

    # ---------------------------------------------------------------
    # DEMAND (MW)
    # ---------------------------------------------------------------
    morning_peak = 120 * np.exp(-((hour - 8) ** 2) / (2 * 2.0 ** 2))
    evening_peak = 160 * np.exp(-((hour - 19) ** 2) / (2 * 2.5 ** 2))
    base_load = 500 + morning_peak + evening_peak

    weekday_factor = np.where(is_weekend == 1, 0.87, 1.0)
    temp_effect = 0.6 * (temperature - 20) ** 2          # heating AND cooling raise demand
    seasonal_component = 25 * np.sin(2 * np.pi * (day_of_year - 200) / 365)

    demand = (base_load + temp_effect) * weekday_factor + seasonal_component
    demand += rng.normal(0, 12, n)
    demand = np.clip(demand, 250, None)

    # ---------------------------------------------------------------
    # GENERATION (MW) — solar + wind + conventional (fills the gap)
    # ---------------------------------------------------------------
    solar_capacity_mw = 300
    solar_gen = np.clip(solar_capacity_mw * (solar_irradiance / 1000) + rng.normal(0, 5, n), 0, solar_capacity_mw)

    wind_capacity_mw = 250
    wind_cf = np.clip((wind_speed / 12) ** 3, 0, 1)       # simplified cubic turbine power curve
    wind_gen = np.clip(wind_capacity_mw * wind_cf + rng.normal(0, 8, n), 0, wind_capacity_mw)

    renewable_gen = solar_gen + wind_gen
    reserve_margin = rng.normal(1.02, 0.015, n)           # transmission losses + spinning reserve
    conventional_gen = np.clip(demand * reserve_margin - renewable_gen, 0, None)
    conventional_gen += np.abs(rng.normal(0, 8, n))

    total_generation = solar_gen + wind_gen + conventional_gen

    df = pd.DataFrame({
        "timestamp": timestamps,
        "hour": hour,
        "day_of_week": day_of_week,
        "month": month,
        "is_weekend": is_weekend,
        "temperature_C": temperature.round(2),
        "solar_irradiance_Wm2": solar_irradiance.round(1),
        "wind_speed_ms": wind_speed.round(2),
        "cloud_cover": cloud_cover.round(2),
        "solar_generation_MW": solar_gen.round(2),
        "wind_generation_MW": wind_gen.round(2),
        "conventional_generation_MW": conventional_gen.round(2),
        "total_generation_MW": total_generation.round(2),
        "demand_MW": demand.round(2),
    })
    return df


if __name__ == "__main__":
    df = generate_smart_grid_data()
    df.to_csv("smart_grid_data.csv", index=False)
    print(f"Generated {len(df)} hourly rows -> smart_grid_data.csv")
    print(df[["demand_MW", "total_generation_MW", "temperature_C",
              "solar_irradiance_Wm2", "wind_speed_ms"]].describe().round(1))
