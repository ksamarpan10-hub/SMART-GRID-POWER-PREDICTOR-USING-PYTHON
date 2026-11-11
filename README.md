# Smart Grid Power Predictor

Forecasting hourly **power demand** and **power generation** for a smart grid using Python and machine learning.

Python Programming – Team Project, SRM Institute of Science & Technology (B.Tech EEE)

**Team:** Pokala Reethika (RA2311005010059) · Samarpan Kharel (RA2311005010104) · Surya Srinivas (RA2311005010077)
**Guide:** Dr. Raghavendra V, Assistant Professor, Department of Computing Technologies

---

## Objectives
- Generate a realistic one-year hourly smart grid dataset (real sensor data was not available).
- Build features from time, weather and past power values.
- Train and compare three regression models for demand and for total generation.
- Evaluate on unseen, later data with MAE, RMSE, R² and MAPE.
- Demonstrate a 24-hour-ahead recursive forecast.

## Dataset
`smart_grid_data.csv` is synthetic: 8,760 hourly rows (1 Jan 2025 00:00 to 31 Dec 2025 23:00), created by `data_generator.py` with a fixed random seed (42), so it is reproducible.

| Column | Description |
|---|---|
| `timestamp`, `hour`, `day_of_week`, `month`, `is_weekend` | Time features |
| `temperature_C` | Air temperature (°C) |
| `solar_irradiance_Wm2` | Solar irradiance (W/m²) |
| `wind_speed_ms` | Wind speed (m/s) |
| `cloud_cover` | Cloud fraction (0 to 1) |
| `solar_generation_MW`, `wind_generation_MW`, `conventional_generation_MW` | Generation by source (MW) |
| `total_generation_MW` | Sum of all generation (MW) |
| `demand_MW` | Power demand (MW) |

To use your own data, keep the same column names (or edit the feature lists in `smart_grid_predictor.py`).

## Method (short)
1. **Features:** sine/cosine encoding of hour, month and day of week; demand/generation lagged by 1 h and 24 h; 24-hour rolling mean (all shifted so the target never leaks into its own inputs); weather inputs.
2. **Split:** chronological, first 85% for training, last 15% for testing (no shuffling).
3. **Models:** Linear Regression (baseline), Random Forest (200 trees), Gradient Boosting (250 trees). The model with the best test R² is kept for each target.
4. **Forecast:** recursive 24-hour-ahead prediction for 31 Dec 2025, where each prediction feeds the next hour's lag features. Weather values come from the dataset in place of a weather-forecast feed.

## Results (test set)

| Target | Best model | MAE (MW) | RMSE (MW) | R² | MAPE (%) |
|---|---|---|---|---|---|
| Demand | Gradient Boosting | 15.71 | 20.09 | 0.9442 | 2.21 |
| Generation | Gradient Boosting | 34.27 | 44.26 | 0.7414 | 4.61 |

24-hour recursive forecast MAE: demand 16.21 MW, generation 43.85 MW.

Full tables, figures and discussion are in the project report in `report/`.

## Requirements
- Python 3.9 or newer
- numpy, pandas, scikit-learn, matplotlib, seaborn, joblib (see `requirements.txt`)

## How to run
```bash
# 1. clone and enter the repository
git clone https://github.com/<your-username>/smart-grid-power-predictor.git
cd smart-grid-power-predictor

# 2. (optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. install dependencies
pip install -r requirements.txt

# 4. generate the dataset (skip if smart_grid_data.csv is already present)
python data_generator.py

# 5. train, evaluate, plot and forecast
python smart_grid_predictor.py
```

Step 5 prints the model comparison and forecast to the terminal, saves all figures and `forecast_next_24h.csv` to `outputs/`, and saves the trained models to `models/`.

### Run in Google Colab
Run each block as its own cell.

**Cell 1: install and upload your files**
```python
!pip install -q -r requirements.txt 2>/dev/null || pip install -q numpy pandas scikit-learn matplotlib seaborn joblib

from google.colab import files
files.upload()   # select data_generator.py and smart_grid_predictor.py
```

**Cell 2: generate data and run the project**
```python
!python data_generator.py
!python smart_grid_predictor.py
```

**Cell 3: show the graphs**
```python
from IPython.display import Image, display
import glob

for f in sorted(glob.glob("outputs/*.png")):
    print(f)
    display(Image(f))
```

**Cell 4 (optional): download the results**
```python
!zip -r results.zip outputs models smart_grid_data.csv
files.download("results.zip")
```

**Using your GitHub repo instead of uploading** (once it's pushed), replace Cell 1 with:
```python
!git clone https://github.com/<your-username>/smart-grid-power-predictor.git
%cd smart-grid-power-predictor
!pip install -q -r requirements.txt
```

Cell 2 will print the model comparison and the 24-hour forecast table.

## Outputs
| File | Content |
|---|---|
| `outputs/01_sample_week_overview.png` | Demand, generation mix and weather for a sample week |
| `outputs/02_correlation_heatmap.png` | Correlation of the main variables |
| `outputs/03_model_comparison.png` | R² of the three models |
| `outputs/04_demand_actual_vs_predicted.png` | Demand prediction vs actual |
| `outputs/05_generation_actual_vs_predicted.png` | Generation prediction vs actual |
| `outputs/06_feature_importance.png` | Feature importance of the best models |
| `outputs/07_next_24h_forecast.png` | 24-hour recursive forecast |
| `outputs/forecast_next_24h.csv` | Forecast values vs actual |
| `models/demand_model.pkl`, `models/generation_model.pkl` | Saved best models (load with `joblib.load`) |

## Repository structure
```
smart-grid-power-predictor/
├── README.md
├── requirements.txt
├── data_generator.py
├── smart_grid_predictor.py
├── smart_grid_data.csv
├── outputs/        # figures and forecast CSV
├── models/         # trained models (.pkl)
└── report/         # project report (PDF)
```

## Limitations
- The data is synthetic, so results show the method works, not how it would perform on a real grid.
- The 24-hour forecast uses actual weather values from the dataset; a real deployment needs a weather-forecast source.
- Lag features make the models depend heavily on accurate recent measurements.
