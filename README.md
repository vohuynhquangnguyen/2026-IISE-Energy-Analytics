# 2026 IISE Energy Analytics Competition

Power-outage forecasting pipeline with walk-forward cross-validation,
centralised feature engineering, and YAML-driven model configuration.

## Project Structure

```
├── configs/                        # YAML configuration files
│   ├── pipeline.yaml               #   Top-level pipeline settings (data, CV, model selection)
│   ├── sarimax.yaml                #   SARIMAX hyperparameters
│   ├── seq2seq.yaml                #   Seq2Seq (LSTM) hyperparameters
│   └── dkl.yaml                    #   DKL (Deep Kernel Learning) hyperparameters
│
├── data/                           # Competition NetCDF files (train.nc, test_48h_demo.nc)
│
├── evaluation/                     # Competition scoring metrics
│   ├── __init__.py
│   └── metrics.py                  #   s1–s4 metrics, long↔matrix converters
│
├── models/                         # Forecasting models
│   ├── __init__.py
│   ├── sarimax.py                  #   County-level SARIMAX baseline
│   ├── seq2seq.py                  #   LSTM encoder–decoder
│   └── dkl.py                      #   Deep Kernel Learning (MLP + sparse GP)
│
├── utils/                          # Shared utilities
│   ├── __init__.py
│   ├── config.py                   #   YAML config loader
│   ├── data_loader.py              #   NetCDF loading + walk-forward splitting
│   └── feature_engineering.py      #   Centralised feature engineering (tabular + sequence)
│
├── results/                        # Output predictions (auto-created)
├── demo.ipynb                      # Organiser-provided reference notebook
├── main.py                         # Pipeline entry point
├── requirements.txt                # Python dependencies
└── README.md
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place competition data in ./data/
#    (train.nc and optionally test_48h_demo.nc)

# 3. Run the full pipeline
python main.py
```

## Pipeline Phases

### Phase 1 — Configuration & Data Loading

All settings are loaded from `configs/pipeline.yaml` (top-level) and the
per-model YAML files.  No hard-coded hyperparameters remain in Python code.

### Phase 2 — Walk-Forward Cross-Validation

Walk-forward validation replaces the old single train/val split.  It creates
**K expanding-window folds** (configurable via `n_folds` in `pipeline.yaml`),
each with a fixed 48-hour validation horizon:

```
Fold 0:  [=====train=====][==val==]...............
Fold 1:  [========train========][==val==]..........
Fold 2:  [===========train===========][==val==]....
```

This mimics k-fold CV while respecting temporal ordering — no future leakage.
Competition metrics (s1–s4) are reported per fold and averaged.

### Phase 3 — Test Inference

Each model is retrained on the **full** training set and produces 48-hour
forecasts with prediction intervals, saved to `results/`.

## Configuration

Edit any YAML file under `configs/` to change hyperparameters:

| File             | Controls                                              |
|------------------|-------------------------------------------------------|
| `pipeline.yaml`  | Data paths, number of CV folds, which models to run   |
| `sarimax.yaml`   | ARIMA order, seasonal order, PI significance level    |
| `seq2seq.yaml`   | LSTM architecture, training, conformal calibration    |
| `dkl.yaml`       | Feature engineering, MLP+GP architecture, training    |

All models read their config via a `from_config()` class method:

```python
from models.dkl import DKLForecaster

model = DKLForecaster.from_config("dkl.yaml")
model.fit(ds_train)
```

## Shared Feature Engineering

`utils/feature_engineering.py` contains **all** feature-engineering logic so
that every model trains and validates on identical inputs:

- **Tabular features** (DKL): temporal encodings, outage lags and rolling stats,
  weather lags and rolling stats, interaction terms, derived variables.
- **Sequence features** (Seq2Seq): z-normalisation and sliding-window construction.

## Models

| Model   | Type                     | Features                     | Intervals           |
|---------|--------------------------|------------------------------|----------------------|
| SARIMAX | Univariate time-series   | Outage history only          | Analytical (SARIMAX) |
| Seq2Seq | LSTM encoder-decoder     | Outage + weather (sequences) | Conformal calibration|
| DKL     | MLP + Variational GP     | Tabular (lags + weather)     | Conformal / posterior|

## Reference

The starting codebase is `demo.ipynb` provided by the competition organisers.
