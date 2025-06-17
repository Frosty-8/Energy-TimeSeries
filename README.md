# ⚡ Energy Consumption Forecasting with XGBoost

This project focuses on analyzing and forecasting hourly energy consumption using time series data. It employs feature engineering, time-based cross-validation, and XGBoost regression to make future predictions.

## 📂 Project Structure

```
frosty-8-energy-timeseries/
├── codes.py              # Core data processing and modeling logic
├── new_code.py           # Enhanced version with rich progress & visual outputs
├── main.py               # Downloads the dataset using KaggleHub
├── pyproject.toml        # Project dependencies
├── uv.lock               # Lockfile for package versions
├── .python-version       # Python version used
└── README.md             # Project documentation
```

## 📦 Installation

1. **Clone the repository:**

```bash
git clone https://github.com/frosty-8/energy-timeseries.git
cd energy-timeseries
```

2. **Create and activate a virtual environment:**

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
# OR (recommended if using uv)
uv pip install -r pyproject.toml
```

4. **Download dataset:**

```bash
python main.py
```

## 🧠 Features Engineered

- Temporal features: hour, day of week, month, quarter, etc.
- Lag features based on past days (`lag1`, `lag2`, `lag3`)
- Historical consumption patterns
- Visualization of outliers, splits, and forecasts

## 🚀 How to Run

You can choose between two scripts:

- `codes.py`: Basic version for training and predicting
- `new_code.py`: Includes Rich CLI visualizations and improved logging

Run with:

```bash
python new_code.py
```

## 📈 Visual Outputs

- Hourly consumption trends
- Histograms and outlier detection
- Training/Testing time split plots
- TimeSeriesSplit fold performance
- Future prediction plots

## 🤖 Model

- **XGBoost Regressor**
- Time-based cross-validation with gap handling
- Final model is trained on entire dataset
- Predictions generated for a future 1-year hourly range

## 📁 Output

- `model.json`: Trained XGBoost model
- Forecast visualizations displayed via `matplotlib`

## 🔧 Dependencies

See `pyproject.toml` for full list. Key ones include:

- `xgboost`
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn`
- `rich` for terminal UI
- `kagglehub` for dataset download

## 📊 Dataset

- Dataset: [robikscube/hourly-energy-consumption](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption)
- Contains hourly energy usage data for the PJM Interconnection

## 📌 Notes

- Ensure your environment uses **Python 3.12+**
- Customize lag features or model hyperparameters in `new_code.py` as needed

---

**Author**: [@frosty-8](https://github.com/frosty-8)  
**License**: MIT  
