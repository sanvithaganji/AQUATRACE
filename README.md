# Groundwater Over-Extraction Detector & Decision System

> **“A transparent, explainable groundwater extraction detector powered by hydrological physics, real-world aquifer benchmarks (GEMS-GER & California continuous), and tank hardware fingerprinting.”**

The system models what an observation well or borewell *should* naturally do (accounting for seasonality, rainfall recharge, and autoregressive storage), flags anomalous drops exceeding natural bounds, distinguishes **regional climate drought** from **suspicious local extraction**, and generates advisory alerts.

> **Advisory Compliance:** The system classifies wells for field verification as *"possible over-pumping, inspect"*. It strictly provides operational decision support and **never** makes legal claims.

---

## 🚀 Execution Order & Pipeline Scripts

Run the pipeline in the following sequence:

```bash
# Step 0: Ingest, clean, QC-filter, and harmonize GEMS-GER and California datasets
python3 load_data.py

# Part A: Fit Expected-Level Models (Baseline Ridge vs PyTorch LSTM, 4-year holdout)
python3 expected_level.py --dataset both

# Part B: Unsupervised Anomaly Detection (Isolation Forest, 25km Haversine context, Injection test)
python3 anomaly.py --dataset both

# Part C: Hardware Tank Fingerprint (Physical simulator, water-level-only features, 0-100 risk score)
python3 tank_fingerprint.py

# Part D & Evaluation: Apply Part D Status Rules, Audit Metrics & Generate Publication Plots
python3 evaluate.py
```

---

## 📁 Pipeline Architecture & Scripts

| Script | Purpose & Key Implementation Details |
| :--- | :--- |
| [`load_data.py`](file:///Users/sanvithaganji/Desktop/triple%20i/load_data.py) | **Data Ingestion & Harmonization:**<br>• Ingests **GEMS-GER** (3,207 wells, 32 years, weekly) and **California Continuous** (790 stations, daily).<br>• Cleans California data: filters QC codes to clean values `[1, 2, 10]`, removes `255` placeholder codes, converts feet to meters, resamples to weekly.<br>• Harmonizes level sign convention: $\text{depth\_m} = \text{Elevation} - \text{GWL}$ (**bigger = deeper / lower water table**).<br>• Projects coordinates to WGS84 `lat`, `lon` via `pyproj`. |
| [`expected_level.py`](file:///Users/sanvithaganji/Desktop/triple%20i/expected_level.py) | **Expected Natural Level Modeling:**<br>• Evaluates 10–20 unbroken wells per dataset with a strict **4-year test hold-out**.<br>• **Model 1 (Baseline):** Same-week historical average + Ridge regression on 1, 4, 12-week rainfall.<br>• **Model 2 (LSTM):** 12-week input `[level, rain, sin(week), cos(week)]` $\rightarrow$ next-week level.<br>• Automatically compares test MAE/RMSE and selects the best model. |
| [`anomaly.py`](file:///Users/sanvithaganji/Desktop/triple%20i/anomaly.py) | **Aquifer Over-Extraction & Spatial Anomaly Detection:**<br>• Computes $z\text{-gap} = (\text{actual} - \text{expected}) / \sigma(\text{train gaps})$.<br>• Features: `z_gap`, `delta_z_gap_4w`, `fall_rate_4w`, `rain_4w`.<br>• Trains **Isolation Forest** on normal training weeks only.<br>• **25 km Haversine Neighbour Check:** Distinguishes `"regional (likely drought)"` (neighbours depressed) from `"local (suspicious)"` (isolated well drawdown).<br>• Stress-tested with synthetic injected anomalies (2–5x SD decline over 4–12 weeks). |
| [`tank_fingerprint.py`](file:///Users/sanvithaganji/Desktop/triple%20i/tank_fingerprint.py) | **Hardware Demo Tank Fingerprint:**<br>• Built-in **Hydraulic Physics Simulator**: pumping discharge rate + proportional recharge + post-pump exponential recovery.<br>• Slices pump events with `pump_on`, but extracts features **strictly from water level time series**: drawdown depth, drop rate, fall duration, % recovered 5 min post pump-off, start hour.<br>• Isolation Forest trained on 50 normal runs, mapped to calibrated **0–100 Risk Score**.<br>• Modular and runs unchanged on real hardware telemetry CSV. |
| [`evaluate.py`](file:///Users/sanvithaganji/Desktop/triple%20i/evaluate.py) | **Decision Engine, Audit & Visualization:**<br>• Implements **Part D Status Rules** (plain if/else):<br>&nbsp;&nbsp;– **Normal:** Low risk score (< 40) AND no local gap.<br>&nbsp;&nbsp;– **Monitor:** Moderate risk (40–70) OR regional drop only.<br>&nbsp;&nbsp;– **Inspect:** High risk (> 70) OR local gap unusual for 3+ consecutive weeks.<br>• Generates publication plots in `reports/`: `expected_vs_actual.png` and `tank_drawdown_curves.png`. |

---

## 📊 Benchmark & Evaluation Results

### 1. Expected-Level Model (Test-Year Holdout)

| Dataset | Wells Evaluated | Baseline MAE | Baseline RMSE | LSTM MAE | LSTM RMSE | Selected Architecture |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **GEMS-GER (Germany)** | 15 | 0.265 m | 0.313 m | **0.069 m** | **0.108 m** | **LSTM (14/15 wells)** |
| **California Continuous** | 15 | 2.654 m | 3.337 m | **0.451 m** | **0.793 m** | **LSTM (15/15 wells)** |

*In Germany, shallow groundwater dynamics are closely autoregressive; in California, multi-year drought trends make LSTM memory ~4x more accurate than static seasonal climatology.*

---

### 2. Aquifer Anomaly Detection (Injected 2–5x SD Drawdowns)

| Dataset | Injected Events | Anomalies Caught (%) | Average Detection Delay | False Alarms / Well-Year |
| :--- | :---: | :---: | :---: | :---: |
| **GEMS-GER** | 30 | **100.0%** (30/30) | **1.20 weeks** | 3.96 alarms / well-year |
| **California** | 30 | **96.7%** (29/30) | **1.59 weeks** | 5.87 alarms / well-year |

---

### 3. Tank Fingerprint Performance (Held-Out Test Set)

| Metric | Result | Interpretation |
| :--- | :---: | :--- |
| **Abnormal Events Caught** | **100.0%** (20/20) | Caught all prolonged runs, throttled recharge, and night extractions |
| **False Alarms on Normal** | **0.0%** (0/10) | Zero false triggers on held-out 2-minute normal runs |
| **Mean Risk (Normal Events)** | **19.3 / 100** | Comfortably within the Green / Normal band (< 40) |
| **Mean Risk (Abnormal Events)** | **90.5 / 100** | High-confidence Red / Inspect priority (> 70) |

---

## 📈 Visual Reports

The evaluation pipeline produces high-resolution figures in `reports/`:
- **`reports/expected_vs_actual.png`**: Multi-panel time series showing expected natural trajectory, actual drawdown, shaded injection intervals, standardized gaps ($z\text{-gap}$), and inspection alert spans.
- **`reports/tank_drawdown_curves.png`**: Direct comparison of normal vs abnormal physical drawdown and recovery profiles.

```text
reports/
├── evaluation_summary.json       # Machine-readable evaluation metrics
├── expected_vs_actual.png        # Shaded anomaly detection plot
└── tank_drawdown_curves.png      # High-draw vs normal pump cycles plot
```
