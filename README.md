# Political Strategy Dashboard 

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Streamlit-based electoral intelligence dashboard for the Varanasi constituency, built for campaign strategists and political data analysts. The app combines 2024 actuals with a pre-trained model bundle to power two panels: a baseline diagnostic view and a 2029 predictive scenario engine.

---

## Core Capabilities

- **Strategic Baseline & Demographics (Panel 1):** 2024 win/loss verdict, constituency-level KPIs (electorate, turnout, vote share, margin), assembly-segment vote share and margin comparisons, caste/community distribution (pie + heatmap), and a filterable, sortable booth classification table (Stronghold / Safe / Vulnerable / Targetable / Opponent Safe).
- **2029 Predictive Scenario Engine (Panel 2):** Live "what-if" simulation with sliders for:
  - Uniform turnout shift
  - Differential demographic mobilization (OBC Yadav, OBC Patel/Kurmi, Muslim, Brahmin, SC Dalit)
  - Direct partisan swing for the target party and leading opponent

  Outputs a live-updating 2029 verdict, projected KPIs vs. 2024 baseline, comparison charts, projected margin-by-segment chart, and a booth flip matrix (gained/lost booths).

---

## How It Works

- `app.py` loads three artifacts at startup: the 2024 actuals CSV, the 2019 backcast CSV, and a `joblib` bundle (`turnout_model`, per-party `party_models`, `train_features`, `all_parties`) — all of which live in the repo root.
- Panel 1 computes baseline stats directly from the 2024 CSV.
- Panel 2 rebuilds the feature matrix by applying the mobilization sliders to caste-share features, re-normalizes caste shares to sum to 1, re-predicts turnout and per-party vote logits (adding the swing sliders to the target/opponent logits), and converts the result to shares via softmax.
- A `SCALE_FACTOR` constant (`5.15`) is applied when aggregating vote/turnout totals from the source CSVs to approximate full constituency-level counts.

> **Note:** This repository does not include the model-training notebook. `model_performance_metrics.csv` holds the recorded validation figures, but the training pipeline itself (feature engineering, model type, hyperparameters) isn't part of `app.py` — treat the `joblib` bundle as a black box unless you also have the training notebook.

---

## Repository Structure

```text
strategic-election-dashboard/
│
├── app.py                              # Main Streamlit application
├── requirements.txt                    # Python package dependencies
├── model_performance_metrics.csv       # Recorded model validation metrics
├── trained_election_models.joblib      # Serialized model bundle (turnout + per-party models)
├── Varanasi_Election_2024_Actuals.csv  # 2024 baseline data and engineered booth features
├── Varanasi_Election_2019_Backcast.csv # 2019 historical backcast dataset
├── .gitignore
├── LICENSE
└── README.md
```

`app.py` loads all its data/model files from the **same directory it runs in** (`DATA_PATH = "./"`). Since the CSVs and the `.joblib` file are already committed to this repo, cloning it is enough — **no manual file upload is required**, in Colab or locally.

---

## Installation & Usage — Local

### Prerequisites
- Python 3.8+
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/AKInfinitiX/strategic-election-dashboard.git
cd strategic-election-dashboard

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

Streamlit will start a local server and print a URL (typically `http://localhost:8501`) — open it in your browser. All required data files are already in the cloned repo, so the app should load without any extra setup.

---

## Installation & Usage — Google Colab

Colab has no direct way to open a localhost port in your browser, so this uses a Cloudflare Tunnel to expose the Streamlit port publicly.

### Step 1: Set up the notebook
Open a new [Google Colab](https://colab.research.google.com/) notebook and run the following in a single cell:

```python
# 1. Clone the repository
!git clone https://github.com/AKInfinitiX/strategic-election-dashboard.git

# 2. Enter the repository directory
%cd strategic-election-dashboard

# 3. Install dependencies
!pip install -q streamlit pandas numpy plotly joblib scikit-learn

# 4. Terminate any old running processes
!pkill -f streamlit
!pkill -f cloudflared

# 5. Download Cloudflare's tunnel binary
!wget -q -nc -O cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
!chmod +x cloudflared

# 6. Launch Streamlit in the background
import subprocess, time
subprocess.Popen([
    "streamlit", "run", "app.py",
    "--server.port", "8501",
    "--server.address", "127.0.0.1",
    "--server.headless", "true"
])

# Wait for the server to initialize
time.sleep(6)

# 7. Start the public Cloudflare tunnel
print("\n=== YOUR DASHBOARD IS LIVE ===")
print("Click the URL below that ends in '.trycloudflare.com'")
print("===============================\n")

!./cloudflared tunnel --url http://127.0.0.1:8501
```

### Step 2: Access the app
Click the generated `*.trycloudflare.com` URL printed at the bottom of the cell output.

> **No file upload step needed.** Since `Varanasi_Election_2024_Actuals.csv`, `Varanasi_Election_2019_Backcast.csv`, and `trained_election_models.joblib` are already committed to this repo, `git clone` pulls them automatically. Just make sure you `%cd` into `strategic-election-dashboard` before launching Streamlit, since `app.py` reads its data files from `./` (the current working directory).

---

## Operational Guide

**Sidebar (always visible):**
- **Select Target / Client Party** — the party the whole dashboard is analyzed from the perspective of.
- **Analytical Modules** — switch between Panel 1 and Panel 2.

**Panel 1 — Strategic Baseline & Demographics:**
- Verdict banner (2024 win/loss vs. the leading opponent)
- KPI row: electorate, votes polled, turnout, target party vote share, net margin
- Assembly segment vote-share comparison and margin bar charts
- Caste/demographic pie chart and per-segment density heatmap
- Filterable booth-level table with segment and category filters

**Panel 2 — 2029 Predictive Scenario Engine:**
- Adjust turnout shift, demographic mobilization sliders, and partisan swing sliders in the sidebar — all charts and metrics update live
- Verdict banner (2029 projected win/loss)
- KPI row comparing projected vs. 2024 baseline
- Vote-share comparison chart (2024 vs. projected) and projected margin-by-segment chart
- Booth flip matrix: booths gained and booths lost by the target party under the simulated scenario

---

## Author & Acknowledgments

**Author:** Akshat Raj Patel (AKInfinitiX)
**Institution:** Indian Institute of Technology (BHU), Varanasi

## License

Distributed under the terms of the MIT License.
