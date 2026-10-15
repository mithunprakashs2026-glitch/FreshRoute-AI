#PS 07 — FreshRoute: Intelligent Decision Support for Perishable Produce Distribution
# FreshRoute AI — Hackathon MVP

A local Streamlit prototype for perishable produce allocation, explainable dispatch planning, and scenario replanning.

## Run on Windows (VS Code terminal)

1. Extract `FreshRoute_AI.zip`.
2. Open the extracted `freshroute_ai` folder in VS Code.
3. Open Terminal → New Terminal.
4. Create a virtual environment (recommended):

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, use `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` and run with `.\.venv\Scripts\python.exe -m streamlit run app.py`.

5. Install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

6. Start the app:

   ```powershell
   python -m streamlit run app.py
   ```

7. Streamlit prints a local URL, usually `http://localhost:8501`. Open it in your browser.

## What to demo

1. Show the baseline vs FreshRoute scorecard.
2. Change transport delay and click Recalculate plans.
3. Explain the dispatch table and feasibility checks.
4. Download the CSV.
5. Be transparent: the dataset is simulated, the optimizer is a greedy heuristic, and metrics are estimates.

## Important
This is a starting MVP, not a claim of globally optimal planning or real-world performance. Test all scenarios before presenting.
