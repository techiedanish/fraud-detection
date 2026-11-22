# Fraud Detection Operations Hub (upload your own CSV)
    pip install -r requirements.txt
    streamlit run app.py
Open http://localhost:8501 → page "0 · Upload & Setup".

## Flow
1. Upload a transactions CSV (or tick the demo-file box).
2. Map your columns (auto-guessed). Only **Timestamp** and **Transaction_Amount** are required; **Card_ID** is strongly recommended (history features); **Fraud_Flag** enables accuracy metrics and in-app training.
3. Choose: pre-trained model (trained on synthetic data) or **Train a new model on my data** (time-ordered 70/15/15 split, thresholds tuned on validation, metrics on the final 15%).
4. Press Run analysis, then browse pages 1–5. Download filtered or all scored transactions as CSV.

## Mapping hints
- PaySim: step → Timestamp (hours), nameOrig → Card_ID, amount → Transaction_Amount, type → Merchant_Category, isFraud → Fraud_Flag.
- IEEE-CIS: TransactionDT → Timestamp (seconds), card1 → Card_ID, TransactionAmt → Transaction_Amount, ProductCD → Merchant_Category, isFraud → Fraud_Flag.
Real datasets lack some fields (device, distance), so detection will be weaker than on the demo file; training on your own labelled data is recommended.

Free hosting: push this folder to GitHub and deploy at share.streamlit.io (main file: app.py).
