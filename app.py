# app.py (refactored)
# Requirements: streamlit, yfinance, tweepy, textblob, scikit-learn, pandas, numpy, mlflow (optional), joblib

import os
import logging
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import tweepy
from textblob import TextBlob
import joblib

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Optional: MLflow (if installed/configured)
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except Exception:
    MLFLOW_AVAILABLE = False

# -----------------------
# CONFIG / SETUP
# -----------------------
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

logging.basicConfig(
    filename="logs/run.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("App started")

BEARER_TOKEN = os.environ.get("TWITTER_BEARER_TOKEN", "YOUR_TWITTER_BEARER_TOKEN_HERE")

# -----------------------
# Helper utilities
# -----------------------
def safe_col_to_str(col):
    if isinstance(col, tuple):
        parts = [str(c).strip() for c in col if str(c).strip() != ""]
        return "_".join(parts) if parts else str(col)
    return str(col)

def normalize_columns(df):
    new_cols = {}
    for col in df.columns:
        new_name = safe_col_to_str(col)
        new_name = new_name.replace("\n", " ").strip()
        base = new_name
        i = 1
        while new_name in new_cols.values():
            new_name = f"{base}_{i}"
            i += 1
        new_cols[col] = new_name
    df = df.rename(columns=new_cols)
    return df, new_cols

def detect_close_column(df):
    for col in df.columns:
        col_str = safe_col_to_str(col)
        if "close" == col_str or col_str.endswith("_close") or "close" in col_str:
            return col
    for col in df.columns:
        col_str = safe_col_to_str(col)
        if "adjclose" in col_str or "adj_close" in col_str or "adjusted close" in col_str:
            return col
    return None

def fetch_twitter_sentiment(bearer_token, query, max_results=50):
    try:
        if bearer_token is None or bearer_token.startswith("YOUR_TWITTER"):
            logging.warning("Twitter bearer token not provided or left default; skipping Twitter fetch.")
            return 0.0
        client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)
        tweets = client.search_recent_tweets(query=query, max_results=min(max_results,100))
        sentiments = []
        if tweets and getattr(tweets, "data", None):
            for t in tweets.data:
                text = t.text if isinstance(t.text, str) else str(t.text)
                s = TextBlob(text).sentiment.polarity
                sentiments.append(s)
            avg = float(np.mean(sentiments)) if sentiments else 0.0
            logging.info(f"Fetched {len(sentiments)} tweets for query='{query}', avg_sentiment={avg:.4f}")
            return avg
        else:
            logging.info(f"No tweets found for query='{query}'")
            return 0.0
    except Exception as e:
        logging.exception("Twitter fetch error")
        return 0.0

def ensure_feature_columns(df, close_col):
    df = df.copy()
    try:
        df["ma_5"] = df[close_col].rolling(window=5, min_periods=1).mean()
    except Exception:
        df["ma_5"] = np.nan
    try:
        df["ma_10"] = df[close_col].rolling(window=10, min_periods=1).mean()
    except Exception:
        df["ma_10"] = np.nan
    try:
        df["return_1"] = df[close_col].pct_change().fillna(0)
    except Exception:
        df["return_1"] = np.nan
    return df

# -----------------------
# MAIN HIGH-LEVEL FUNCTIONS
# -----------------------

def fetch_data_and_sentiment(ticker, start_date, end_date, max_tweets, bearer_token):
    """
    Fetch historical price data, normalize columns, detect Close column,
    and fetch Twitter sentiment (average polarity).
    Returns: df_norm (renamed dataframe), close_col (string column name in df_norm), avg_sentiment (float)
    """
    # Fetch price data
    try:
        raw = yf.download(ticker, start=start_date, end=end_date, interval="1d", progress=False)
    except Exception as e:
        logging.exception("yfinance download failed")
        raise RuntimeError(f"Failed to fetch data from yfinance: {e}")

    if raw is None or raw.empty:
        logging.warning("Empty DataFrame from yfinance")
        raise RuntimeError("No price data returned for this ticker/date range.")

    raw_reset = raw.reset_index()
    df_norm, mapping = normalize_columns(raw_reset)

    # Detect close column
    close_col_candidate = detect_close_column(df_norm)
    if close_col_candidate is None:
        logging.warning("Could not detect 'Close' column; using synthetic Close.")
        df_norm["Close"] = np.random.rand(len(df_norm)) * 100
        close_col = "Close"
    else:
        # Map to normalized column name: safe_col_to_str may differ from actual df column name,
        # so pick the actual column in df_norm that includes 'close'
        close_col = None
        for c in df_norm.columns:
            if "close" in c.lower():
                close_col = c
                break
        if close_col is None:
            # as ultimate fallback, assign synthetic
            df_norm["Close"] = np.random.rand(len(df_norm)) * 100
            close_col = "Close"

    # Fetch Twitter sentiment
    twitter_query = f"{ticker} stock -is:retweet lang:en"
    avg_sentiment = fetch_twitter_sentiment(bearer_token, twitter_query, max_results=max_tweets)

    return df_norm, close_col, avg_sentiment

def compute_features_and_prepare(df, close_col, avg_sentiment):
    """
    Compute technical features (ma_5, ma_10, return_1), add sentiment, trim invalid rows,
    create target (next-day direction), and return X,y, df_trimmed, existing_features.
    """
    df_features = ensure_feature_columns(df, close_col)
    df_features["sentiment"] = avg_sentiment

    desired_features = ["ma_5", "ma_10", "return_1", "sentiment"]
    existing_features = [f for f in desired_features if f in df_features.columns]

    if not existing_features:
        logging.warning("No computed features detected; creating fallback synthetic features.")
        df_features["ma_5"] = np.random.rand(len(df_features))
        df_features["ma_10"] = np.random.rand(len(df_features))
        df_features["return_1"] = np.random.rand(len(df_features))
        df_features["sentiment"] = avg_sentiment
        existing_features = ["ma_5", "ma_10", "return_1", "sentiment"]

    # drop rows where ALL of the three numeric features are NaN
    numeric_feature_subset = [c for c in ["ma_5", "ma_10", "return_1"] if c in df_features.columns]
    if numeric_feature_subset:
        try:
            df_trimmed = df_features.dropna(subset=numeric_feature_subset, how="all")
        except KeyError:
            df_trimmed = df_features.copy()
    else:
        df_trimmed = df_features.copy()

    if df_trimmed.empty or len(df_trimmed) < 10:
        raise RuntimeError("Not enough data after feature computation. Try widening the date range.")

    # create target (next-day direction) and drop last row (nan target)
    df_trimmed["target"] = (df_trimmed[close_col].shift(-1) > df_trimmed[close_col]).astype(int)
    df_model = df_trimmed.iloc[:-1].copy()

    X = df_model[existing_features]
    y = df_model["target"]

    # drop rows that have all NaNs in X
    if X.isnull().all(axis=1).any():
        mask = ~X.isnull().all(axis=1)
        X = X[mask]
        y = y[mask]

    if len(X) < 10:
        raise RuntimeError("Not enough valid training samples after cleaning.")

    return X, y, df_trimmed, existing_features

def train_evaluate_and_save(X, y, existing_features, ticker, avg_sentiment, mlflow_available):
    """
    Train HistGradientBoostingClassifier, evaluate, log to MLflow if available,
    save model and processed CSV. Returns (model, acc, model_path, proc_path).
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = HistGradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # MLflow logging (optional)
    if mlflow_available:
        try:
            mlflow.start_run()
            mlflow.log_param("ticker", ticker)
            mlflow.log_param("model_type", "HistGradientBoostingClassifier")
            mlflow.log_param("features", existing_features)
            mlflow.log_metric("accuracy", float(acc))
            mlflow.log_param("twitter_avg_sentiment", float(avg_sentiment))
            mlflow.end_run()
        except Exception:
            logging.exception("MLflow logging failed")

    # Save model and processed data
    model_path = os.path.join("models", f"{ticker.replace('.', '_')}_hgb_model.pkl")
    joblib.dump(model, model_path)

    # We'll let the caller save df_model if they want. For convenience, return a placeholder processed path.
    proc_path = os.path.join("data", "processed", f"{ticker.replace('.', '_')}_processed.csv")

    return model, acc, model_path, proc_path, X_test, y_test, y_pred

# -----------------------
# STREAMLIT UI (main)
# -----------------------
st.set_page_config(page_title="Stock Prediction with Twitter Sentiment", layout="wide")
st.title("📈 Stock Prediction App with Twitter Sentiment (Refactored)")
STOCK_LIST = [

    # ======================
    # US STOCKS (NASDAQ / NYSE)
    # ======================
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA",
    "NFLX", "INTC", "AMD", "ORCL", "IBM", "ADBE", "CSCO",
    "QCOM", "AVGO", "CRM", "PYPL", "UBER", "LYFT",
    "BA", "GE", "JPM", "BAC", "WFC", "GS", "MS",
    "KO", "PEP", "MCD", "NKE", "DIS", "WMT", "COST",
    "XOM", "CVX", "PFE", "JNJ", "MRNA", "ABBV",

    # ======================
    # INDIAN STOCKS (NSE)
    # ======================
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "WIPRO.NS",
    "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS",
    "KOTAKBANK.NS", "INDUSINDBK.NS",

    "ITC.NS", "HINDUNILVR.NS", "NESTLEIND.NS", "BRITANNIA.NS",
    "TATACONSUM.NS",

    "LT.NS", "ULTRACEMCO.NS", "GRASIM.NS", "ADANIENT.NS",
    "ADANIPORTS.NS", "ADANIGREEN.NS",

    "TATAMOTORS.NS", "MARUTI.NS", "M&M.NS", "BAJAJ-AUTO.NS",
    "HEROMOTOCO.NS",

    "BHARTIARTL.NS", "IDEA.NS",

    "POWERGRID.NS", "NTPC.NS", "ONGC.NS", "COALINDIA.NS",

    "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS",

    "HCLTECH.NS", "TECHM.NS", "LTIM.NS",

    "ASIANPAINT.NS", "TITAN.NS", "UPL.NS",

]

ticker = st.selectbox(
    "Select Stock Ticker:",
    options=STOCK_LIST,
    index=STOCK_LIST.index("RELIANCE.NS") if "RELIANCE.NS" in STOCK_LIST else 0
)

start_date = st.date_input("Start date", value=pd.to_datetime("2018-01-01"))
end_date = st.date_input("End date", value=pd.to_datetime("2024-12-31"))
max_tweets = st.slider("Max tweets to fetch for sentiment", min_value=10, max_value=200, value=50, step=10)
run = st.button("Fetch Data & Train")

if run:
    try:
        # 1) Fetch data and sentiment
        st.info("Fetching price data and Twitter sentiment...")
        df_norm, close_col, avg_sentiment = fetch_data_and_sentiment(
            ticker, start_date, end_date, max_tweets, BEARER_TOKEN
        )
        st.success("Data fetched and normalized.")
        st.write("Sample data:")
        st.dataframe(df_norm.head())

        st.write(f"Using Close column: `{close_col}`")
        st.write(f"Avg Twitter sentiment (recent): {avg_sentiment:.4f}")

        # 2) Compute features and prepare X,y
        st.info("Computing features and preparing dataset...")
        X, y, df_trimmed, existing_features = compute_features_and_prepare(df_norm, close_col, avg_sentiment)
        st.success("Features prepared.")
        st.write(f"Using features: {existing_features}")
        st.write("Sample features:")
        st.dataframe(X.head())

        # 3) Train, evaluate and save
        st.info("Training model...")
        model, acc, model_path, proc_path, X_test, y_test, y_pred = train_evaluate_and_save(
            X, y, existing_features, ticker, avg_sentiment, MLFLOW_AVAILABLE
        )
        st.success(f"Training complete — test accuracy: {acc:.3f}")
        st.text("Classification report (test set):")
        st.text(classification_report(y_test, y_pred, zero_division=0))

        # Save processed dataframe (df_trimmed -> drop last row already used for modeling)
        df_trimmed.iloc[:-1].to_csv(proc_path, index=False)
        st.info(f"Saved processed data to `{proc_path}`")

        st.success(f"Saved trained model to `{model_path}`")

        # Predict next day using latest features row
        latest_row = df_trimmed[existing_features].iloc[-1]
        latest_X = np.array(latest_row).reshape(1, -1)
        try:
            pred = model.predict(latest_X)[0]
            prob = model.predict_proba(latest_X)[0] if hasattr(model, "predict_proba") else None
        except Exception:
            pred = model.predict(latest_row.values.reshape(1, -1))[0]
            prob = model.predict_proba(latest_row.values.reshape(1, -1))[0] if hasattr(model, "predict_proba") else None

        result = "📈 UP" if int(pred) == 1 else "📉 DOWN"
        st.subheader("Prediction for Next Trading Day")
        st.write(f"Prediction: **{result}**")
        if prob is not None:
            st.write(f"Probability: {prob}")

        # Plot price and MAs
        # Robust Streamlit plot (replace your plotting section with this)
        import matplotlib.pyplot as plt

        try:
            st.subheader("Price & Moving Averages (last 200 rows - matplotlib)")
            plot_cols = [close_col] + [c for c in ["ma_5", "ma_10"] if c in df_trimmed.columns]
            plot_df = df_trimmed[plot_cols].tail(200).copy()
            for c in plot_df.columns:
                plot_df[c] = pd.to_numeric(plot_df[c], errors='coerce')
            plot_df = plot_df.dropna(how='all')
            if plot_df.empty:
                raise ValueError("No numeric data to plot.")

            # set datetime index if available
            if 'Date' in df_trimmed.columns:
                plot_df.index = pd.to_datetime(df_trimmed['Date'].tail(len(plot_df)))
            plt.figure(figsize=(10,5))
            for col in plot_df.columns:
                plt.plot(plot_df.index, plot_df[col], label=col)
            plt.legend()
            plt.title(f"{ticker} Close & Moving Averages")
            plt.xlabel("Date")
            plt.ylabel("Price")
            st.pyplot(plt.gcf())
            plt.close()

        except Exception as e:
            logging.exception("Matplotlib plotting failed")
            st.warning(f"Could not plot price chart: {str(e)}")



        logging.info(f"Completed run for {ticker} — accuracy={acc:.4f}, sentiment={avg_sentiment:.4f}")
        st.balloons()

    except Exception as e:
        logging.exception("Main run error")
        st.error(f"Error: {e}")
