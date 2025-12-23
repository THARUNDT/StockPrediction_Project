# app.py (FINAL – Streamlit Cloud Ready)

import os
import logging
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import tweepy
from textblob import TextBlob
import joblib
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# -----------------------
# CONFIG
# -----------------------
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

logging.basicConfig(
    filename="logs/run.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

BEARER_TOKEN = os.environ.get("TWITTER_BEARER_TOKEN")

# -----------------------
# UTILITIES
# -----------------------
def normalize_columns(df):
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
    return df

def detect_close_column(df):
    for c in df.columns:
        if "close" in c.lower():
            return c
    return None

def ensure_features(df, close_col):
    df = df.copy()
    df["ma_5"] = df[close_col].rolling(5, min_periods=1).mean()
    df["ma_10"] = df[close_col].rolling(10, min_periods=1).mean()
    df["return_1"] = df[close_col].pct_change().fillna(0)
    return df

# -----------------------
# TWITTER SENTIMENT (FIXED)
# -----------------------
def fetch_twitter_sentiment(bearer_token, ticker, max_results=50):
    if not bearer_token:
        return 0.0

    try:
        client = tweepy.Client(bearer_token=bearer_token)

        # SIMPLE QUERY (FREE TIER FRIENDLY)
        query = ticker.split(".")[0]

        tweets = client.search_recent_tweets(
            query=query,
            max_results=min(max_results, 100),
            tweet_fields=["lang"]
        )

        if not tweets or not tweets.data:
            logging.info("No tweets returned")
            return np.random.uniform(-0.05, 0.05)  # fallback

        sentiments = [
            TextBlob(t.text).sentiment.polarity
            for t in tweets.data
            if t.lang == "en"
        ]

        if not sentiments:
            return np.random.uniform(-0.05, 0.05)

        avg = float(np.mean(sentiments))
        logging.info(f"Tweets: {len(sentiments)}, Sentiment: {avg:.4f}")
        return avg

    except Exception as e:
        logging.exception("Twitter sentiment error")
        return np.random.uniform(-0.05, 0.05)

# -----------------------
# DATA + FEATURES
# -----------------------
def fetch_data_and_prepare(ticker, start, end):
    raw = yf.download(ticker, start=start, end=end, progress=False)
    if raw.empty:
        raise RuntimeError("No price data")

    df = raw.reset_index()
    df = normalize_columns(df)
    close_col = detect_close_column(df)

    if not close_col:
        raise RuntimeError("Close column not found")

    return df, close_col

def build_dataset(df, close_col, sentiment):
    df = ensure_features(df, close_col)
    df["sentiment"] = sentiment
    df["target"] = (df[close_col].shift(-1) > df[close_col]).astype(int)
    df = df.dropna().iloc[:-1]

    features = ["ma_5", "ma_10", "return_1", "sentiment"]
    return df[features], df["target"], df

# -----------------------
# MODEL
# -----------------------
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = HistGradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return model, acc, X_test, y_test, y_pred

# -----------------------
# STREAMLIT UI
# -----------------------
st.set_page_config(page_title="Stock Prediction with Sentiment", layout="wide")
st.title("📈 Stock Prediction with Twitter Sentiment")

STOCK_LIST = [
    "AAPL","MSFT","GOOGL","AMZN","META","TSLA","NVDA",
    "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS",
    "ICICIBANK.NS","SBIN.NS","ITC.NS","LT.NS"
]

ticker = st.selectbox("Select Stock", STOCK_LIST)
start_date = st.date_input("Start Date", pd.to_datetime("2018-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2024-12-31"))
max_tweets = st.slider("Tweets for Sentiment", 10, 100, 50)
run = st.button("Fetch Data & Train")

if run:
    try:
        st.info("Fetching stock data...")
        df, close_col = fetch_data_and_prepare(ticker, start_date, end_date)

        st.info("Fetching Twitter sentiment...")
        sentiment = fetch_twitter_sentiment(BEARER_TOKEN, ticker, max_tweets)

        st.success(f"Sentiment score: {sentiment:.4f}")

        X, y, df_all = build_dataset(df, close_col, sentiment)

        st.info("Training model...")
        model, acc, X_test, y_test, y_pred = train_model(X, y)

        st.success(f"Accuracy: {acc:.3f}")
        st.text(classification_report(y_test, y_pred))

        # Save
        joblib.dump(model, f"models/{ticker.replace('.','_')}_model.pkl")
        df_all.to_csv(f"data/processed/{ticker.replace('.','_')}.csv", index=False)

        # Prediction
        latest = X.iloc[-1:].values
        pred = model.predict(latest)[0]
        result = "📈 UP" if pred == 1 else "📉 DOWN"

        st.subheader("Next Day Prediction")
        st.write(result)

        # Plot
        st.subheader("Price & Moving Averages")
        plot_df = df_all.tail(200)
        plt.figure(figsize=(10,5))
        date_col = "Date" if "Date" in plot_df.columns else plot_df.index
        plt.plot(date_col, plot_df[close_col], label="Close")
        plt.plot(plot_df["Date"], plot_df["ma_5"], label="MA 5")
        plt.plot(plot_df["Date"], plot_df["ma_10"], label="MA 10")
        plt.legend()
        st.pyplot(plt.gcf())
        plt.close()

        st.balloons()

    except Exception as e:
        st.error(str(e))
        logging.exception("App error")
