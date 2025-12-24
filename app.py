# ============================
# app.py
# ============================

import os
import logging
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from newsapi import NewsApiClient
from textblob import TextBlob

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# -----------------------
# ENV SETUP
# -----------------------
load_dotenv()
NEWS_API_KEY = os.environ.get("NEWS_API_KEY")

st.set_page_config(page_title="Stock Prediction with Sentiment", layout="wide")
st.title("📈 Stock Prediction using Market News Sentiment")

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -----------------------
# NEWS SENTIMENT FUNCTION
# -----------------------
def fetch_news_sentiment(company_name):
    if not NEWS_API_KEY:
        st.warning("News API key not found. Sentiment set to 0.")
        return 0.0

    try:
        newsapi = NewsApiClient(api_key=NEWS_API_KEY)
        articles = newsapi.get_everything(
            q=company_name,
            language="en",
            sort_by="relevancy",
            page_size=20
        )

        sentiments = []
        for article in articles["articles"]:
            text = (article["title"] or "") + " " + (article["description"] or "")
            polarity = TextBlob(text).sentiment.polarity
            sentiments.append(polarity)

        return float(np.mean(sentiments)) if sentiments else 0.0

    except Exception:
        logging.exception("News sentiment error")
        return 0.0

# -----------------------
# FEATURE ENGINEERING
# -----------------------
def add_features(df):
    df["ma_5"] = df["Close"].rolling(5).mean()
    df["ma_10"] = df["Close"].rolling(10).mean()
    df["return_1"] = df["Close"].pct_change()
    return df

# -----------------------
# TRAINING PIPELINE
# -----------------------
def train_model(ticker, company_name, start_date, end_date):

    df = yf.download(ticker, start=start_date, end=end_date)
    if df.empty:
        raise RuntimeError("No stock data found.")

    df.reset_index(inplace=True)

    # Ensure datetime
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    sentiment = fetch_news_sentiment(company_name)

    df = add_features(df)
    df["sentiment"] = sentiment

    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df.dropna(inplace=True)

    features = ["ma_5", "ma_10", "return_1", "sentiment"]
    X = df[features]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = HistGradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    model_path = f"models/{ticker}_model.pkl"
    joblib.dump(model, model_path)

    return model, acc, sentiment, df, X_test, y_test, y_pred, model_path

# -----------------------
# STREAMLIT UI
# -----------------------
STOCKS = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Google",
    "AMZN": "Amazon",
    "TSLA": "Tesla",
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS": "Tata Consultancy Services",
    "INFY.NS": "Infosys",
    "HDFCBANK.NS": "HDFC Bank"
}

ticker = st.selectbox("Select Stock", list(STOCKS.keys()))
company_name = STOCKS[ticker]

start_date = st.date_input("Start Date", pd.to_datetime("2019-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2024-12-31"))

if st.button("Train & Predict"):
    try:
        with st.spinner("Training model and analyzing sentiment..."):
            model, acc, sentiment, df, X_test, y_test, y_pred, model_path = train_model(
                ticker, company_name, start_date, end_date
            )

        st.success("Model trained successfully!")

        st.metric("Model Accuracy", f"{acc:.2f}")
        st.metric("News Sentiment", f"{sentiment:.3f}")

        st.text("Classification Report")
        st.text(classification_report(y_test, y_pred))

        latest = df[["ma_5", "ma_10", "return_1", "sentiment"]].iloc[-1].values.reshape(1, -1)
        pred = model.predict(latest)[0]
        prob = model.predict_proba(latest)[0]

        result = "📈 UP" if pred == 1 else "📉 DOWN"

        st.subheader("Next Trading Day Prediction")
        st.write(f"Prediction: **{result}**")
        st.write(f"Probability: {prob}")

        st.success(f"Model saved at `{model_path}`")

        # -----------------------
        # FIXED STOCK PRICE CHART
        # -----------------------
        st.subheader("Stock Price Chart")

        plt.figure(figsize=(10, 4))
        plt.plot(df["Date"], df["Close"], label="Close Price")
        plt.plot(df["Date"], df["ma_5"], label="MA 5")
        plt.plot(df["Date"], df["ma_10"], label="MA 10")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.title(f"{ticker} Closing Price")
        plt.legend()
        plt.grid(True)

        st.pyplot(plt.gcf())
        plt.close()

    except Exception as e:
        st.error(str(e))
        logging.exception("Application error")
