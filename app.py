import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from transformers import pipeline
import ta

# Sentiment analysis setup
classifier = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert"
)

def analyze_sentiment(headlines):
    scores = []
    for text in headlines:
        result = classifier(text)[0]
        label = result["label"]
        if label == "positive":
            scores.append(1)
        elif label == "negative":
            scores.append(-1)
        else:
            scores.append(0)
    return sum(scores) / len(scores) if scores else 0

def add_indicators(df):
    df["RSI"] = ta.momentum.RSIIndicator(close=df["Close"]).rsi()
    macd = ta.trend.MACD(df["Close"])
    df["MACD"] = macd.macd()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()
    return df

def generate_recommendation(sentiment_score, rsi):
    if sentiment_score > 0.3 and rsi < 70:
        return "BUY"
    elif sentiment_score < -0.3:
        return "SELL"
    return "HOLD"

st.set_page_config(
    page_title="AI Stock Dashboard",
    layout="wide"
)

st.title("📈 AI Stock Analysis Dashboard")

stock = st.selectbox(
    "Select Stock",
    ["AAPL", "MSFT", "GOOG", "TSLA", "AMZN"]
)

df = yf.download(stock, period="1y")
df = add_indicators(df)
current_price = df["Close"].iloc[-1]

# Sentiment
ticker = yf.Ticker(stock)
news = ticker.news
headlines = [item["title"] for item in news[:5]]
sentiment_score = analyze_sentiment(headlines)

# Metrics
returns = df["Close"].pct_change()
volatility = returns.std() * (252 ** 0.5) * 100

col1, col2, col3 = st.columns(3)
col1.metric("Current Price", f"${current_price:.2f}")
col2.metric("Volatility", f"{volatility:.2f}%")
col3.metric("RSI", f"{df['RSI'].iloc[-1]:.2f}")

# Plot
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["Close"],
        mode="lines",
        name="Close Price"
    )
)
st.plotly_chart(fig, use_container_width=True)

# Recommendation
recommendation = generate_recommendation(sentiment_score, df["RSI"].iloc[-1])
st.subheader(f"AI Recommendation: {recommendation}")