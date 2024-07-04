import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler


class OptionsData:
    def __init__(self, calls, puts):
        self.calls = calls
        self.puts = puts


def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data


def fetch_options_data(ticker):
    stock = yf.Ticker(ticker)
    options = stock.option_chain()
    return OptionsData(options.calls.copy(), options.puts.copy())


def preprocess_stock_data(stock_data):
    # Forward fill missing values
    stock_data = stock_data.fillna(method="ffill")

    # Feature engineering
    stock_data["MA_10"] = stock_data["Close"].rolling(window=10).mean()
    stock_data["MA_50"] = stock_data["Close"].rolling(window=50).mean()
    stock_data["Volatility"] = stock_data["Close"].rolling(window=10).std()
    stock_data["Daily Return"] = stock_data["Close"].pct_change()

    # Drop remaining missing values
    stock_data = stock_data.dropna()

    # Standardize the data features
    scaler = StandardScaler()
    # stock_data_scaled = scaler.fit_transform(stock_data[["Open", "High", "Low", "Close", "Volume"]])
    stock_data[["MA_10", "MA_50", "Volatility", "Daily Return"]] = scaler.fit_transform(
        stock_data[["MA_10", "MA_50", "Volatility", "Daily Return"]]
    )

    return stock_data


def extract_expiration(contract_symbol):
    # Extract the expiration date from the contract symbol
    return pd.to_datetime(contract_symbol[-15:-9], format="%y%m%d")


def preprocess_options_data(options_data):
    # Extract the expiration dates from the contract symbol
    exp = options_data.calls["contractSymbol"].apply(extract_expiration)
    options_data.calls["expiration"] = exp

    exp = options_data.puts["contractSymbol"].apply(extract_expiration)
    options_data.puts["expiration"] = exp

    # Handle missing values
    options_data.calls = options_data.calls.dropna(subset=["strike", "bid", "ask"])
    options_data.puts = options_data.puts.dropna(subset=["strike", "bid", "ask"])

    # Feature engineering
    options_data.calls["mid_price"] = 0.5 * (
        options_data.calls["bid"] + options_data.calls["ask"]
    )
    options_data.puts["mid_price"] = 0.5 * (
        options_data.puts["bid"] + options_data.puts["ask"]
    )

    # Calculate time to expiration in days
    current_date = pd.to_datetime("today")
    options_data.calls["days_to_expiration"] = (
        options_data.calls["expiration"] - current_date
    ).dt.days
    options_data.puts["days_to_expiration"] = (
        options_data.puts["expiration"] - current_date
    ).dt.days

    # Standardize the data features
    scaler = StandardScaler()
    options_data.calls[["mid_price", "days_to_expiration"]] = scaler.fit_transform(
        options_data.calls[["mid_price", "days_to_expiration"]]
    )
    options_data.puts[["mid_price", "days_to_expiration"]] = scaler.fit_transform(
        options_data.puts[["mid_price", "days_to_expiration"]]
    )

    return options_data


def perform_eda(ticker, stock_data, options_data):
    # Perform exploratory data analysis
    print("\nSummary Statistics for Stock Data:")
    print(stock_data.describe())

    print("\nSummary Statistics for Call Options Data:")
    print(options_data.calls.describe())

    print("\nSummary Statistics for Put Options Data:")
    print(options_data.puts.describe())

    # Visualizations
    plt.figure(figsize=(12, 6))
    plt.plot(stock_data["Close"], label="Close Price")
    plt.title(ticker + " Closing Price Over Time")
    plt.xlabel("Date")
    plt.ylabel("Closing Price")
    plt.show()


if __name__ == "__main__":
    start_date = "2021-01-01"
    end_date = "2024-06-30"

    tickers = ["AAPL", "GME"]

    for ticker in tickers:
        print(f"Processing data for {ticker}...")

        # Fetch stock data
        stock_data = fetch_stock_data(ticker, start_date, end_date)
        print(f"Raw data ({ticker}):")
        print(stock_data.head())

        # Preprocess stock data
        stock_data = preprocess_stock_data(stock_data)
        print(f"\nPre-processed data ({ticker}):")
        print(stock_data.head())

        # Fetch options data
        options_data = fetch_options_data(ticker)
        print("\nCall options data (raw):")
        print(options_data.calls.head())
        print("\nPut options data (raw):")
        print(options_data.puts.head())

        # Preprocess options data
        options_data = preprocess_options_data(options_data)
        print("\nCall options data (pre-processed):")
        print(options_data.calls.head())
        print("\nPut options data (pre-processed):")
        print(options_data.puts.head())

        # Perform exploratory data analysis
        perform_eda(ticker, stock_data, options_data)
