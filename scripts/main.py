import numpy as np
import pandas as pd
import yfinance as yf


def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data


def fetch_options_data(ticker):
    stock = yf.Ticker(ticker)
    options = stock.option_chain()
    return options


if __name__ == "__main__":
    ticker = "AAPL"
    start_date = "2021-01-01"
    end_date = "2024-01-01"

    # Fetch stock data
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    print(stock_data.head())

    # Fetch options data
    options_data = fetch_options_data(ticker)
    print("\nCall Options Data:")
    print(options_data.calls.head())
    print("\nPut Options Data:")
    print(options_data.puts.head())
