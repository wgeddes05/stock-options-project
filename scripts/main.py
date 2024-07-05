import os
import talib as ta
import pandas as pd
import yfinance as yf
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


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
    stock_data["RSI"] = ta.RSI(stock_data["Close"].values, timeperiod=14)

    # Bollinger bands
    stock_data["upper_band"], stock_data["middle_band"], stock_data["lower_band"] = (
        ta.BBANDS(stock_data["Close"], timeperiod=20)
    )

    # Drop remaining missing values
    stock_data = stock_data.dropna()

    # Standardize the data features
    scaler = StandardScaler()
    # stock_data_scaled = scaler.fit_transform(stock_data[["Open", "High", "Low", "Close", "Volume"]])
    stock_data[["MA_10", "MA_50", "Volatility", "Daily Return"]] = scaler.fit_transform(
        stock_data[["MA_10", "MA_50", "Volatility", "Daily Return"]]
    )
    stock_data[["RSI", "upper_band", "middle_band", "lower_band"]] = (
        scaler.fit_transform(
            stock_data[["RSI", "upper_band", "middle_band", "lower_band"]]
        )
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


def prepare_datasets(stock_data):
    # Define target variable (i.e. if the price will go up or down)
    stock_data["Target"] = (stock_data["Close"].shift(-1) > stock_data["Close"]).astype(
        int
    )
    stock_data = stock_data.dropna()

    # Define features and target variable
    X = stock_data.drop(columns=["Target", "Close"])
    y = stock_data["Target"]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    # Train a Random Forest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: ", accuracy)
    report = classification_report(y_test, y_pred)
    print("Classification report:\n", report)

    return accuracy, report


if __name__ == "__main__":
    os.chdir(r"C:\Users\wgedd\Documents\GitHub\stock-options-project\data")

    start_date = "2021-01-01"
    end_date = "2024-06-30"

    tickers = ["AAPL", "GME"]

    for ticker in tickers:
        print(f"Processing data for {ticker}...\n")

        # Fetch stock data
        stock_data = fetch_stock_data(ticker, start_date, end_date)
        print(f"Raw data ({ticker}):\n")
        print(stock_data.head())

        # Pre-process stock data
        preprocessed_stock_data = preprocess_stock_data(stock_data)
        path = f"June_30_preprocessed_stock_data_{ticker}.csv"
        preprocessed_stock_data.to_csv(path, index=True)
        print(f"\nPre-processed data ({ticker}):\n")
        print(preprocessed_stock_data.head())

        # Fetch options data
        options_data = fetch_options_data(ticker)
        print("\nCall options data (raw):\n")
        print(options_data.calls.head())
        print("\nPut options data (raw):\n")
        print(options_data.puts.head())

        # Pre-process options data
        preprocessed_options_data = preprocess_options_data(options_data)
        print("\nCall options data (pre-processed):\n")
        print(preprocessed_options_data.calls.head())
        path = f"June_30_preprocessed_calls_data_{ticker}.csv"
        preprocessed_options_data.calls.to_csv(path, index=True)
        print("\nPut options data (pre-processed):\n")
        print(preprocessed_options_data.puts.head())
        path = f"June_30_preprocessed_puts_data_{ticker}.csv"
        preprocessed_options_data.puts.to_csv(path, index=True)

        # Perform exploratory data analysis
        perform_eda(ticker, preprocessed_stock_data, preprocessed_options_data)

        # Prepare the datasets
        X_train, X_test, y_train, y_test = prepare_datasets(preprocessed_stock_data)

        # Train the model
        model = train_model(X_train, y_train)

        # Evaluate the model
        accuracy, report = evaluate_model(model, X_test, y_test)
