import ta
import talib
import pandas as pd
import yfinance as yf
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


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
    stock_data["RSI"] = talib.RSI(stock_data["Close"].values, timeperiod=14)

    # Bollinger bands
    bb_indicator = ta.volatility.BollingerBands(stock_data["Close"], window=20)
    stock_data["upper_band"] = bb_indicator.bollinger_hband()
    stock_data["middle_band"] = bb_indicator.bollinger_mavg()
    stock_data["lower_band"] = bb_indicator.bollinger_lband()

    # MACD
    macd = ta.trend.MACD(
        stock_data["Close"], window_slow=26, window_fast=12, window_sign=9
    )
    stock_data["MACD"] = macd.macd()
    stock_data["MACD_Signal"] = macd.macd_signal()
    stock_data["MACD_Histogram"] = macd.macd_diff()

    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(
        stock_data["High"], stock_data["Low"], stock_data["Close"], window=14
    )
    stock_data["Stoch_K"] = stoch.stoch()
    stock_data["Stoch_D"] = stoch.stoch_signal()

    # Drop remaining missing values
    stock_data = stock_data.dropna()

    # Standardize the data features
    scaler = StandardScaler()
    stock_data[
        [
            "MA_10",
            "MA_50",
            "Volatility",
            "Daily Return",
            "RSI",
            "upper_band",
            "middle_band",
            "lower_band",
            "MACD",
            "MACD_Signal",
            "MACD_Histogram",
            "Stoch_K",
            "Stoch_D",
        ]
    ] = scaler.fit_transform(
        stock_data[
            [
                "MA_10",
                "MA_50",
                "Volatility",
                "Daily Return",
                "RSI",
                "upper_band",
                "middle_band",
                "lower_band",
                "MACD",
                "MACD_Signal",
                "MACD_Histogram",
                "Stoch_K",
                "Stoch_D",
            ]
        ]
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
    print("\nSummary Statistics for Stock Data:\n")
    print(stock_data.describe())

    print("\nSummary Statistics for Call Options Data:\n")
    print(options_data.calls.describe())

    print("\nSummary Statistics for Put Options Data:\n")
    print(options_data.puts.describe())

    # Visualizations
    plt.figure(figsize=(12, 6))
    plt.plot(stock_data["Close"], label="Close Price")
    plt.title(ticker + " Closing Price Over Time")
    plt.xlabel("Date")
    plt.ylabel("Closing Price")
    plt.show()


def split_data(stock_data, test_size=0.2):
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
        X, y, test_size=test_size, random_state=42
    )
    return X_train, X_test, y_train, y_test


def train_classification_model(X_train, y_train):
    """Start with a Random Forest classifier to predict whether the stock price will go up or down."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_regression_model(X_train, y_train):
    """Use a Random Forest regressor to predict the future stock price."""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_classification_model(model, X_test, y_test):
    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print("\nAccuracy: ", accuracy)
    report = classification_report(y_test, y_pred)
    print("\nClassification report:\n", report)

    return accuracy, report


def evaluate_regression_model(model, X_test, y_test):
    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    """print("\nMean Squared Error: ", mse)
    print("\nMean Absolute Error: ", mae)
    print("\nR-squared: ", r2)"""

    return mse, mae, r2


def tune_regression_hyperparameters(X_train, y_train):
    param_grid = {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [None, 5, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
    }

    model = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    print(f"Best hyperparameters: {grid_search.best_params_}")
    return grid_search.best_estimator_


def analyze_feature_importance(model, X_train):
    importances = model.feature_importances_
    feature_names = X_train.columns

    feature_importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": importances}
    )
    feature_importance_df = feature_importance_df.sort_values(
        by="Importance", ascending=False
    )

    print("\nFeature Importance:")
    print(feature_importance_df)

    return feature_importance_df
