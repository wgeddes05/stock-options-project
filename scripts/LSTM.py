import os
import numpy as np
import pandas as pd
import functions as f
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout


def prepare_LSTM_data(stock_data, target_column="Open", sequence_length=50):
    """Prepare the data for an LSTM model."""
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(stock_data)
    scaled_df = pd.DataFrame(scaled_data, columns=stock_data.columns)

    X = []
    y = []

    for i in range(sequence_length, len(scaled_df)):
        X.append(scaled_df.iloc[i - sequence_length : i].values)
        y.append(scaled_df.iloc[i][target_column])

    X, y = np.array(X), np.array(y)

    # Split into training and testing datasets
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    return X_train, X_test, y_train, y_test, scaler


def build_LSTM_model(input_shape, layers_info=[(50, True), (50, False)]):
    """Build LSTM model based on specified layers info: (units, return_sequences)."""
    model = Sequential()
    for i, (units, return_sequences) in enumerate(layers_info):
        if i == 0:
            # First layer need to specify input_shape
            model.add(
                LSTM(
                    units=units,
                    return_sequences=return_sequences,
                    input_shape=input_shape,
                )
            )
        else:
            # Subsequent layers infer input_shape from the previous layer
            model.add(LSTM(units=units, return_sequences=return_sequences))
        model.add(Dropout(0.2))

    """model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    # model.add(LSTM(units=50))
    # model.add(Dropout(0.2))"""

    model.add(Dense(units=1))  # output layer for regression
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def train_LSTM_model(X_train, y_train, epochs=50, batch_size=32):
    """Train an LSTM model."""
    input_shape = (X_train.shape[1], X_train.shape[2])

    model = build_LSTM_model(input_shape)
    model.fit(
        X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1
    )

    return model


def evaluate_LSTM_model(model, X_test, y_test, scaler):
    """Evaluate an LSTM model using the provided scaler."""
    y_pred = model.predict(X_test)

    # y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    # y_pred_scaled = scaler.inverse_transform(y_pred.reshape(-1, 1))

    target_min = scaler.min_[0]
    target_scale = scaler.scale_[0]
    y_test_scaled = y_test * target_scale + target_min
    y_pred_scaled = y_pred * target_scale + target_min

    mse = mean_squared_error(y_test_scaled, y_pred_scaled)
    mae = mean_absolute_error(y_test_scaled, y_pred_scaled)
    r2 = r2_score(y_test_scaled, y_pred_scaled)

    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R-squared Score: {r2}")

    return mse, mae, r2


def time_series_cross_validation(X, y, model, num_splits=5):
    """Perform time series cross-validation."""
    tscv = TimeSeriesSplit(n_splits=num_splits)
    mse_scores, mae_scores, r2_scores = [], [], []

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # model = train_LSTM_model(X_train, y_train)
        # mse, mae, r2 = evaluate_LSTM_model(model, X_test, y_test, scaler)

        model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)
        predictions = model.predict(X_test)

        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        mse_scores.append(mse)
        mae_scores.append(mae)
        r2_scores.append(r2)

    return np.mean(mse_scores), np.mean(mae_scores), np.mean(r2_scores)


def simple_backtest(prices, predictions, initial_investment=5000):
    cash = initial_investment
    position = 0
    portfolio_values = []

    for price, predicted_next_price in zip(prices[:1], predictions[1:]):
        if predicted_next_price > price:
            # Buy if prediction is that the price will go up
            if cash >= price:
                shares_to_buy = cash // price
                position += shares_to_buy
                cash -= shares_to_buy * price
        elif predicted_next_price < price:
            # Sell if prediction is that the price will go down
            if position > 0:
                cash += position * price
                position = 0

        portfolio_value = cash + position * price
        portfolio_values.append(portfolio_value)

    if portfolio_values:
        print("Final portfolio value: ", portfolio_values[-1])
    else:
        print("No data.")

    return portfolio_values


if __name__ == "__main__":
    os.chdir(r"C:\Users\wgedd\Documents\GitHub\stock-options-project\data")

    start_date = "2021-01-01"
    end_date = "2024-07-31"

    tickers = ["AAPL"]

    for ticker in tickers:
        print(f"Processing data for {ticker}...")

        stock_data = f.fetch_stock_data(ticker, start_date, end_date)
        print("\nRaw stock data:")
        print(stock_data.head())

        preprocessed_stock_data = f.preprocess_stock_data(stock_data)
        print("\nPreprocessed stock data:")
        print(preprocessed_stock_data.head())

        options_data = f.fetch_options_data(ticker)
        # print("\nRaw call options data:")
        # print(options_data.calls.head())
        # print("\nRaw put options data:")
        # print(options_data.puts.head())

        preprocessed_options_data = f.preprocess_options_data(options_data)
        # print("\nPre-processed call options data:")
        # print(preprocessed_options_data.calls.head())
        # print("\nPre-processed put options data:")
        # print(preprocessed_options_data.puts.head())

        X_train, X_test, y_train, y_test, scaler = prepare_LSTM_data(
            preprocessed_stock_data, target_column="Open", sequence_length=50
        )

        input_shape = (X_train.shape[1], X_train.shape[2])

        # LSTM_model = build_LSTM_model(input_shape)
        # LSTM_model = train_LSTM_model(X_train, y_train)
        # evaluate_LSTM_model(LSTM_model, X_test, y_test, MinMaxScaler())

        best_r2, best_MSE = 0, float("inf")
        best_model, best_config = None, None

        layer_configurations = [
            [(50, False)],
            [(50, True), (50, False)],
            [(50, True), (50, True), (50, False)],
            [(100, True), (50, False)],
        ]

        for config in layer_configurations:
            print(f"Training with configuration: {config}")
            LSTM_model = build_LSTM_model(input_shape, config)

            """LSTM_model.fit(
                X_train, y_train, epochs=50, batch_size=32, validation_split=0.1
            )
            evaluate_LSTM_model(LSTM_model, X_test, y_test, scaler)"""

            mse, mae, r2 = time_series_cross_validation(X_train, y_train, LSTM_model)
            print(f"CV MSE:  {mse}, MAE:  {mae}, R^2:  {r2}")

            if r2 > best_r2:
                best_r2 = r2
                best_MSE = mse
                best_model = LSTM_model
                best_config = config

        """cv_results = time_series_cross_validation()
        for result in cv_results:
            print(f"CV MSE: {result[0]}, MAE: {result[1]}, R^2: {result[2]}")"""

        print("y_test (prices): ", y_test)
        print(f"Best {ticker} configuration: {best_config}")
        print(f"Best MSE: {best_MSE} & best r^2: {best_r2}")

        predictions = best_model.predict(X_test)
        print("Predictions: ", predictions)
        predictions = best_model.predict(X_test).flatten()
        print("Flattened predictions: ", predictions)

        backtest_results = simple_backtest(y_test, predictions, initial_investment=5000)
        print("Backtest results: ", backtest_results)
        dates = list(range(len(backtest_results)))
        print("Dates: ", dates)

        plt.figure(figsize=(12, 6))
        plt.plot(dates, backtest_results, label="Portfolio Value")
        plt.title("Backtest Results")
        plt.xlabel("Days")
        plt.ylabel("Portfolio Value ($)")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{ticker}_backtest_results.png")
        # plt.show
