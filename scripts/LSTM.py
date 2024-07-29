import os
import numpy as np
import pandas as pd
import functions as f
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout


def prepare_LTSM_data(stock_data, target_column="Open", sequence_length=50):
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


if __name__ == "__main__":
    os.chdir(r"C:\Users\wgedd\Documents\GitHub\stock-options-project\data")

    start_date = "2021-01-01"
    end_date = "2024-06-30"

    tickers = ["AAPL", "GME"]

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

        X_train, X_test, y_train, y_test, scaler = prepare_LTSM_data(
            preprocessed_stock_data, target_column="Open", sequence_length=50
        )

        input_shape = (X_train.shape[1], X_train.shape[2])

        # LSTM_model = build_LSTM_model(input_shape)
        # LSTM_model = train_LSTM_model(X_train, y_train)
        # evaluate_LSTM_model(LSTM_model, X_test, y_test, MinMaxScaler())

        layer_configurations = [
            [(50, False)],
            [(50, True), (50, False)],
            [(50, True), (50, True), (50, False)],
            [(100, True), (50, False)],
        ]

        for config in layer_configurations:
            print(f"Training with configuration: {config}")
            LSTM_model = build_LSTM_model(input_shape, config)
            LSTM_model.fit(
                X_train, y_train, epochs=50, batch_size=32, validation_split=0.1
            )
            evaluate_LSTM_model(LSTM_model, X_test, y_test, scaler)
