import os
import functions as f


if __name__ == "__main__":
    os.chdir(r"C:\Users\wgedd\Documents\GitHub\stock-options-project\data")
    print("\nCurrent working directory: ", os.getcwd())

    start_date = "2021-01-01"
    end_date = "2024-06-30"

    tickers = ["AAPL", "GME"]

    for ticker in tickers:
        print(f"\nProcessing data for {ticker}...\n")

        # Fetch stock data:
        stock_data = f.fetch_stock_data(ticker, start_date, end_date)
        print(f"\nRaw data ({ticker}):\n")
        print(stock_data.head())

        # Pre-process stock data:
        processed_data = f.preprocess_stock_data(stock_data)
        print(f"\nPre-processed stock data ({ticker}):\n")
        print(processed_data.head())

        # Save pre-processed stock data to CSV:
        """path = f"June_30_preprocessed_stock_data_{ticker}.csv"
        processed_data.to_csv(path, index=True)
        print("Pre-processed stock data saved to CSV.\n")"""

        # Fetch options data:
        options_data = f.fetch_options_data(ticker)
        print("\nCall options data (raw):\n")
        print(options_data.calls.head())
        print("\nPut options data (raw):\n")
        print(options_data.puts.head())

        # Pre-process options data:
        preprocessed_options_data = f.preprocess_options_data(options_data)
        print("\nCall options data (pre-processed):\n")
        print(preprocessed_options_data.calls.head())
        print("\nPut options data (pre-processed):\n")
        print(preprocessed_options_data.puts.head())

        # Save pre-processed options data to CSV:
        """path = f"June_30_preprocessed_calls_data_{ticker}.csv"
        preprocessed_options_data.calls.to_csv(path, index=True)
        path = f"June_30_preprocessed_puts_data_{ticker}.csv"
        preprocessed_options_data.puts.to_csv(path, index=True)
        print("Pre-processed options data saved to CSV.\n")"""

        # Perform exploratory data analysis:
        f.perform_eda(ticker, processed_data, preprocessed_options_data)

        # Split the data into training and test sets:
        X_train, X_test, y_train, y_test = f.split_data(processed_data, test_size=0.2)

        # Train & evaluate classification model:
        """model = f.train_classification_model(X_train, y_train)
        accuracy, report = f.evaluate_classification_model(model, X_test, y_test)"""

        # Train & evaluate regression model:
        """model = f.train_regression_model(X_train, y_train)
        mse, mae, r2 = f.evaluate_regression_model(model, X_test, y_test)"""
        best_model = f.tune_regression_hyperparameters(X_train, y_train)
        a, b, c = f.evaluate_regression_model(best_model, X_test, y_test)
        print(f"\nBest model: {best_model}\n")
        print(f"\nBest MSE: {a}\n")
        print(f"\nBest MAE: {b}\n")
        print(f"\nBest R^2: {c}\n")

        # Analyze feature importance:
        feature_importance_df = f.analyze_feature_importance(best_model, X_train)
