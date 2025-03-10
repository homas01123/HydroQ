import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from wq_uncertainty_model import build_model, evaluate_model, load_data, train_model


def main():
    # Set the file path for the data
    data_path = "../data/insitu_wq_data.csv"

    # Load and preprocess data
    print("Loading and preprocessing data...")

    # Try different encodings if the default utf-8 fails
    encodings_to_try = ["utf-8", "latin-1", "ISO-8859-1", "cp1252"]

    for encoding in encodings_to_try:
        try:
            print(f"Attempting to load data with {encoding} encoding...")
            # First try with default column names
            try:
                X_train, y_train, X_val, y_val, X_test, y_test, scaler_X, scaler_y = (
                    load_data(data_path, encoding=encoding)
                )
                print(
                    f"Successfully loaded data with {encoding} encoding and default column names"
                )
                break
            except KeyError as column_error:
                print(f"Column error: {str(column_error)}")
                # Try to load with a simple preview to manually identify columns
                import pandas as pd

                preview = pd.read_csv(data_path, encoding=encoding, nrows=5)
                print("\nPreview of data:")
                print(preview)

                # Ask for column names
                print("\nPlease enter the correct column names from the preview above:")
                chl_col = input(
                    "Enter column name for chlorophyll (similar to 'chl_a_mg_L'): "
                ).strip()
                temp_col = input(
                    "Enter column name for temperature (similar to 'water_temp_celcius'): "
                ).strip()
                do_col = input(
                    "Enter column name for dissolved oxygen (similar to 'doxy_mg_L'): "
                ).strip()

                X_train, y_train, X_val, y_val, X_test, y_test, scaler_X, scaler_y = (
                    load_data(
                        data_path,
                        encoding=encoding,
                        chl_col=chl_col,
                        temp_col=temp_col,
                        do_col=do_col,
                    )
                )
                print(
                    f"Successfully loaded data with {encoding} encoding and specified column names"
                )
                break

        except UnicodeDecodeError as e:
            print(f"Failed to load with {encoding} encoding: {str(e)}")
            if encoding == encodings_to_try[-1]:
                print("All encoding attempts failed. Please check your data file.")
                return

    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Test data shape: {X_test.shape}")

    # Build the model
    print("Building model...")
    model = build_model(input_dim=X_train.shape[1])
    model.summary()

    # Train the model
    print("Training model...")
    model, history = train_model(model, X_train, y_train, X_val, y_val, epochs=200)

    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    # Fix: Use the actual metric name in the history dictionary
    plt.plot(history.history["mean_prediction_mse"], label="Training MSE")
    plt.plot(history.history["val_mean_prediction_mse"], label="Validation MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_history.png")

    # Evaluate the model
    print("Evaluating model...")
    mean_pred, std_pred, y_test_original = evaluate_model(
        model, X_test, y_test, scaler_y
    )

    # Add diagnostics for uncertainty values
    print("\nUncertainty Diagnostics:")
    print(f"Mean uncertainty: {np.mean(std_pred):.6f}")
    print(f"Min uncertainty: {np.min(std_pred):.6f}")
    print(f"Max uncertainty: {np.max(std_pred):.6f}")

    # Check the raw log variance outputs
    raw_preds = model.predict(X_test)
    log_vars = raw_preds[:, 1]
    print(f"Mean log variance: {np.mean(log_vars):.6f}")
    print(f"Min log variance: {np.min(log_vars):.6f}")
    print(f"Max log variance: {np.max(log_vars):.6f}")

    # Save the model and scalers
    print("Saving model and scalers...")
    if not os.path.exists("model"):
        os.makedirs("model")

    model.save("model/do_uncertainty_model.keras")

    with open("model/scaler_X.pkl", "wb") as f:
        pickle.dump(scaler_X, f)

    with open("model/scaler_y.pkl", "wb") as f:
        pickle.dump(scaler_y, f)

    print("Done!")


if __name__ == "__main__":
    main()
