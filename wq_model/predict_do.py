# /data/wq/HydroQ/predict_do.py

import argparse
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf


def load_model_and_scalers(model_dir="model"):
    """Load the trained model and scalers"""

    # Define the custom metric function that was used during training
    def mean_prediction_mse(y_true, y_pred):
        """Calculate MSE only on the mean prediction (first column of y_pred)"""
        mean_pred = y_pred[:, 0]  # Extract just the mean prediction
        return tf.keras.losses.mse(y_true, mean_pred)

    model = tf.keras.models.load_model(
        f"{model_dir}/do_uncertainty_model.keras",
        custom_objects={
            "gaussian_nll": lambda y_true, y_pred: 0.5
            * (
                tf.math.log(2 * np.pi)
                + y_pred[:, 1]
                + tf.square(y_true - y_pred[:, 0]) / tf.exp(y_pred[:, 1])
            ),
            "mean_prediction_mse": mean_prediction_mse,  # Include the custom metric
        },
    )

    # Load scalers
    with open(f"{model_dir}/scaler_X.pkl", "rb") as f:
        scaler_X = pickle.load(f)

    with open(f"{model_dir}/scaler_y.pkl", "rb") as f:
        scaler_y = pickle.load(f)

    return model, scaler_X, scaler_y


def predict_do(model, scaler_X, scaler_y, chl_values, temperatures):
    """Predict DO values with uncertainty given chlorophyll values and temperatures"""
    # Check if inputs are single values or lists/arrays
    if not hasattr(chl_values, "__len__"):
        chl_values = [chl_values]
        temperatures = [temperatures]

    # Create input array and normalize
    X = np.array(list(zip(chl_values, temperatures)))
    X_scaled = scaler_X.transform(X)

    # Make prediction
    predictions = model.predict(X_scaled)

    # Extract mean and log variance
    means = predictions[:, 0]
    log_variances = predictions[:, 1]

    # Convert to original scale
    means_original = scaler_y.inverse_transform(means.reshape(-1, 1)).flatten()

    # Improved calculation of standard deviation in original scale
    # Add a small epsilon to ensure numerical stability
    var_pred = np.exp(log_variances) + np.finfo(float).eps
    std_pred = np.sqrt(var_pred)

    # Scale the standard deviation back to original scale
    # Using the standard deviation of the target, not just the variance
    std_original = std_pred * scaler_y.scale_[0]

    # Ensure uncertainty is never too small
    min_uncertainty = 0.01 * np.abs(means_original)
    std_original = np.maximum(std_original, min_uncertainty)

    # Calculate 95% confidence intervals
    lower_ci = means_original - 1.96 * std_original
    upper_ci = means_original + 1.96 * std_original

    return means_original, std_original, lower_ci, upper_ci


def predict_from_file(
    model,
    scaler_X,
    scaler_y,
    input_file,
    output_file=None,
    chl_col="chl_a_mg_L",
    temp_col="water_temp_celcius",
):
    """Predict DO values from a CSV file"""
    # Read input file
    df = pd.read_csv(input_file)

    # Print available columns for debugging
    print("Available columns in the input file:")
    print(df.columns.tolist())

    # Check if required columns exist
    if chl_col not in df.columns or temp_col not in df.columns:
        print(
            f"Warning: Required columns not found. Looking for '{chl_col}' and '{temp_col}'."
        )
        # Try to prompt for correct column names
        print("Available columns:", df.columns.tolist())
        chl_col = (
            input(
                f"Enter column name for chlorophyll (similar to '{chl_col}'): "
            ).strip()
            or chl_col
        )
        temp_col = (
            input(
                f"Enter column name for temperature (similar to '{temp_col}'): "
            ).strip()
            or temp_col
        )

        if chl_col not in df.columns or temp_col not in df.columns:
            raise ValueError(
                f"Input file must contain '{chl_col}' and '{temp_col}' columns"
            )

    # Make predictions
    means, stds, lower_ci, upper_ci = predict_do(
        model, scaler_X, scaler_y, df[chl_col].values, df[temp_col].values
    )

    # Add predictions to dataframe
    df["DO_Predicted"] = means
    df["DO_Uncertainty"] = stds
    df["DO_LowerCI"] = lower_ci
    df["DO_UpperCI"] = upper_ci

    # Save to file if specified
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")

    return df


def visualize_predictions(predictions_df, output_file=None):
    """Visualize predictions with uncertainty"""
    # Sort by predicted value for better visualization
    df_sorted = predictions_df.sort_values(by="DO_Predicted")

    plt.figure(figsize=(12, 6))

    # Plot predictions with error bars
    plt.errorbar(
        range(len(df_sorted)),
        df_sorted["DO_Predicted"],
        yerr=1.96 * df_sorted["DO_Uncertainty"],
        fmt="o",
        alpha=0.6,
        label="Predicted DO with 95% CI",
    )

    # If actual DO exists in the dataframe, plot it too
    if "DO" in df_sorted.columns:
        indices = np.argsort(df_sorted["DO_Predicted"])
        plt.plot(
            range(len(df_sorted)),
            df_sorted["DO"].values[indices],
            "rx",
            label="Actual DO",
        )

    plt.xlabel("Sample Index")
    plt.ylabel("Dissolved Oxygen (DO)")
    plt.title("Predicted DO Values with Uncertainty")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Visualization saved to {output_file}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Predict DO values with uncertainty")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="model",
        help="Directory containing the trained model and scalers",
    )

    # Input options group
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--file", type=str, help="CSV file with ChlValues and Temperature columns"
    )
    input_group.add_argument(
        "--values",
        type=str,
        help='Comma-separated chlorophyll and temperature values (e.g., "5.2,23.8")',
    )

    # Output options
    parser.add_argument("--output", type=str, help="Output file path for CSV results")
    parser.add_argument("--plot", type=str, help="Output file path for visualization")
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Do not print header row in console output",
    )

    # Column name options
    parser.add_argument(
        "--chl-col",
        type=str,
        default="chl_a_mg_L",
        help="Column name for chlorophyll values",
    )
    parser.add_argument(
        "--temp-col",
        type=str,
        default="water_temp_celcius",
        help="Column name for temperature values",
    )

    args = parser.parse_args()

    # Load model and scalers
    try:
        model, scaler_X, scaler_y = load_model_and_scalers(args.model_dir)
        print("Model and scalers loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Process file input
    if args.file:
        try:
            predictions_df = predict_from_file(
                model,
                scaler_X,
                scaler_y,
                args.file,
                args.output,
                chl_col=args.chl_col,
                temp_col=args.temp_col,
            )

            # Display preview of results
            print("\nPrediction Results (first 5 rows):")
            print(predictions_df.head().to_string())

            # Visualize if requested
            if args.plot:
                visualize_predictions(predictions_df, args.plot)

        except Exception as e:
            print(f"Error processing file: {e}")

    # Process direct value input
    else:
        try:
            # Parse the input values
            parts = args.values.split(",")
            if len(parts) != 2:
                raise ValueError(
                    "Input values must be in format 'chlorophyll,temperature'"
                )

            chl = float(parts[0])
            temp = float(parts[1])

            # Make prediction
            means, stds, lower_ci, upper_ci = predict_do(
                model, scaler_X, scaler_y, chl, temp
            )

            # Print results
            if not args.no_header:
                print("\nChl\tTemp\tPredicted_DO\tUncertainty\tLower_CI\tUpper_CI")

            print(
                f"{chl}\t{temp}\t{means[0]:.4f}\t{stds[0]:.4f}\t{lower_ci[0]:.4f}\t{upper_ci[0]:.4f}"
            )

            # Save to file if specified
            if args.output:
                result_df = pd.DataFrame(
                    {
                        "chl_a_mg_L": [chl],
                        "water_temp_celcius": [temp],
                        "DO_Predicted": means,
                        "DO_Uncertainty": stds,
                        "DO_LowerCI": lower_ci,
                        "DO_UpperCI": upper_ci,
                    }
                )
                result_df.to_csv(args.output, index=False)
                print(f"\nResults saved to {args.output}")

        except Exception as e:
            print(f"Error processing input values: {e}")


if __name__ == "__main__":
    main()
