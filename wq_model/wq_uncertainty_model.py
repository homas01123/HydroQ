import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


# Function to load and preprocess the data
def load_data(
    file_path,
    encoding="utf-8",
    chl_col="chl_a_mg_L",
    temp_col="water_temp_celcius",
    do_col="doxy_mg_L",
):
    data = pd.read_csv(file_path, encoding=encoding)

    # Print column names to help debugging
    print("Available columns in the dataset:")
    print(data.columns.tolist())

    # Check if the specified columns exist
    required_cols = [chl_col, temp_col, do_col]
    missing_cols = [col for col in required_cols if col not in data.columns]

    if missing_cols:
        raise KeyError(f"The following required columns are missing: {missing_cols}")

    # Handle missing values
    data = data.dropna(subset=[chl_col, temp_col, do_col])

    # Extract features and target
    X = data[[chl_col, temp_col]].values
    y = data[do_col].values

    # Split data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    # Standardize the data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_X.fit_transform(X_train)
    X_val = scaler_X.transform(X_val)
    X_test = scaler_X.transform(X_test)

    # Reshape y for the scaler
    y_train = y_train.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    y_train = scaler_y.fit_transform(y_train).flatten()
    y_val = scaler_y.transform(y_val).flatten()
    y_test = scaler_y.transform(y_test).flatten()

    return (X_train, y_train, X_val, y_val, X_test, y_test, scaler_X, scaler_y)


# Custom loss function for the probabilistic model
def gaussian_nll(y_true, y_pred):
    """
    Negative log likelihood of Gaussian distribution
    y_pred: [mean, log_var]
    """
    mean = y_pred[:, 0]
    log_var = y_pred[:, 1]  # We predict log variance for numerical stability

    # Gaussian NLL
    return 0.5 * (
        tf.math.log(2 * np.pi) + log_var + tf.square(y_true - mean) / tf.exp(log_var)
    )


# Custom MSE metric that only uses the mean prediction
def mean_prediction_mse(y_true, y_pred):
    """
    Calculate MSE only on the mean prediction (first column of y_pred)
    """
    mean_pred = y_pred[:, 0]  # Extract just the mean prediction
    # Use tf.keras.losses.mse instead of the non-existent metrics.mean_squared_error
    return tf.keras.losses.mse(y_true, mean_pred)


# Define the probabilistic neural network model
def build_model(input_dim=2):
    inputs = keras.Input(shape=(input_dim,))

    # Shared layers
    x = keras.layers.Dense(64, activation="relu")(inputs)
    x = keras.layers.Dense(64, activation="relu")(x)

    # Mean prediction branch
    mean = keras.layers.Dense(32, activation="relu")(x)
    mean = keras.layers.Dense(1)(mean)

    # Log variance prediction branch - with improved initialization and activation
    log_var = keras.layers.Dense(32, activation="relu")(x)
    # Use a kernel_initializer closer to 0 and a bias_initializer to start with reasonable variance
    # A bias of -1.0 corresponds to a standard deviation of about 0.6, which is reasonable
    log_var = keras.layers.Dense(
        1,
        kernel_initializer="he_normal",
        bias_initializer=tf.keras.initializers.Constant(-1.0),
    )(log_var)
    # Add a small constant to prevent numerical instability
    log_var_constrained = log_var + tf.keras.backend.epsilon()

    # Combine outputs
    outputs = keras.layers.Concatenate()([mean, log_var_constrained])

    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Compile model with custom loss and custom MSE metric
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=gaussian_nll,
        metrics=[mean_prediction_mse],
    )

    return model


# Training function
def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
    # Create TensorBoard callback
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1
    )

    # Early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=20, restore_best_weights=True
    )

    # Train the model
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, tensorboard_callback],
        verbose=1,
    )

    return model, history


# Evaluation function
def evaluate_model(model, X_test, y_test, scaler_y):
    # Predict on test data
    y_pred = model.predict(X_test)

    # Extract mean and variance predictions
    mean_pred = y_pred[:, 0]
    log_var_pred = y_pred[:, 1]
    var_pred = np.exp(log_var_pred)
    std_pred = np.sqrt(var_pred)

    # Convert back to original scale
    mean_pred_original = scaler_y.inverse_transform(mean_pred.reshape(-1, 1)).flatten()
    y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # Calculate standard deviation in original scale
    std_pred_original = std_pred * np.sqrt(scaler_y.var_[0])

    # Calculate metrics
    mse = np.mean((y_test_original - mean_pred_original) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test_original - mean_pred_original))

    print(f"Test MSE: {mse:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE: {mae:.4f}")

    # Plot predictions vs actual with uncertainty
    plt.figure(figsize=(10, 6))

    # Sort for better visualization
    indices = np.argsort(y_test_original)
    y_test_sorted = y_test_original[indices]
    y_pred_sorted = mean_pred_original[indices]
    y_std_sorted = std_pred_original[indices]

    plt.errorbar(
        np.arange(len(y_test_sorted)),
        y_pred_sorted,
        yerr=1.96 * y_std_sorted,
        fmt="o",
        alpha=0.5,
        label="Predictions with 95% confidence",
    )
    plt.plot(np.arange(len(y_test_sorted)), y_test_sorted, "rx", label="Actual")
    plt.xlabel("Sample index")
    plt.ylabel("DO")
    plt.title("DO Predictions vs Actual with Uncertainty")
    plt.legend()
    plt.savefig("prediction_uncertainty.png")

    return mean_pred_original, std_pred_original, y_test_original
