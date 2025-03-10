import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score


def plot_training_history(history, save_dir="."):
    """Generate comprehensive plots for training history"""
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Plot loss curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (NLL)")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)

    # Plot MSE curves
    plt.subplot(1, 2, 2)
    plt.plot(history.history["mean_prediction_mse"], label="Training MSE")
    plt.plot(history.history["val_mean_prediction_mse"], label="Validation MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Training and Validation MSE")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_history.png", dpi=300)
    plt.close()

    # Plot learning curves on log scale to see details
    plt.figure(figsize=(10, 6))
    plt.semilogy(history.history["loss"], label="Training Loss")
    plt.semilogy(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.title("Learning Curve")
    plt.legend()
    plt.grid(alpha=0.3, which="both")
    plt.savefig(f"{save_dir}/learning_curve.png", dpi=300)
    plt.close()

    return


def plot_prediction_results(
    model, X_train, y_train, X_val, y_val, X_test, y_test, scaler_y, save_dir="."
):
    """Generate plots comparing predictions with actual values for all datasets"""
    os.makedirs(save_dir, exist_ok=True)

    # Get predictions for each dataset
    y_train_pred = model.predict(X_train)[:, 0]  # Take only mean values, not variance
    y_val_pred = model.predict(X_val)[:, 0]
    y_test_pred = model.predict(X_test)[:, 0]

    # Scale back to original units
    y_train_orig = scaler_y.inverse_transform(y_train.reshape(-1, 1)).flatten()
    y_val_orig = scaler_y.inverse_transform(y_val.reshape(-1, 1)).flatten()
    y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

    y_train_pred_orig = scaler_y.inverse_transform(
        y_train_pred.reshape(-1, 1)
    ).flatten()
    y_val_pred_orig = scaler_y.inverse_transform(y_val_pred.reshape(-1, 1)).flatten()
    y_test_pred_orig = scaler_y.inverse_transform(y_test_pred.reshape(-1, 1)).flatten()

    # Calculate metrics
    train_r2 = r2_score(y_train_orig, y_train_pred_orig)
    val_r2 = r2_score(y_val_orig, y_val_pred_orig)
    test_r2 = r2_score(y_test_orig, y_test_pred_orig)

    train_rmse = np.sqrt(mean_squared_error(y_train_orig, y_train_pred_orig))
    val_rmse = np.sqrt(mean_squared_error(y_val_orig, y_val_pred_orig))
    test_rmse = np.sqrt(mean_squared_error(y_test_orig, y_test_pred_orig))

    # Create scatter plots for each dataset
    plt.figure(figsize=(18, 6))

    # Training set
    plt.subplot(1, 3, 1)
    plt.scatter(y_train_orig, y_train_pred_orig, alpha=0.5)
    plt.plot(
        [min(y_train_orig), max(y_train_orig)],
        [min(y_train_orig), max(y_train_orig)],
        "r--",
    )
    plt.xlabel("Actual DO")
    plt.ylabel("Predicted DO")
    plt.title(f"Training Set\nR² = {train_r2:.4f}, RMSE = {train_rmse:.4f}")
    plt.grid(alpha=0.3)

    # Validation set
    plt.subplot(1, 3, 2)
    plt.scatter(y_val_orig, y_val_pred_orig, alpha=0.5)
    plt.plot(
        [min(y_val_orig), max(y_val_orig)], [min(y_val_orig), max(y_val_orig)], "r--"
    )
    plt.xlabel("Actual DO")
    plt.ylabel("Predicted DO")
    plt.title(f"Validation Set\nR² = {val_r2:.4f}, RMSE = {val_rmse:.4f}")
    plt.grid(alpha=0.3)

    # Test set
    plt.subplot(1, 3, 3)
    plt.scatter(y_test_orig, y_test_pred_orig, alpha=0.5)
    plt.plot(
        [min(y_test_orig), max(y_test_orig)],
        [min(y_test_orig), max(y_test_orig)],
        "r--",
    )
    plt.xlabel("Actual DO")
    plt.ylabel("Predicted DO")
    plt.title(f"Test Set\nR² = {test_r2:.4f}, RMSE = {test_rmse:.4f}")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/prediction_scatter_plots.png", dpi=300)
    plt.close()

    # Plot residuals
    plt.figure(figsize=(18, 6))

    # Training residuals
    plt.subplot(1, 3, 1)
    train_residuals = y_train_orig - y_train_pred_orig
    plt.scatter(y_train_pred_orig, train_residuals, alpha=0.5)
    plt.axhline(y=0, color="r", linestyle="--")
    plt.xlabel("Predicted DO")
    plt.ylabel("Residuals")
    plt.title("Training Set Residuals")
    plt.grid(alpha=0.3)

    # Validation residuals
    plt.subplot(1, 3, 2)
    val_residuals = y_val_orig - y_val_pred_orig
    plt.scatter(y_val_pred_orig, val_residuals, alpha=0.5)
    plt.axhline(y=0, color="r", linestyle="--")
    plt.xlabel("Predicted DO")
    plt.ylabel("Residuals")
    plt.title("Validation Set Residuals")
    plt.grid(alpha=0.3)

    # Test residuals
    plt.subplot(1, 3, 3)
    test_residuals = y_test_orig - y_test_pred_orig
    plt.scatter(y_test_pred_orig, test_residuals, alpha=0.5)
    plt.axhline(y=0, color="r", linestyle="--")
    plt.xlabel("Predicted DO")
    plt.ylabel("Residuals")
    plt.title("Test Set Residuals")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/residual_plots.png", dpi=300)
    plt.close()

    return


def plot_uncertainty_analysis(model, X_test, y_test, scaler_y, save_dir="."):
    """Generate plots analyzing the model's uncertainty estimates"""
    os.makedirs(save_dir, exist_ok=True)

    # Get predictions with uncertainty
    test_preds = model.predict(X_test)
    means = test_preds[:, 0]
    log_vars = test_preds[:, 1]

    # Convert to original scale
    means_orig = scaler_y.inverse_transform(means.reshape(-1, 1)).flatten()
    y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # Calculate uncertainties
    vars_pred = np.exp(log_vars)
    stds_pred = np.sqrt(vars_pred)
    stds_orig = stds_pred * scaler_y.scale_[0]

    # Calculate absolute errors
    abs_errors = np.abs(y_test_orig - means_orig)

    # Calculate residuals (missing in the original code)
    test_residuals = y_test_orig - means_orig

    # Sort data by uncertainty for visualization
    sorted_indices = np.argsort(stds_orig)
    sorted_stds = stds_orig[sorted_indices]
    sorted_errors = abs_errors[sorted_indices]

    # Plot uncertainty vs error
    plt.figure(figsize=(10, 6))
    plt.scatter(stds_orig, abs_errors, alpha=0.5)
    plt.xlabel("Predicted Uncertainty (std)")
    plt.ylabel("Absolute Error")
    plt.title("Uncertainty vs Prediction Error")

    # Add trend line
    z = np.polyfit(stds_orig, abs_errors, 1)
    p = np.poly1d(z)
    plt.plot(
        np.sort(stds_orig),
        p(np.sort(stds_orig)),
        "r--",
        label=f"Trend: y={z[0]:.2f}x+{z[1]:.2f}",
    )

    # Calculate correlation
    corr = np.corrcoef(stds_orig, abs_errors)[0, 1]
    plt.legend(loc="upper left")
    plt.text(
        0.05,
        0.95,
        f"Correlation: {corr:.4f}",
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
    )

    plt.grid(alpha=0.3)
    plt.savefig(f"{save_dir}/uncertainty_vs_error.png", dpi=300)
    plt.close()

    # Plot calibration curve (reliability diagram)
    # Bin test examples by predicted uncertainty
    num_bins = 10
    bin_edges = np.percentile(stds_orig, np.linspace(0, 100, num_bins + 1))
    bin_indices = np.digitize(stds_orig, bin_edges)

    bin_mean_uncertainty = np.zeros(num_bins)
    bin_mean_error = np.zeros(num_bins)

    for i in range(1, num_bins + 1):
        bin_mask = bin_indices == i
        if np.any(bin_mask):
            bin_mean_uncertainty[i - 1] = np.mean(stds_orig[bin_mask])
            bin_mean_error[i - 1] = np.mean(abs_errors[bin_mask])

    plt.figure(figsize=(10, 6))
    plt.scatter(bin_mean_uncertainty, bin_mean_error, s=100)

    # Ideal calibration line
    max_val = max(np.max(bin_mean_uncertainty), np.max(bin_mean_error)) * 1.1
    plt.plot([0, max_val], [0, max_val], "r--", label="Ideal calibration")

    plt.xlabel("Mean Predicted Uncertainty")
    plt.ylabel("Mean Absolute Error")
    plt.title("Uncertainty Calibration Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.axis("equal")
    plt.savefig(f"{save_dir}/uncertainty_calibration.png", dpi=300)
    plt.close()

    # Plot error distribution
    plt.figure(figsize=(10, 6))
    plt.hist(test_residuals, bins=30, alpha=0.5, color="blue", edgecolor="black")
    plt.axvline(x=0, color="r", linestyle="--")
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.title("Error Distribution")
    plt.grid(alpha=0.3)
    plt.savefig(f"{save_dir}/error_distribution.png", dpi=300)
    plt.close()

    return


def plot_feature_importance(model, X_train, X_raw, feature_names, save_dir="."):
    """Generate plots for feature analysis"""
    os.makedirs(save_dir, exist_ok=True)

    # For feature importance, a simple approach is to use correlation with target
    plt.figure(figsize=(10, 6))

    # Create correlation matrix
    corr_matrix = np.corrcoef(X_raw.T)
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        xticklabels=feature_names,
        yticklabels=feature_names,
    )
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/feature_correlation.png", dpi=300)
    plt.close()

    return
