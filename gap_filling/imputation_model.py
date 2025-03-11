import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import RobustScaler, StandardScaler

warnings.filterwarnings("ignore")


# Load the dataset
def load_data(file_path):
    try:
        # Try standard UTF-8 encoding first
        df = pd.read_csv(file_path)
    except UnicodeDecodeError:
        # If UTF-8 fails, try different encodings
        encodings = ["latin1", "ISO-8859-1", "cp1252"]
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                print(f"Successfully loaded with {encoding} encoding")
                break
            except Exception as e:
                if encoding == encodings[-1]:  # If we've tried all encodings
                    print(f"Failed to load file with available encodings: {str(e)}")
                    raise
                continue
    return df


# Preprocess data for autoencoder
def preprocess_data(df):
    # Select only these specific columns that need imputation
    target_cols = [
        "water_temp_celcius",
        "doxy_mg_L",
        "ammonia_umol_kg",
        "pH",
        "chl_a_mg_L",
    ]

    # Check which target columns exist in the dataframe
    numeric_cols = [col for col in target_cols if col in df.columns]

    if not numeric_cols:
        raise ValueError(
            "None of the specified columns for imputation exist in the dataset"
        )

    if len(numeric_cols) < len(target_cols):
        missing_cols = set(target_cols) - set(numeric_cols)
        print(
            f"Warning: The following columns specified for imputation are not in the dataset: {missing_cols}"
        )

    # Create a copy to avoid modifying original data
    data = df[numeric_cols].copy()

    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data.fillna(data.mean()))

    # Create mask for missing values (1 for missing, 0 for present)
    mask = data.isna().astype(np.float32).values

    # Create corrupted input (with placeholders for missing values)
    corrupted_data = data_scaled.copy()
    corrupted_data[mask == 1] = 0  # Replace missing with zeros after scaling

    return corrupted_data, data_scaled, mask, scaler, numeric_cols


# Add temporal feature extraction if timestamp is available
def extract_temporal_features(df):
    """Extract temporal features if timestamp column exists"""
    temporal_features = df.copy()

    # Look for datetime columns
    datetime_cols = []
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                # Attempt to parse as datetime
                pd.to_datetime(df[col])
                datetime_cols.append(col)
            except:
                pass

    # Process datetime columns
    for col in datetime_cols:
        try:
            # Convert to datetime
            temporal_features[col] = pd.to_datetime(df[col])

            # Extract features
            temporal_features[f"{col}_hour"] = temporal_features[col].dt.hour
            temporal_features[f"{col}_day"] = temporal_features[col].dt.day
            temporal_features[f"{col}_month"] = temporal_features[col].dt.month
            temporal_features[f"{col}_year"] = temporal_features[col].dt.year
            temporal_features[f"{col}_dayofweek"] = temporal_features[col].dt.dayofweek

            # Cyclical encoding for time features
            temporal_features[f"{col}_hour_sin"] = np.sin(
                2 * np.pi * temporal_features[f"{col}_hour"] / 24
            )
            temporal_features[f"{col}_hour_cos"] = np.cos(
                2 * np.pi * temporal_features[f"{col}_hour"] / 24
            )
            temporal_features[f"{col}_month_sin"] = np.sin(
                2 * np.pi * temporal_features[f"{col}_month"] / 12
            )
            temporal_features[f"{col}_month_cos"] = np.cos(
                2 * np.pi * temporal_features[f"{col}_month"] / 12
            )

            # Drop original column
            temporal_features = temporal_features.drop(col, axis=1)

            print(f"Extracted temporal features from {col}")
        except Exception as e:
            print(f"Could not process {col} as datetime: {str(e)}")

    return temporal_features


# Function to find correlations between features
def find_correlated_features(df, target_cols, threshold=0.3):
    """Find features correlated with target variables"""
    correlated_features = {}

    # Get all numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for target in target_cols:
        if target in numeric_cols:
            corr_features = []
            for col in numeric_cols:
                if col != target and not df[col].isna().all():
                    # Use Pearson correlation and handle missing values
                    mask = ~(df[target].isna() | df[col].isna())
                    if mask.sum() > 10:  # Need enough data points
                        corr, p_value = pearsonr(
                            df.loc[mask, target], df.loc[mask, col]
                        )
                        if abs(corr) > threshold and p_value < 0.05:
                            corr_features.append((col, abs(corr)))

            # Sort by correlation strength
            correlated_features[target] = sorted(
                corr_features, key=lambda x: x[1], reverse=True
            )

    return correlated_features


# Add function to detect and remove outliers
def remove_outliers(X, y, threshold=3.0):
    """
    Remove outliers from training data using z-score method

    Args:
        X: Features dataframe
        y: Target series
        threshold: Z-score threshold for outlier detection (default: 3.0)

    Returns:
        X_clean: Features dataframe without outliers
        y_clean: Target series without outliers
    """
    # Convert inputs to numpy arrays if they're not already
    X_np = X.values if hasattr(X, "values") else X
    y_np = y.values if hasattr(y, "values") else y

    # Calculate z-score for target variable
    y_mean = np.nanmean(y_np)
    y_std = np.nanstd(y_np)

    if y_std == 0 or np.isnan(
        y_std
    ):  # Handle case where all values are the same or std is NaN
        return X, y

    z_scores = np.abs((y_np - y_mean) / y_std)

    # Find indices of non-outliers
    non_outlier_indices = z_scores <= threshold

    # Print info about outliers
    n_outliers = np.sum(~non_outlier_indices)
    if n_outliers > 0:
        print(f"  Removed {n_outliers} outliers ({n_outliers/len(y_np):.1%} of data)")

    # Return filtered data
    if hasattr(X, "iloc"):  # If X is a DataFrame
        X_clean = X.iloc[non_outlier_indices]
    else:  # If X is a numpy array
        X_clean = X_np[non_outlier_indices]

    if hasattr(y, "iloc"):  # If y is a Series
        y_clean = y.iloc[non_outlier_indices]
    else:  # If y is a numpy array
        y_clean = y_np[non_outlier_indices]

    return X_clean, y_clean


class RobustImputationEnsemble:
    """Ensemble of multiple imputation methods without deep learning dependencies"""

    def __init__(
        self,
        methods=["knn", "iterative", "rf"],
        should_remove_outliers=True,
        outlier_threshold=3.0,
    ):
        self.methods = methods
        self.models = {}
        self.scalers = {}
        self.should_remove_outliers = should_remove_outliers
        self.outlier_threshold = outlier_threshold

    def fit_transform(self, df, cols_to_impute):
        """Fit multiple models and impute using an ensemble approach"""
        # Create a copy of the dataframe to store results
        imputed_df = df.copy()
        original_df = df.copy()

        # Extract temporal features if available
        print("\nExtracting temporal features...")
        df_with_features = extract_temporal_features(df)

        # Find correlated features for each target
        print("\nAnalyzing feature correlations...")
        corr_features = find_correlated_features(df_with_features, cols_to_impute)

        # Print correlation information
        for col, features in corr_features.items():
            if features:
                print(f"\nTop correlated features for {col}:")
                for feat, corr in features[:5]:  # Show top 5
                    print(f"  - {feat}: correlation = {corr:.3f}")
            else:
                print(f"\nNo strong correlations found for {col}")

        # Get masks for missing values
        missing_masks = {col: df[col].isna() for col in cols_to_impute}

        # Store method results
        method_results = {}

        # 1. Random Forest imputation for each column separately
        if "rf" in self.methods:
            print("\nFitting Random Forest models for each column...")

            rf_results = pd.DataFrame(index=df.index, columns=cols_to_impute)
            rf_models = {}

            for target_col in cols_to_impute:
                if not missing_masks[target_col].any():
                    continue

                print(f"  Training RF model for {target_col}")

                # Find best features for this target based on correlation
                best_features = [f[0] for f in corr_features.get(target_col, [])]

                # If no strong correlations, use all numeric features
                if not best_features:
                    best_features = [
                        col
                        for col in df_with_features.select_dtypes(
                            include=[np.number]
                        ).columns
                        if col != target_col and col not in cols_to_impute
                    ]

                # Add other non-missing target columns as features
                for other_col in cols_to_impute:
                    if other_col != target_col:
                        best_features.append(other_col)

                # Keep only features that exist in the dataframe
                features = [
                    col for col in best_features if col in df_with_features.columns
                ]

                if not features:
                    continue

                # Select rows where target is not missing for training
                train_mask = ~missing_masks[target_col]
                if train_mask.sum() < 10:
                    continue

                X_train = df_with_features.loc[train_mask, features]
                y_train = df.loc[train_mask, target_col]

                # Handle missing values in features
                X_train = X_train.fillna(X_train.mean())

                # Remove outliers if specified
                if (
                    self.should_remove_outliers and len(y_train) > 20
                ):  # Only remove outliers if enough data
                    X_train, y_train = remove_outliers(
                        X_train, y_train, threshold=self.outlier_threshold
                    )

                # Train model
                model = RandomForestRegressor(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,  # Use all available cores
                )
                model.fit(X_train, y_train)

                # Predict missing values
                missing_idx = missing_masks[target_col]
                if missing_idx.any():
                    X_pred = df_with_features.loc[missing_idx, features]
                    X_pred = X_pred.fillna(X_train.mean())
                    preds = model.predict(X_pred)
                    rf_results.loc[missing_idx, target_col] = preds

                rf_models[target_col] = {"model": model, "features": features}

            method_results["rf"] = rf_results
            self.models["rf"] = rf_models

        # 2. KNN imputation with feature selection
        if "knn" in self.methods:
            print("\nFitting KNN models for each column...")

            knn_results = pd.DataFrame(index=df.index, columns=cols_to_impute)
            knn_models = {}

            for target_col in cols_to_impute:
                if not missing_masks[target_col].any():
                    continue

                print(f"  Training KNN model for {target_col}")

                # Get best features for this column based on correlation
                best_features = [f[0] for f in corr_features.get(target_col, [])]

                # Use top correlated features or all numeric if none found
                if not best_features:
                    features_for_knn = [
                        col
                        for col in df.select_dtypes(include=[np.number]).columns
                        if col != target_col and col not in cols_to_impute
                    ]
                else:
                    # Use top 10 correlated features
                    features_for_knn = [f[0] for f in corr_features[target_col][:10]]

                # Add other target columns as features
                for other_col in cols_to_impute:
                    if other_col != target_col:
                        features_for_knn.append(other_col)

                # Keep only columns that exist
                features = [col for col in features_for_knn if col in df.columns]

                if not features:
                    continue

                # Get non-missing rows for training
                train_mask = ~missing_masks[target_col]
                if train_mask.sum() < 10:
                    continue

                # Prepare data
                X = df[features].copy()

                # Handle missing values in features
                X = X.fillna(X.mean())

                # Scale features for better KNN performance
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                X_train_scaled = X_scaled[train_mask]
                y_train = df.loc[train_mask, target_col]

                # Remove outliers if specified
                if self.should_remove_outliers and len(y_train) > 20:
                    # Create indices array for X_train_scaled
                    indices = np.arange(len(X_train_scaled))

                    # Use local variables to store results to avoid confusion with the function
                    X_cleaned = X_train_scaled.copy()
                    y_cleaned = y_train.copy()

                    # Only proceed with outlier removal if we have a valid Series/DataFrame
                    if hasattr(y_train, "iloc") or isinstance(y_train, np.ndarray):
                        # Get clean indices by using indices array and y_train
                        clean_indices, _ = remove_outliers(
                            indices, y_train, threshold=self.outlier_threshold
                        )

                        # Apply clean indices to both X and y
                        if len(clean_indices) > 0:  # Only if we have valid indices
                            X_cleaned = X_train_scaled[clean_indices]
                            if hasattr(y_train, "iloc"):
                                y_cleaned = y_train.iloc[clean_indices]
                            else:
                                y_cleaned = y_train[clean_indices]

                    # Use cleaned data for training
                    X_train_scaled = X_cleaned
                    y_train = y_cleaned

                # Train KNN model - adaptive neighborhood size based on data volume
                n_neighbors = min(20, max(3, int(len(y_train) * 0.1)))
                knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights="distance")
                knn.fit(X_train_scaled, y_train)

                # Predict missing values
                missing_idx = missing_masks[target_col]
                if missing_idx.any():
                    X_missing = X_scaled[missing_idx]
                    preds = knn.predict(X_missing)
                    knn_results.loc[missing_idx, target_col] = preds

                knn_models[target_col] = {
                    "model": knn,
                    "scaler": scaler,
                    "features": features,
                }

            method_results["knn"] = knn_results
            self.models["knn"] = knn_models

        # 3. Iterative imputation (MICE)
        if "iterative" in self.methods:
            print("\nFitting iterative imputer (MICE)...")

            # Use gradient boosting as estimator for MICE
            estimator = GradientBoostingRegressor(
                n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42
            )

            mice_imputer = IterativeImputer(
                estimator=estimator, max_iter=10, random_state=42, verbose=1
            )

            # Create a copy of the data with just the target columns
            mice_data = df[cols_to_impute].copy()

            # Check if we have enough data for MICE
            if not mice_data.dropna().empty:
                try:
                    # Fit and transform
                    mice_result_array = mice_imputer.fit_transform(mice_data)
                    mice_result = pd.DataFrame(
                        mice_result_array, columns=cols_to_impute, index=df.index
                    )

                    method_results["iterative"] = mice_result
                    self.models["iterative"] = mice_imputer
                except Exception as e:
                    print(f"MICE imputation failed: {str(e)}")
            else:
                print("Not enough data for MICE imputation")

        # 4. Gradient Boosting for each column
        if "gbm" in self.methods:
            print("\nFitting Gradient Boosting models for each column...")

            gbm_results = pd.DataFrame(index=df.index, columns=cols_to_impute)
            gbm_models = {}

            for target_col in cols_to_impute:
                if not missing_masks[target_col].any():
                    continue

                print(f"  Training GBM model for {target_col}")

                # Find best features based on correlation
                best_features = [f[0] for f in corr_features.get(target_col, [])]

                # If no strong correlations, use all numeric features
                if not best_features:
                    best_features = [
                        col
                        for col in df_with_features.select_dtypes(
                            include=[np.number]
                        ).columns
                        if col != target_col and col not in cols_to_impute
                    ]

                # Add other target columns as features
                for other_col in cols_to_impute:
                    if other_col != target_col:
                        best_features.append(other_col)

                # Keep only columns that exist
                features = [
                    col for col in best_features if col in df_with_features.columns
                ]

                if not features:
                    continue

                # Get training data
                train_mask = ~missing_masks[target_col]
                if train_mask.sum() < 10:
                    continue

                X_train = df_with_features.loc[train_mask, features]
                y_train = df.loc[train_mask, target_col]

                # Handle missing values in features
                X_train = X_train.fillna(X_train.mean())

                # Remove outliers if specified
                if self.should_remove_outliers and len(y_train) > 20:
                    X_train, y_train = remove_outliers(
                        X_train, y_train, threshold=self.outlier_threshold
                    )

                # Train model with careful tuning
                model = GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=3,
                    min_samples_split=5,
                    min_samples_leaf=5,
                    random_state=42,
                )

                model.fit(X_train, y_train)

                # Predict missing values
                missing_idx = missing_masks[target_col]
                if missing_idx.any():
                    X_pred = df_with_features.loc[missing_idx, features]
                    X_pred = X_pred.fillna(X_train.mean())
                    preds = model.predict(X_pred)
                    gbm_results.loc[missing_idx, target_col] = preds

                gbm_models[target_col] = {"model": model, "features": features}

            method_results["gbm"] = gbm_results
            self.models["gbm"] = gbm_models

        # Combine results with adaptive weighting
        print("\nCombining methods with adaptive weighting...")
        for target_col in cols_to_impute:
            missing_mask = missing_masks[target_col]
            if not missing_mask.any():
                continue

            # Get predictions from each method for this column
            col_preds = {}
            for method, result_df in method_results.items():
                if target_col in result_df.columns and not result_df[
                    target_col
                ].isna().any(axis=0):
                    col_preds[method] = result_df.loc[missing_mask, target_col].values

            if not col_preds:
                continue

            # Determine adaptive weights based on data characteristics
            weights = {}

            # Calculate missing ratio
            n_missing = missing_mask.sum()
            n_total = len(df)
            missing_ratio = n_missing / n_total

            # Assign weights based on missing ratio
            if missing_ratio > 0.5:  # Many missing values
                weights = {"rf": 0.4, "gbm": 0.3, "knn": 0.2, "iterative": 0.1}
            elif missing_ratio > 0.2:  # Moderate missing values
                weights = {"rf": 0.35, "gbm": 0.3, "knn": 0.25, "iterative": 0.1}
            else:  # Few missing values
                weights = {"rf": 0.3, "gbm": 0.2, "knn": 0.4, "iterative": 0.1}

            # Use only available methods
            available_methods = list(col_preds.keys())
            method_weights = {m: weights.get(m, 0.25) for m in available_methods}

            # Normalize weights
            weight_sum = sum(method_weights.values())
            normalized_weights = {m: w / weight_sum for m, w in method_weights.items()}

            # Print the weights being used
            print(
                f"  For {target_col}, using weights: {', '.join([f'{m}={w:.2f}' for m, w in normalized_weights.items()])}"
            )

            # Calculate weighted ensemble predictions
            ensemble_pred = np.zeros(shape=missing_mask.sum())

            for method, preds in col_preds.items():
                ensemble_pred += normalized_weights[method] * preds

            # Apply reasonable bounds based on existing data
            if not df[target_col].dropna().empty:
                non_missing = df.loc[~df[target_col].isna(), target_col]
                q01 = np.percentile(non_missing, 1)
                q99 = np.percentile(non_missing, 99)
                ensemble_pred = np.clip(ensemble_pred, q01, q99)

            # Round to 2 decimal places
            ensemble_pred = np.round(ensemble_pred, 2)

            # Update the imputed dataframe
            imputed_df.loc[missing_mask, target_col] = ensemble_pred

        return imputed_df


# Function to evaluate imputation performance with fixed NaN handling
def evaluate_imputation(original_df, imputed_df, cols_to_evaluate, save_path=None):
    # Create directory if it doesn't exist
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)

    # Create distribution plots for each imputed column
    for col in cols_to_evaluate:
        plt.figure(figsize=(12, 5))

        # Plot distributions
        plt.subplot(1, 2, 1)
        sns.kdeplot(
            original_df[col].dropna(),
            label="Original (non-missing)",
            fill=True,
            alpha=0.5,
        )

        # Check if there are imputed values to plot
        if (
            original_df[col].isna().any()
            and not imputed_df.loc[original_df[col].isna(), col].isna().all()
        ):
            sns.kdeplot(
                imputed_df.loc[original_df[col].isna(), col].dropna(),
                label="Imputed values",
                fill=True,
                alpha=0.5,
            )
        else:
            print(f"No imputed values available for {col}")

        plt.title(f"Distribution Comparison for {col}")
        plt.xlabel(col)
        plt.ylabel("Density")
        plt.legend()

        # Plot scatter of original vs imputed if we have enough data
        plt.subplot(1, 2, 2)

        # For testing accuracy, we'll randomly mask 10% of non-missing values and compare
        non_missing_mask = ~original_df[col].isna()
        if non_missing_mask.sum() < 20:
            plt.text(
                0.5,
                0.5,
                "Not enough non-missing data for validation",
                ha="center",
                va="center",
                transform=plt.gca().transAxes,
            )
        else:
            # Set random seed for reproducibility
            np.random.seed(42)
            test_mask = non_missing_mask & (np.random.rand(len(original_df)) < 0.1)

            if test_mask.sum() > 10:  # Only if we have enough test points
                # Store true values before masking
                true_values = original_df.loc[test_mask, col].copy()

                # Create a copy of the dataframe to avoid modifying the original
                test_df = original_df.copy()
                test_df.loc[test_mask, col] = np.nan

                try:
                    # Use a simplified version of the imputation process to avoid recursion issues
                    test_ensemble = RobustImputationEnsemble(
                        methods=["rf", "knn", "gbm"],
                        should_remove_outliers=False,  # Disable outlier removal for test
                    )
                    test_imputed = test_ensemble.fit_transform(test_df, [col])
                    test_imputed_values = test_imputed.loc[test_mask, col].values

                    # Make sure there are no NaN values in the results
                    valid_indices = ~np.isnan(test_imputed_values)
                    if np.any(valid_indices):
                        true_subset = true_values.values[valid_indices]
                        imputed_subset = test_imputed_values[valid_indices]

                        plt.scatter(true_subset, imputed_subset, alpha=0.6)

                        # Add diagonal perfect prediction line
                        min_val = min(np.min(true_subset), np.min(imputed_subset))
                        max_val = max(np.max(true_subset), np.max(imputed_subset))
                        plt.plot([min_val, max_val], [min_val, max_val], "r--")

                        plt.title(f"Imputation Accuracy for {col}")
                        plt.xlabel("Actual Values")
                        plt.ylabel("Imputed Values")

                        # Calculate and display metrics
                        mse = np.mean((true_subset - imputed_subset) ** 2)
                        r2 = r2_score(true_subset, imputed_subset)
                        plt.text(
                            0.05,
                            0.95,
                            f"MSE: {mse:.4f}\nR²: {r2:.4f}",
                            transform=plt.gca().transAxes,
                            verticalalignment="top",
                        )
                    else:
                        plt.text(
                            0.5,
                            0.5,
                            "All imputed values were NaN",
                            ha="center",
                            va="center",
                            transform=plt.gca().transAxes,
                        )
                except Exception as e:
                    print(f"Error in evaluation for {col}: {str(e)}")
                    plt.text(
                        0.5,
                        0.5,
                        f"Error in evaluation",
                        ha="center",
                        va="center",
                        transform=plt.gca().transAxes,
                    )
            else:
                plt.text(
                    0.5,
                    0.5,
                    "Not enough test points for validation",
                    ha="center",
                    va="center",
                    transform=plt.gca().transAxes,
                )

        plt.tight_layout()

        if save_path:
            plt.savefig(os.path.join(save_path, f"imputation_evaluation_{col}.png"))
            print(
                f"Imputation evaluation plot for {col} saved to {os.path.join(save_path, f'imputation_evaluation_{col}.png')}"
            )

        plt.show()


# Main imputation function
def impute_missing_values(
    df, file_path_to_save=None, should_remove_outliers=True, outlier_threshold=3.0
):
    """Robust imputation function using traditional ML methods"""
    # Select target columns
    target_cols = [
        "water_temp_celcius",
        "doxy_mg_L",
        "ammonia_umol_kg",
        "pH",
        "chl_a_mg_L",
    ]
    available_targets = [col for col in target_cols if col in df.columns]

    if not available_targets:
        raise ValueError(
            "None of the specified columns for imputation exist in the dataset"
        )

    print("Using robust ML-based ensemble imputation approach...")
    if should_remove_outliers:
        print(f"Outlier removal enabled (threshold: {outlier_threshold})")

    # Create and apply the imputation ensemble
    ensemble = RobustImputationEnsemble(
        methods=["rf", "knn", "iterative", "gbm"],
        should_remove_outliers=should_remove_outliers,
        outlier_threshold=outlier_threshold,
    )
    imputed_df = ensemble.fit_transform(df, available_targets)

    # Save to file if path is provided
    if file_path_to_save:
        imputed_df.to_csv(file_path_to_save, index=False)

    return imputed_df, ensemble, None  # No training history for traditional ML


import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# New function to generate model performance plots
def plot_model_performance(df, imputed_df, ensemble, cols_to_evaluate, save_path=None):
    """Generate plots showing performance of individual models and the ensemble"""

    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)

    for col in cols_to_evaluate:
        # Skip if no missing values
        if not df[col].isna().any():
            continue

        # Create a test set by artificially masking 10% of the non-missing values
        test_df = df.copy()
        non_missing_mask = ~df[col].isna()
        test_indices = non_missing_mask & (np.random.rand(len(df)) < 0.1)

        if test_indices.sum() < 10:
            print(f"Not enough data to evaluate {col} performance")
            continue

        # Save the true values before masking
        true_values = df.loc[test_indices, col].values

        # Mask these values
        test_df.loc[test_indices, col] = np.nan

        # Dictionary to store predictions from each method
        method_predictions = {}

        # Get predictions from each method and the ensemble
        methods = ensemble.methods
        available_models = ensemble.models

        try:
            # RF predictions if available
            if "rf" in available_models and col in available_models["rf"]:
                try:
                    rf_model = available_models["rf"][col]["model"]
                    features = available_models["rf"][col]["features"]

                    # Extract features with temporal information if available
                    test_features = extract_temporal_features(test_df)

                    # Prepare input data
                    X_test = test_features.loc[test_indices, features]
                    X_test = X_test.fillna(X_test.mean())

                    # Get predictions
                    rf_preds = rf_model.predict(X_test)

                    # Make sure there are no NaN values
                    if not np.isnan(rf_preds).any():
                        method_predictions["Random Forest"] = rf_preds
                    else:
                        print(f"RF predictions contain NaN values for {col}")
                except Exception as e:
                    print(f"Error getting RF predictions for {col}: {str(e)}")

            # KNN predictions if available
            if "knn" in available_models and col in available_models.get("knn", {}):
                try:
                    knn_info = available_models["knn"][col]
                    knn_model = knn_info["model"]
                    knn_scaler = knn_info["scaler"]
                    knn_features = knn_info["features"]

                    # Prepare input data
                    X_test = test_df.loc[test_indices, knn_features]
                    X_test = X_test.fillna(X_test.mean())
                    X_test_scaled = knn_scaler.transform(X_test)

                    # Get predictions
                    knn_preds = knn_model.predict(X_test_scaled)

                    # Check for NaN values
                    if not np.isnan(knn_preds).any():
                        method_predictions["KNN"] = knn_preds
                    else:
                        print(f"KNN predictions contain NaN values for {col}")
                except Exception as e:
                    print(f"Error getting KNN predictions for {col}: {str(e)}")

            # GBM predictions if available
            if "gbm" in available_models and col in available_models.get("gbm", {}):
                try:
                    gbm_model = available_models["gbm"][col]["model"]
                    features = available_models["gbm"][col]["features"]

                    # Extract features with temporal information
                    test_features = extract_temporal_features(test_df)

                    # Prepare input data
                    X_test = test_features.loc[test_indices, features]
                    X_test = X_test.fillna(X_test.mean())

                    # Get predictions
                    gbm_preds = gbm_model.predict(X_test)

                    # Check for NaN values
                    if not np.isnan(gbm_preds).any():
                        method_predictions["Gradient Boosting"] = gbm_preds
                    else:
                        print(f"GBM predictions contain NaN values for {col}")
                except Exception as e:
                    print(f"Error getting GBM predictions for {col}: {str(e)}")

            # Iterative imputation predictions (MICE)
            if "iterative" in available_models:
                try:
                    # Create a copy with just the target columns to use with MICE
                    mice_df = test_df[cols_to_evaluate].copy()

                    # Fill NaN values with mean for MICE
                    for target_col in cols_to_evaluate:
                        if target_col != col:  # Don't fill the target column
                            mice_df[target_col] = mice_df[target_col].fillna(
                                mice_df[target_col].mean()
                            )

                    # Impute using MICE
                    mice_imputer = available_models["iterative"]
                    mice_imputed = mice_imputer.transform(mice_df)

                    # Convert back to dataframe
                    mice_result = pd.DataFrame(
                        mice_imputed, columns=cols_to_evaluate, index=test_df.index
                    )

                    # Get predictions
                    mice_preds = mice_result.loc[test_indices, col].values

                    # Check for NaN values
                    if not np.isnan(mice_preds).any():
                        method_predictions["MICE"] = mice_preds
                    else:
                        print(f"MICE predictions contain NaN values for {col}")
                except Exception as e:
                    print(f"Error getting MICE predictions for {col}: {str(e)}")

            # Get ensemble predictions
            try:
                # Use a simplified ensemble approach for test data
                test_ensemble = RobustImputationEnsemble(
                    methods=[m for m in methods if m in available_models]
                )
                test_imputed_df = test_ensemble.fit_transform(test_df, cols_to_evaluate)
                ensemble_preds = test_imputed_df.loc[test_indices, col].values

                # Check for NaN values
                if not np.isnan(ensemble_preds).any():
                    method_predictions["Ensemble"] = ensemble_preds
                else:
                    print(f"Ensemble predictions contain NaN values for {col}")
            except Exception as e:
                print(f"Error getting ensemble predictions for {col}: {str(e)}")

            # Skip if no valid predictions were obtained
            if not method_predictions:
                print(
                    f"No valid predictions available for {col}. Skipping performance plot."
                )
                continue

            # Create performance plot
            fig = plt.figure(figsize=(18, 12))
            gs = GridSpec(2, 3, figure=fig)

            # 1. Scatter plot for each method vs true values
            ax1 = fig.add_subplot(gs[0, :])

            # Setup for metrics table
            methods = list(method_predictions.keys())
            metrics = {"MAE": [], "RMSE": [], "R²": []}

            # Plot each method with different color and calculate metrics
            markers = ["o", "s", "D", "^", "*"]
            for i, (method_name, preds) in enumerate(method_predictions.items()):
                ax1.scatter(
                    true_values,
                    preds,
                    alpha=0.6,
                    marker=markers[i % len(markers)],
                    label=f"{method_name}",
                )

                # Calculate metrics
                mae = mean_absolute_error(true_values, preds)
                rmse = np.sqrt(mean_squared_error(true_values, preds))
                r2 = r2_score(true_values, preds)

                metrics["MAE"].append(f"{mae:.4f}")
                metrics["RMSE"].append(f"{rmse:.4f}")
                metrics["R²"].append(f"{r2:.4f}")

            # Add perfect prediction line
            all_values = np.concatenate(
                [true_values] + [p for p in method_predictions.values()]
            )
            min_val = np.nanmin(all_values)
            max_val = np.nanmax(all_values)
            ax1.plot([min_val, max_val], [min_val, max_val], "k--")

            ax1.set_title(f"Model Predictions vs True Values for {col}", fontsize=14)
            ax1.set_xlabel("True Values", fontsize=12)
            ax1.set_ylabel("Predicted Values", fontsize=12)
            ax1.legend(fontsize=12)
            ax1.grid(True, alpha=0.3)

            # 2. Error distribution plot
            ax2 = fig.add_subplot(gs[1, 0])

            for method_name, preds in method_predictions.items():
                errors = preds - true_values
                sns.kdeplot(errors, ax=ax2, label=method_name)

            ax2.set_title("Error Distribution", fontsize=14)
            ax2.set_xlabel("Prediction Error", fontsize=12)
            ax2.axvline(0, color="k", linestyle="--")
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=10)

            # 3. Method comparison bar chart
            ax3 = fig.add_subplot(gs[1, 1])

            # Create metrics table
            cell_text = []
            for method in methods:
                idx = methods.index(method)
                cell_text.append(
                    [metrics["MAE"][idx], metrics["RMSE"][idx], metrics["R²"][idx]]
                )

            table = ax3.table(
                cellText=cell_text,
                rowLabels=methods,
                colLabels=list(metrics.keys()),
                loc="center",
                cellLoc="center",
            )

            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1, 1.5)
            ax3.set_title("Performance Metrics Comparison", fontsize=14)
            ax3.axis("off")

            # 4. Feature importance plot (for Random Forest or GBM)
            ax4 = fig.add_subplot(gs[1, 2])

            show_feature_importance = False

            if (
                "Random Forest" in method_predictions
                and "rf" in available_models
                and col in available_models["rf"]
            ):
                try:
                    model = available_models["rf"][col]["model"]
                    features = available_models["rf"][col]["features"]

                    # Get feature importance
                    importance = model.feature_importances_

                    # Sort by importance
                    indices = np.argsort(importance)[::-1]
                    top_indices = indices[
                        : min(10, len(features))
                    ]  # Show top 10 features or all if fewer

                    if len(top_indices) > 0:
                        # Bar chart of feature importance
                        ax4.barh(range(len(top_indices)), importance[top_indices])
                        ax4.set_yticks(range(len(top_indices)))
                        ax4.set_yticklabels(
                            [
                                (
                                    features[i].split("_")[-1]
                                    if len(features[i]) > 15
                                    else features[i]
                                )
                                for i in top_indices
                            ]
                        )
                        ax4.set_title("Top Feature Importance", fontsize=14)
                        ax4.set_xlabel("Importance", fontsize=12)
                        show_feature_importance = True
                except Exception as e:
                    print(f"Error displaying RF feature importance for {col}: {str(e)}")

            elif (
                "Gradient Boosting" in method_predictions
                and "gbm" in available_models
                and col in available_models["gbm"]
            ):
                try:
                    model = available_models["gbm"][col]["model"]
                    features = available_models["gbm"][col]["features"]

                    # Get feature importance
                    importance = model.feature_importances_

                    # Sort by importance
                    indices = np.argsort(importance)[::-1]
                    top_indices = indices[
                        : min(10, len(features))
                    ]  # Show top 10 features or all if fewer

                    if len(top_indices) > 0:
                        # Bar chart of feature importance
                        ax4.barh(range(len(top_indices)), importance[top_indices])
                        ax4.set_yticks(range(len(top_indices)))
                        ax4.set_yticklabels(
                            [
                                (
                                    features[i].split("_")[-1]
                                    if len(features[i]) > 15
                                    else features[i]
                                )
                                for i in top_indices
                            ]
                        )
                        ax4.set_title("Top Feature Importance", fontsize=14)
                        ax4.set_xlabel("Importance", fontsize=12)
                        show_feature_importance = True
                except Exception as e:
                    print(
                        f"Error displaying GBM feature importance for {col}: {str(e)}"
                    )

            if not show_feature_importance:
                ax4.text(
                    0.5,
                    0.5,
                    "Feature importance not available",
                    ha="center",
                    va="center",
                    transform=ax4.transAxes,
                )
                ax4.axis("off")

            plt.tight_layout()

            if save_path:
                plt.savefig(os.path.join(save_path, f"model_performance_{col}.png"))
                print(
                    f"Model performance plot for {col} saved to {os.path.join(save_path, f'model_performance_{col}.png')}"
                )

            plt.show()

        except Exception as e:
            print(f"Error creating performance plot for {col}: {str(e)}")
            import traceback

            traceback.print_exc()


# Main execution
if __name__ == "__main__":
    data_path = "../data/insitu_wq_data.csv"
    output_path = "imputed_water_quality_data.csv"
    plots_dir = "imputation_plots"

    # Set outlier parameters
    should_remove_outliers = True  # Set to False to disable outlier removal
    outlier_threshold = (
        2.5  # Z-score threshold (lower = more aggressive outlier removal)
    )

    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    df = load_data(data_path)

    # Use the robust ML-based ensemble approach with outlier removal
    print("Using robust ML-based ensemble imputation...")
    imputed_df, model, _ = impute_missing_values(
        df,
        output_path,
        should_remove_outliers=should_remove_outliers,
        outlier_threshold=outlier_threshold,
    )

    # Define our target columns
    target_cols = [
        "water_temp_celcius",
        "doxy_mg_L",
        "ammonia_umol_kg",
        "pH",
        "chl_a_mg_L",
    ]
    available_targets = [col for col in target_cols if col in df.columns]

    # Print summary of imputation
    print("\nImputation summary for target columns:")
    for col in available_targets:
        before = df[col].isna().sum()
        after = imputed_df[col].isna().sum()
        total = len(df[col])
        print(
            f"{col}: {before} missing values before ({before/total:.1%}), {after} missing after ({after/total:.1%})"
        )

    # Evaluate imputation for our target columns
    evaluate_imputation(df, imputed_df, available_targets, save_path=plots_dir)

    # Generate detailed model performance plots
    print("\nGenerating model performance plots...")
    plot_model_performance(
        df, imputed_df, model, available_targets, save_path=plots_dir
    )

    print(f"\nAll plots saved to {os.path.abspath(plots_dir)}")
