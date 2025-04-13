import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load your data
def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        # Rename columns to standardized headers
        column_mapping = {
            'chl_a_mg_L': 'chla',
            'water_temp_celcius': 'temp',
            'doxy_mg_L': 'do',
            'pH': 'ph'
        }
        df = df.rename(columns=column_mapping)
        print(f"Data loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def explore_data(df):
    print("\nData Overview:")
    print(df.head())
    
    print("\nBasic Statistics:")
    print(df.describe())
    
    print("\nChecking for missing values:")
    print(df.isnull().sum())

    # Correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.savefig('correlation_matrix.png')
    
    # Feature vs target plots
    features = ['chla', 'temp', 'do']
    plt.figure(figsize=(15, 5))
    for i, feature in enumerate(features, 1):
        plt.subplot(1, 3, i)
        plt.scatter(df[feature], df['ph'])
        plt.xlabel(feature)
        plt.ylabel('pH')
        plt.title(f'{feature} vs pH')
    plt.tight_layout()
    plt.savefig('feature_relationships.png')
    
    return

def preprocess_data(df, use_feature_engineering=True, poly_degree=2):
    """Preprocess data with optional feature engineering."""
    # Drop rows with missing values (or you could impute them)
    df_clean = df.dropna()
    
    # Extract basic features and target
    X_basic = df_clean[['chla', 'temp', 'do']]
    y = df_clean['ph']
    
    if use_feature_engineering:
        print("\nApplying feature engineering...")
        # Create interaction features directly on the dataframe
        df_features = create_interaction_features(df_clean)
        
        # Extract engineered features excluding the target
        feature_cols = [col for col in df_features.columns if col != 'ph']
        X = df_features[feature_cols]
        
        # Apply polynomial features
        X = create_polynomial_features(X, degree=poly_degree)
        print(f"Feature matrix shape after engineering: {X.shape}")
    else:
        X = X_basic
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, scaler, feature_cols if use_feature_engineering else ['chla', 'temp', 'do']

def create_polynomial_features(X, degree=2):
    """Create polynomial features to capture non-linear relationships."""
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    return poly.fit_transform(X)

def create_interaction_features(df):
    """Create interaction features between key water parameters."""
    df_features = df.copy()
    
    # Basic features
    basic_features = ['chla', 'temp', 'do']
    
    # Temperature and DO interaction (affects gas solubility)
    df_features['temp_do_interaction'] = df['temp'] * df['do']
    
    # Chlorophyll and temperature interaction (affects algal activity)
    df_features['chla_temp_interaction'] = df['chla'] * df['temp']
    
    # Chlorophyll and dissolved oxygen interaction (photosynthesis relationship)
    df_features['chla_do_interaction'] = df['chla'] * df['do']
    
    # Ratio features (important for biological and chemical processes)
    df_features['do_temp_ratio'] = df['do'] / (df['temp'] + 1e-5)  # Avoid division by zero
    df_features['chla_do_ratio'] = df['chla'] / (df['do'] + 1e-5)
    
    # Squared terms (to capture non-linear effects)
    for feature in basic_features:
        df_features[f'{feature}_squared'] = df[feature] ** 2
    
    # Log transformations (for skewed data)
    for feature in basic_features:
        # Add small constant to avoid log(0)
        df_features[f'{feature}_log'] = np.log(df[feature] + 1e-5)
    
    return df_features

def create_nn_model(input_dim):
    """Create a neural network model for regression."""
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=input_dim))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso(),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'ElasticNet': ElasticNet(random_state=42),
        'SVR': SVR(),
        'XGBoost': XGBRegressor(random_state=42),
        'LightGBM': LGBMRegressor(random_state=42),
        'CatBoost': CatBoostRegressor(random_state=42, verbose=0)
    }
    
    # Dictionary to store results
    results = {}
    
    print("\nModel Evaluation:")
    print("-" * 60)
    print("{:<20} {:<12} {:<12} {:<12}".format("Model", "RMSE", "MAE", "R²"))
    print("-" * 60)
    
    # Store training history for best models
    histories = {}
    
    # Train and evaluate regular models
    for name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Store results
        results[name] = {'model': model, 'rmse': rmse, 'mae': mae, 'r2': r2}
        
        print("{:<20} {:<12.4f} {:<12.4f} {:<12.4f}".format(name, rmse, mae, r2))
        
        # Add residual plots for each model
        plt.figure(figsize=(10, 6))
        residuals = y_test - y_pred
        plt.scatter(y_pred, residuals)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted pH')
        plt.ylabel('Residuals')
        plt.title(f'Residual Plot - {name}')
        plt.savefig(f'residuals_{name.replace(" ", "_")}.png')
        plt.close()
    
    # Train and evaluate neural network separately
    print("\nTraining Neural Network...")
    try:
        # Create and train neural network
        input_dim = X_train.shape[1]
        nn_model = create_nn_model(input_dim)
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        nn_model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=32,
            callbacks=[early_stop],
            verbose=0
        )
        
        # Make predictions
        y_pred_nn = nn_model.predict(X_test).flatten()
        
        # Calculate metrics
        rmse_nn = np.sqrt(mean_squared_error(y_test, y_pred_nn))
        mae_nn = mean_absolute_error(y_test, y_pred_nn)
        r2_nn = r2_score(y_test, y_pred_nn)
        
        # Store results
        results['Neural Network'] = {'model': nn_model, 'rmse': rmse_nn, 'mae': mae_nn, 'r2': r2_nn}
        
        print("{:<20} {:<12.4f} {:<12.4f} {:<12.4f}".format("Neural Network", rmse_nn, mae_nn, r2_nn))
        
        # Add learning curve plot for neural network
        if 'Neural Network' in results:
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(early_stop.model.history.history['loss'])
            plt.plot(early_stop.model.history.history['val_loss'])
            plt.title('Neural Network Learning Curves')
            plt.ylabel('Loss (MSE)')
            plt.xlabel('Epoch')
            plt.legend(['Training', 'Validation'], loc='upper right')
            
            plt.subplot(1, 2, 2)
            plt.plot(early_stop.model.history.history['mae'])
            plt.plot(early_stop.model.history.history['val_mae'])
            plt.title('Mean Absolute Error')
            plt.ylabel('MAE')
            plt.xlabel('Epoch')
            plt.legend(['Training', 'Validation'], loc='upper right')
            plt.tight_layout()
            plt.savefig('neural_network_learning_curves.png')
            plt.close()
    except Exception as e:
        print(f"Error training Neural Network: {e}")
    
    # Find the best model based on RMSE
    best_model_name = min(results, key=lambda x: results[x]['rmse'])
    print("\nBest model based on RMSE:", best_model_name)
    
    return results, best_model_name

def optimize_best_model(X_train, y_train, best_model_name):
    param_grids = {
        'Linear Regression': {},  # No hyperparameters to tune
        'Ridge Regression': {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]},
        'Lasso Regression': {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]},
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        },
        'Gradient Boosting': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        },
        'ElasticNet': {
            'alpha': [0.001, 0.01, 0.1, 1],
            'l1_ratio': [0.1, 0.5, 0.7, 0.9]
        },
        'SVR': {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto', 0.1, 1],
            'kernel': ['rbf', 'poly', 'sigmoid']
        },
        'XGBoost': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0]
        },
        'LightGBM': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'num_leaves': [31, 50, 100]
        },
        'CatBoost': {
            'iterations': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'depth': [4, 6, 8]
        },
        'Neural Network': {}  # Neural Network hyperparameters are tuned separately
    }
    
    # Update the models dictionary to include the new models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso(),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'ElasticNet': ElasticNet(random_state=42),
        'SVR': SVR(),
        'XGBoost': XGBRegressor(random_state=42),
        'LightGBM': LGBMRegressor(random_state=42),
        'CatBoost': CatBoostRegressor(random_state=42, verbose=0)
    }
    
    if best_model_name == 'Linear Regression':
        print("Linear Regression has no hyperparameters to tune. Skipping optimization.")
        return None
    elif best_model_name == 'Neural Network':
        print("Neural Network optimization performed separately.")
        return optimize_neural_network(X_train, y_train)
    
    print(f"\nOptimizing {best_model_name} hyperparameters...")
    grid_search = GridSearchCV(
        models[best_model_name],
        param_grids[best_model_name],
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    print("Best parameters:", grid_search.best_params_)
    print("Best score (negative MSE):", grid_search.best_score_)
    
    return grid_search.best_estimator_

def optimize_neural_network(X_train, y_train):
    """Optimize hyperparameters for a neural network model."""
    print("Optimizing Neural Network architecture...")
    
    # Split the training data for validation during optimization
    X_train_opt, X_val_opt, y_train_opt, y_val_opt = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Try different architectures
    architectures = [
        # [First hidden layer, Second hidden layer]
        [32, 16],
        [64, 32],
        [128, 64],
        [64, 32, 16]  # Three hidden layers
    ]
    
    dropout_rates = [0.1, 0.2, 0.3]
    learning_rates = [0.001, 0.01, 0.1]
    batch_sizes = [16, 32, 64]
    
    best_val_loss = float('inf')
    best_model = None
    best_config = {}
    
    for architecture in architectures:
        for dropout_rate in dropout_rates:
            for lr in learning_rates:
                for batch_size in batch_sizes:
                    # Create and compile model
                    model = Sequential()
                    model.add(Dense(architecture[0], activation='relu', input_dim=X_train.shape[1]))
                    model.add(Dropout(dropout_rate))
                    
                    for units in architecture[1:]:
                        model.add(Dense(units, activation='relu'))
                        model.add(Dropout(dropout_rate))
                    
                    model.add(Dense(1, activation='linear'))
                    
                    # Use specified learning rate
                    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
                    model.compile(optimizer=optimizer, loss='mse')
                    
                    # Train with early stopping
                    early_stop = EarlyStopping(
                        monitor='val_loss', patience=10, restore_best_weights=True
                    )
                    
                    history = model.fit(
                        X_train_opt, y_train_opt,
                        validation_data=(X_val_opt, y_val_opt),
                        epochs=100,
                        batch_size=batch_size,
                        callbacks=[early_stop],
                        verbose=0
                    )
                    
                    # Evaluate
                    val_loss = model.evaluate(X_val_opt, y_val_opt, verbose=0)
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model = model
                        best_config = {
                            'architecture': architecture,
                            'dropout_rate': dropout_rate,
                            'learning_rate': lr,
                            'batch_size': batch_size
                        }
    
    print("Best Neural Network configuration:", best_config)
    print(f"Best validation MSE: {best_val_loss:.4f}")
    
    # Train final model with best configuration on all training data
    final_model = Sequential()
    final_model.add(Dense(best_config['architecture'][0], activation='relu', input_dim=X_train.shape[1]))
    final_model.add(Dropout(best_config['dropout_rate']))
    
    for units in best_config['architecture'][1:]:
        final_model.add(Dense(units, activation='relu'))
        final_model.add(Dropout(best_config['dropout_rate']))
    
    final_model.add(Dense(1, activation='linear'))
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=best_config['learning_rate'])
    final_model.compile(optimizer=optimizer, loss='mse')
    
    # Train with early stopping
    early_stop = EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True
    )
    
    # Split off a small validation set just for early stopping
    X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42
    )
    
    final_model.fit(
        X_train_final, y_train_final,
        validation_data=(X_val_final, y_val_final),
        epochs=150,
        batch_size=best_config['batch_size'],
        callbacks=[early_stop],
        verbose=0
    )
    
    return final_model

def main():
    # Replace with your actual data file path
    df = load_data('./data/water_quality_data_for_pH.csv')  
    if df is None:
        return
    
    # Explore the data
    explore_data(df)
    
    # Run experiments with and without feature engineering
    print("\n" + "="*50)
    print("EXPERIMENT 1: BASELINE MODEL (NO FEATURE ENGINEERING)")
    print("="*50)
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(
        df, use_feature_engineering=False
    )
    
    # Train and evaluate baseline models
    results_baseline, best_model_name_baseline = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    print("\n" + "="*50)
    print("EXPERIMENT 2: WITH FEATURE ENGINEERING")
    print("="*50)
    X_train_eng, X_test_eng, y_train_eng, y_test_eng, scaler_eng, feature_names_eng = preprocess_data(
        df, use_feature_engineering=True, poly_degree=2
    )
    
    # Train and evaluate engineered models
    results_eng, best_model_name_eng = train_and_evaluate_models(X_train_eng, X_test_eng, y_train_eng, y_test_eng)
    
    # Decide which approach worked better
    best_rmse_baseline = results_baseline[best_model_name_baseline]['rmse']
    best_rmse_eng = results_eng[best_model_name_eng]['rmse']
    
    if best_rmse_eng < best_rmse_baseline:
        print("\nFeature engineering improved performance! Using engineered features.")
        best_model_name = best_model_name_eng
        X_train, X_test, y_train, y_test = X_train_eng, X_test_eng, y_train_eng, y_test_eng
        results = results_eng
        scaler_final = scaler_eng
        feature_names_final = feature_names_eng
    else:
        print("\nBaseline features performed better. Using original features.")
        best_model_name = best_model_name_baseline
        results = results_baseline
        scaler_final = scaler
        feature_names_final = feature_names
    
    # Optimize the best model
    optimized_model = optimize_best_model(X_train, y_train, best_model_name)
    
    if optimized_model is not None:
        final_model = optimized_model
    else:
        final_model = results[best_model_name]['model']
    
    # Final evaluation
    if best_model_name == 'Neural Network':
        y_pred = final_model.predict(X_test).flatten()
    else:
        y_pred = final_model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print("\nFinal model performance:")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    
    # Save the model, scaler, and feature names
    if best_model_name == 'Neural Network':
        final_model.save('ph_nn_model.h5')
        print("Neural network model saved as h5 file.")
    else:
        joblib.dump(final_model, 'ph_prediction_model.pkl')
        print("Model saved as pkl file.")
    
    joblib.dump(scaler_final, 'feature_scaler.pkl')
    joblib.dump(feature_names_final, 'feature_names.pkl')
    
    print("\nScaler and feature names saved.")
    
    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual pH')
    plt.ylabel('Predicted pH')
    plt.title('Actual vs Predicted pH Values')
    plt.savefig('prediction_results.png')
    
    # Feature importance visualization (for models that support it)
    try:
        if hasattr(final_model, 'feature_importances_'):
            plt.figure(figsize=(12, 6))
            
            # For polynomial features, use indices as labels
            if len(feature_names_final) != len(final_model.feature_importances_):
                feature_names_plot = [f"Feature_{i}" for i in range(len(final_model.feature_importances_))]
            else:
                feature_names_plot = feature_names_final
                
            importances = pd.Series(final_model.feature_importances_, index=feature_names_plot)
            importances.sort_values().plot(kind='barh')
            plt.title('Feature Importances')
            plt.tight_layout()
            plt.savefig('feature_importances.png')
            print("Feature importance plot created.")
    except Exception as e:
        print(f"Could not create feature importance plot: {e}")
    
    # Add comparison plot of model performances
    plt.figure(figsize=(12, 8))
    model_names = list(results.keys())
    rmse_values = [results[model]['rmse'] for model in model_names]
    r2_values = [results[model]['r2'] for model in model_names]
    
    plt.subplot(2, 1, 1)
    bars = plt.bar(model_names, rmse_values, color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('RMSE (lower is better)')
    plt.title('Model Comparison - RMSE')
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                 f"{rmse_values[i]:.3f}", ha='center')
    
    plt.subplot(2, 1, 2)
    bars = plt.bar(model_names, r2_values, color='lightgreen')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('R² (higher is better)')
    plt.title('Model Comparison - R²')
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                 f"{r2_values[i]:.3f}", ha='center')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()
    
    # Add scatter plot with uncertainty bands if applicable
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred, alpha=0.6, label='Test Predictions')
    
    # Add prediction vs actual with uncertainty band if appropriate model
    if best_model_name in ['Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM', 'CatBoost']:
        # Sort for cleaner visualization
        y_test_s, y_pred_s = zip(*sorted(zip(y_test, y_pred)))
        
        # Create prediction intervals (simple approximation)
        std_residuals = np.std(y_test - y_pred)
        plt.fill_between(sorted(y_test), 
                        [y - 2*std_residuals for y in y_pred_s],
                        [y + 2*std_residuals for y in y_pred_s],
                        alpha=0.2, label='95% Prediction Interval')
        
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label='Perfect Predictions')
    plt.xlabel('Actual pH')
    plt.ylabel('Predicted pH')
    plt.title(f'Actual vs Predicted pH Values ({best_model_name})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('prediction_results_with_uncertainty.png')
    plt.close()

if __name__ == "__main__":
    main()