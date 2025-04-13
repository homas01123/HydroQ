import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import pickle
import joblib
import os

from clean_data import clean_data

def train_do_estimator(data_path='./data/water_quality_data_for_DO.csv', model_dir='./models'):
    """
    Train and evaluate DO estimation models using water temperature and chlorophyll-a.
    
    Args:
        data_path: Path to the input data CSV
        model_dir: Directory to save trained models
    
    Returns:
        Dictionary containing model evaluation results
    """
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Clean data
    print("Cleaning data...")
    data = clean_data(data_path)
    
    # Select features and target
    print("Preparing features and target...")
    X = data[['water_temp_celcius', 'chl_a_mg_L']].copy()
    y = data['doxy_mg_L'].copy()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    # Define models to evaluate
    models = {
        'Linear Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('model', LinearRegression())
        ]),
        'Polynomial Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2)),
            ('model', LinearRegression())
        ]),
        'Ridge Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('model', Ridge(alpha=0.1))
        ]),
        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestRegressor(n_estimators=100, random_state=42))
        ]),
        'Gradient Boosting': Pipeline([
            ('scaler', StandardScaler()),
            ('model', GradientBoostingRegressor(n_estimators=100, random_state=42))
        ])
    }
    
    # Train and evaluate models
    results = {}
    
    print("\nTraining and evaluating models:")
    print("-" * 40)
    
    for name, pipeline in models.items():
        print(f"\nTraining {name}...")
        
        # Train
        pipeline.fit(X_train, y_train)
        
        # Predict
        y_pred = pipeline.predict(X_test)
        
        # Evaluate
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'Model': pipeline
        }
        
        print(f"  MSE: {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")
        
        # 5-fold cross-validation
        cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')
        print(f"  Cross-validation R² scores: {cv_scores}")
        print(f"  Mean CV R²: {cv_scores.mean():.4f}")
        
        # Save model
        model_path = os.path.join(model_dir, f"{name.lower().replace(' ', '_')}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(pipeline, f)
        print(f"  Model saved to {model_path}")
    
    # Identify best model
    best_model_name = max(results, key=lambda k: results[k]['R²'])
    best_model = results[best_model_name]['Model']
    
    print("\n" + "=" * 40)
    print(f"Best model: {best_model_name}")
    print(f"R² score: {results[best_model_name]['R²']:.4f}")
    print("=" * 40)
    
    # Feature importance for tree-based models
    if hasattr(best_model.named_steps['model'], 'feature_importances_'):
        importances = best_model.named_steps['model'].feature_importances_
        for i, feature in enumerate(X.columns):
            print(f"{feature}: {importances[i]:.4f}")
    
    # Visualize actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, best_model.predict(X_test), alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual DO (mg/L)')
    plt.ylabel('Predicted DO (mg/L)')
    plt.title(f'Actual vs Predicted DO using {best_model_name}')
    plt.savefig(os.path.join(model_dir, 'do_prediction_results.png'))
    print(f"Saved visualization to {os.path.join(model_dir, 'do_prediction_results.png')}")
    
    # Save best model separately
    best_model_path = os.path.join(model_dir, 'best_do_model.pkl')
    with open(best_model_path, 'wb') as f:
        pickle.dump({
            'model': best_model,
            'name': best_model_name,
            'metrics': results[best_model_name]
        }, f)
    print(f"Best model saved to {best_model_path}")
    
    return results

def load_model(model_path='./models/best_do_model.pkl'):
    """Load a saved DO estimation model"""
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data

def predict_do(model, water_temp, chl_a):
    """
    Predict dissolved oxygen using the trained model
    
    Args:
        model: Trained model (Pipeline)
        water_temp: Water temperature in Celsius
        chl_a: Chlorophyll-a concentration in mg/L
        
    Returns:
        Predicted DO in mg/L
    """
    input_data = np.array([[water_temp, chl_a]])
    return model.predict(input_data)[0]

if __name__ == "__main__":
    train_do_estimator()