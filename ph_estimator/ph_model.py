import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
import joblib
import os

# Create directory for outputs if it doesn't exist
os.makedirs('outputs', exist_ok=True)

# Load the dataset
data = pd.read_csv('data/water_quality_data_for_pH.csv')

# Display basic information about the dataset
print("Dataset shape:", data.shape)
print("\nFirst few rows:")
print(data.head())
print("\nSummary statistics:")
print(data.describe())

# Check for missing values
missing_values = data.isnull().sum()
print("\nMissing values:")
print(missing_values)

# Data visualization
plt.figure(figsize=(15, 10))
for i, column in enumerate(data.columns):
    plt.subplot(2, 2, i+1)
    sns.histplot(data[column], kde=True)
    plt.title(f'Distribution of {column}')
plt.tight_layout()
plt.savefig('outputs/feature_distributions.png')
plt.show()

# Correlation analysis
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.savefig('outputs/correlation_matrix.png')
plt.show()

# Pairplot for visualizing relationships between variables
sns.pairplot(data, diag_kind='kde')
plt.savefig('outputs/pairplot.png')
plt.show()

# Define features and target
X = data[['chl_a_mg_L', 'water_temp_celcius', 'doxy_mg_L']]
y = data['pH']

# Create additional engineered features
print("\nEngineering additional features...")
X['temp_doxy_ratio'] = X['water_temp_celcius'] / X['doxy_mg_L']
X['chl_a_doxy_ratio'] = X['chl_a_mg_L'] / X['doxy_mg_L']
X['chl_a_temp_ratio'] = X['chl_a_mg_L'] / X['water_temp_celcius']
X['log_chl_a'] = np.log1p(X['chl_a_mg_L'])  # log1p to handle small values better
X['temp_squared'] = X['water_temp_celcius'] ** 2
X['doxy_squared'] = X['doxy_mg_L'] ** 2

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")
print(f"Total features after engineering: {X.shape[1]}")

# Create a pipeline with features, scaling, and ridge regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)),
    ('select', SelectKBest(f_regression, k='all')),  # We'll tune k
    ('ridge', Ridge(alpha=1.0))
])

# Define hyperparameters to tune
param_grid = {
    'poly__degree': [1, 2],  # Test with and without polynomial features
    'select__k': [5, 10, 15, 'all'],  # Select top k features
    'ridge__alpha': [0.001, 0.01, 0.1, 1, 10, 100]
}

print("\nPerforming grid search to find optimal parameters...")
# Grid search with cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1)
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
print(f"\nBest parameters: {best_params}")

# Make predictions on test set
y_pred = best_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# Plot predicted vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual pH')
plt.ylabel('Predicted pH')
plt.title('Actual vs Predicted pH Values')
plt.grid(True)
plt.savefig('outputs/actual_vs_predicted.png')
plt.show()

# Residual plot
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.7)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted pH')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.grid(True)
plt.savefig('outputs/residual_plot.png')
plt.show()

# Try to extract feature importance if possible
try:
    # Get all feature names after polynomial transformation
    if best_params['poly__degree'] > 1:
        poly = best_model.named_steps['poly']
        feature_names = poly.get_feature_names_out(X.columns)
    else:
        feature_names = X.columns
    
    # Get selected features
    if best_params['select__k'] != 'all':
        select = best_model.named_steps['select']
        selected_indices = select.get_support(indices=True)
        selected_features = [feature_names[i] for i in selected_indices]
    else:
        selected_features = feature_names
    
    # Get coefficients
    ridge = best_model.named_steps['ridge']
    coefficients = ridge.coef_
    
    # Create dataframe for feature importance
    if len(selected_features) == len(coefficients):
        feature_importance = pd.DataFrame({
            'Feature': selected_features,
            'Coefficient': coefficients
        }).sort_values(by='Coefficient', key=abs, ascending=False)
        
        print("\nTop 10 Feature Importance:")
        print(feature_importance.head(10))
        
        # Plot feature importance for top features
        plt.figure(figsize=(12, 8))
        top_features = feature_importance.head(10)
        sns.barplot(x='Coefficient', y='Feature', data=top_features)
        plt.title('Feature Importance (Top 10 Features)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('outputs/feature_importance.png')
        plt.show()
except Exception as e:
    print(f"Couldn't extract feature importance: {e}")
    
# Define function for new predictions
def predict_ph(chl_a, water_temp, doxy, model=best_model):
    """
    Predict pH based on input features using the trained model.
    
    Parameters:
    - chl_a: Chlorophyll-a concentration in mg/L
    - water_temp: Water temperature in Celsius
    - doxy: Dissolved oxygen in mg/L
    - model: Trained ridge regression model
    
    Returns:
    - Predicted pH value
    """
    # Create the same features as used in training
    input_data = pd.DataFrame({
        'chl_a_mg_L': [chl_a],
        'water_temp_celcius': [water_temp],
        'doxy_mg_L': [doxy],
    })
    
    # Add the engineered features
    input_data['temp_doxy_ratio'] = input_data['water_temp_celcius'] / input_data['doxy_mg_L']
    input_data['chl_a_doxy_ratio'] = input_data['chl_a_mg_L'] / input_data['doxy_mg_L']
    input_data['chl_a_temp_ratio'] = input_data['chl_a_mg_L'] / input_data['water_temp_celcius']
    input_data['log_chl_a'] = np.log1p(input_data['chl_a_mg_L'])
    input_data['temp_squared'] = input_data['water_temp_celcius'] ** 2
    input_data['doxy_squared'] = input_data['doxy_mg_L'] ** 2
    
    return model.predict(input_data)[0]

# Example usage
print("\nExample prediction:")
sample_chl_a = 0.01
sample_temp = 15.0
sample_doxy = 9.5
sample_ph = predict_ph(sample_chl_a, sample_temp, sample_doxy)
print(f"Input: chl_a={sample_chl_a} mg/L, temp={sample_temp}°C, doxy={sample_doxy} mg/L")
print(f"Predicted pH: {sample_ph:.2f}")

# Save the model for future use
joblib.dump(best_model, 'outputs/ridge_ph_model.pkl')
print("\nModel saved as 'outputs/ridge_ph_model.pkl'")

# Perform comparison between original and enhanced model
print("\nComparing basic model with enhanced feature model:")
# Create basic pipeline without polynomial features or feature selection
basic_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', Ridge(alpha=best_params['ridge__alpha']))
])

# Fit basic model
basic_pipeline.fit(X_train[['chl_a_mg_L', 'water_temp_celcius', 'doxy_mg_L']], y_train)
basic_pred = basic_pipeline.predict(X_test[['chl_a_mg_L', 'water_temp_celcius', 'doxy_mg_L']])
basic_r2 = r2_score(y_test, basic_pred)
basic_rmse = np.sqrt(mean_squared_error(y_test, basic_pred))

print(f"Basic model R²: {basic_r2:.4f}, RMSE: {basic_rmse:.4f}")
print(f"Enhanced model R²: {r2:.4f}, RMSE: {rmse:.4f}")
print(f"Improvement: R² +{r2-basic_r2:.4f}, RMSE -{basic_rmse-rmse:.4f}")