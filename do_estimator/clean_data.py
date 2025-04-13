import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def clean_data(input_path='./data/water_quality_data_for_DO.csv', output_path=None):
    """
    Clean the water quality data for DO estimation modeling.
    
    Args:
        input_path: Path to the raw data CSV file
        output_path: Optional path to save cleaned data
    
    Returns:
        Cleaned pandas DataFrame
    """
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Basic info
    print(f"Dataset shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")
    
    # Remove rows with missing DO (target variable)
    df_clean = df.dropna(subset=['doxy_mg_L'])
    print(f"Removed {len(df) - len(df_clean)} rows with missing DO values")
    
    # Handle extreme outliers using IQR method
    for col in ['water_temp_celcius', 'doxy_mg_L', 'chl_a_mg_L']:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        outliers = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
        if outliers > 0:
            print(f"Found {outliers} outliers in {col}")
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    
    print(f"Final dataset shape after cleaning: {df_clean.shape}")
    
    # Visualize relationships
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(df_clean['water_temp_celcius'], df_clean['doxy_mg_L'], alpha=0.5)
    plt.title('DO vs Water Temperature')
    plt.xlabel('Water Temperature (°C)')
    plt.ylabel('Dissolved Oxygen (mg/L)')
    
    plt.subplot(1, 2, 2)
    plt.scatter(df_clean['chl_a_mg_L'], df_clean['doxy_mg_L'], alpha=0.5)
    plt.title('DO vs Chlorophyll-a')
    plt.xlabel('Chlorophyll-a (mg/L)')
    plt.ylabel('Dissolved Oxygen (mg/L)')
    
    plt.tight_layout()
    plt.savefig('do_relationships.png')
    print("Created visualization: do_relationships.png")
    
    # Save cleaned data if output path is provided
    if output_path:
        df_clean.to_csv(output_path, index=False)
        print(f"Saved cleaned data to {output_path}")
    
    return df_clean

if __name__ == "__main__":
    clean_data(output_path="./data/water_quality_cleaned.csv")