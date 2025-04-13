import pandas as pd
import os

# Try different encodings if the default utf-8 fails
encodings_to_try = ["utf-8", "latin-1", "ISO-8859-1", "cp1252"]
df = None

for encoding in encodings_to_try:
    try:
        print(f"Attempting to load data with {encoding} encoding...")
        df = pd.read_csv('../data/insitu_wq_data.csv', encoding=encoding)
        print(f"Successfully loaded data with {encoding} encoding and default column names")
        break
    except UnicodeDecodeError:
        print(f"Failed to decode with {encoding} encoding")
    except Exception as e:
        print(f"Error reading file with {encoding} encoding: {e}")

# Check if data was successfully loaded
if df is None:
    print("Failed to load the data with any of the attempted encodings. Exiting.")
    exit(1)

# Check the available columns 
print("Available columns:", df.columns.tolist())

# Check if required columns exist
required_columns = ['chl_a_mg_L', 'water_temp_celcius', 'doxy_mg_L']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    print(f"Error: Missing required columns: {missing_columns}")
    print("Available columns:", df.columns.tolist())
    exit(1)

# Basic data exploration
print("\nData overview:")
print(df[required_columns].describe())

# Check for any extreme outliers
print("\nChecking for outliers...")
for col in required_columns:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
    print(f"{col} - Outliers: {outliers} ({(outliers/df.shape[0])*100:.2f}%)")

# Filter rows that have non-null values for all three parameters
filtered_df = df.dropna(subset=required_columns)

# Remove columns that contain only NaN values
empty_columns = filtered_df.columns[filtered_df.isna().all()].tolist()
if empty_columns:
    print(f"\nRemoving {len(empty_columns)} completely empty columns: {empty_columns}")
    filtered_df = filtered_df.drop(columns=empty_columns)

# Display the filtered data information
print(f"\nOriginal dataset: {len(df)} rows")
print(f"Filtered dataset: {len(filtered_df)} rows")
print(f"Removed {len(df) - len(filtered_df)} rows with missing values")

# Create output directory if it doesn't exist
output_dir = './data'
os.makedirs(output_dir, exist_ok=True)

# Save the filtered data to a new CSV
output_path = os.path.join(output_dir, 'water_quality_data_for_DO.csv')
filtered_df.to_csv(output_path, index=False, encoding='utf-8')
print(f"Filtered data saved to {output_path}")

# Create a correlation matrix for the parameters
print("\nCorrelation matrix:")
correlation = filtered_df[required_columns].corr()
print(correlation)

print("\nData cleaning complete!")