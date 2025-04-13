from train_do_estimator import load_model, predict_do
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from get_temperature import get_surface_temperature
import pandas as pd
import argparse
from datetime import datetime

def process_pond_data(csv_path, temp_offset=0):
    """
    Process pond data to predict DO levels
    
    Parameters:
    csv_path (str): Path to pond_chl.csv
    temp_offset (float): Temperature offset to subtract from skin temperature
    
    Returns:
    pandas.DataFrame: Results including lat, lon, datetime, pond_id, chl, surface_temp, and DO
    """
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    # Convert datetime to consistent format
    df['datetime'] = pd.to_datetime(df['datetime']).dt.strftime('%Y-%m-%d')
    
    # Load the DO prediction model
    model_data = load_model()
    model = model_data['model']
    
    # Initialize lists for new data
    surface_temps = []
    predicted_dos = []
    
    # Process each row
    for _, row in df.iterrows():
        # Get temperature for each location and date
        temp = get_surface_temperature(row['lat'], row['lon'], row['datetime'])
        
        # Apply temperature offset
        if temp is not None:
            temp = temp - temp_offset
        else:
            temp = None
        
        surface_temps.append(temp)
        
        # Predict DO if we have valid temperature
        if temp is not None:
            do = predict_do(model, temp, row['chl'])
            predicted_dos.append(do)
        else:
            predicted_dos.append(None)
    
    # Add new columns to DataFrame
    df['surface_temp'] = surface_temps
    df['predicted_do'] = predicted_dos
    
    return df

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process pond data for DO prediction.')
    parser.add_argument('input_csv', help='Path to input CSV file')
    parser.add_argument('temp_offset', type=float, help='Temperature offset value')
    args = parser.parse_args()
    
    # Process the data
    results_df = process_pond_data(args.input_csv, args.temp_offset)
    
    # Create output path
    output_dir = os.path.join(os.path.dirname(args.input_csv), 'results')
    output_csv = os.path.join(output_dir, 'pond_do_predictions.csv')
    
    # Create results directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    results_df.to_csv(output_csv, index=False)
    print(f"Results saved to: {output_csv}")

if __name__ == "__main__":
    main()