import joblib
import numpy as np

# Load the model and scaler
model = joblib.load('ph_prediction_model.pkl')
scaler = joblib.load('feature_scaler.pkl')

def predict_ph(chla, temp, do):
    """
    Predict pH based on chlorophyll-a, water temperature, and dissolved oxygen
    
    Parameters:
    - chla: Chlorophyll-a concentration
    - temp: Water temperature
    - do: Dissolved oxygen
    
    Returns:
    - Predicted pH value
    """
    # Create input array and reshape it
    features = np.array([chla, temp, do]).reshape(1, -1)
    
    # Scale the features
    features_scaled = scaler.transform(features)
    
    # Make prediction
    predicted_ph = model.predict(features_scaled)[0]
    
    return predicted_ph

# Example usage
if __name__ == "__main__":
    # Example values
    chla = 5.0  # µg/L
    temp = 25.0  # °C
    do = 8.0  # mg/L
    
    predicted = predict_ph(chla, temp, do)
    print(f"Predicted pH: {predicted:.2f}")