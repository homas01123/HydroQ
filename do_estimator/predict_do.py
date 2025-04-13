from train_do_estimator import load_model, predict_do

# Load the best model
model_data = load_model()
model = model_data['model']

# Predict DO for new data points
water_temp = 25.0  # Celsius
chl_a = 0.1  # mg/L
predicted_do = predict_do(model, water_temp, chl_a)
print(f"Predicted DO: {predicted_do:.2f} mg/L")