import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# Function to generate 200 seconds of fluctuating load demand
def fluctuating_load_pattern(base_demand, num_steps=200, max_variation=0.2, inertia=0.05):
    """
    Generates a fluctuating load demand pattern with inertia.
    base_demand: The base load demand around which fluctuations occur.
    num_steps: The number of time steps for the simulation.
    max_variation: The maximum variation (percentage) in load demand.
    inertia: Controls how much the previous value affects the current value.
    """
    load = np.zeros(num_steps)
    load[0] = base_demand  # Starting load demand is the base demand
    
    for t in range(1, num_steps):
        # Generate random fluctuation
        fluctuation = np.random.uniform(-max_variation, max_variation) * base_demand
        # Apply inertia (smooth out large fluctuations)
        load[t] = load[t-1] * (1 - inertia) + (base_demand + fluctuation) * inertia
        
        # Ensure that the load demand stays within reasonable bounds
        load[t] = max(0, load[t])  # Prevent negative load
        load[t] = min(base_demand * 1.5, load[t])  # Prevent excessively high load
    
    return load

# Load trained LSTM model
def load_lstm_model(model_path='lstm_load_prediction.h5'):
    print(f"Loading LSTM model from {model_path}...")
    return load_model(model_path, compile=False)

# Load power allocation model
def load_power_allocation_model(model_path='power_allocation_model.h5'):
    print(f"Loading power allocation model from {model_path}...")
    return load_model(model_path, compile=False)

# Load MinMax Scalers
scaler_X = joblib.load('scaler_X.pkl')
scaler_y = joblib.load('scaler_y.pkl')

# Predict next 100 seconds of load
def predict_next_100_seconds(last_200_seconds, model):
    print("\n[INFO] Predicting next 100 seconds of load...")
    
    # Reshape input
    last_200_seconds = np.array(last_200_seconds).reshape(1, -1)
    print(f"[DEBUG] Shape before scaling: {last_200_seconds.shape}")
    
    last_200_seconds_scaled = scaler_X.transform(last_200_seconds).reshape(1, 200, 1)
    print(f"[DEBUG] Shape after scaling: {last_200_seconds_scaled.shape}")

    # Predict
    predicted_scaled = model.predict(last_200_seconds_scaled)[0]
    print(f"[DEBUG] Raw scaled prediction shape: {predicted_scaled.shape}")

    # Inverse transform
    predicted_actual = scaler_y.inverse_transform(predicted_scaled.reshape(1, -1))[0]
    print(f"[DEBUG] Final prediction shape: {predicted_actual.shape}")

    return predicted_actual

# Predict power allocation using the second model
def predict_power_allocation(load_prediction, model):
    print("\n[INFO] Predicting power allocation values...")

    # Ensure correct shape
    load_prediction = np.array(load_prediction).reshape(1, -1)
    print(f"[DEBUG] Input shape for power allocation model: {load_prediction.shape}")

    # Predict power allocation
    allocation_prediction = model.predict(load_prediction)[0]
    print(f"[DEBUG] Power allocation prediction shape: {allocation_prediction.shape}")

    return allocation_prediction

# Load both models
lstm_model = load_lstm_model()
power_allocation_model = load_power_allocation_model()

# Generate 200 seconds of fluctuating load demand
base_demand = 500  # Set an appropriate base demand
latest_data = fluctuating_load_pattern(base_demand, num_steps=200)

print("\n[INFO] Generated fluctuating load demand for 200 seconds.")

# Predict next 100 seconds of load
predicted_load = predict_next_100_seconds(latest_data, lstm_model)
print("\n[RESULT] Predicted Load for Next 100 Seconds:", predicted_load)

# Predict power allocation
power_allocation = predict_power_allocation(predicted_load, power_allocation_model)
print("\n[RESULT] Predicted Power Allocation (4 values):", power_allocation)

