import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib  # To save and load pre-trained scalers

def load_trained_model(model_path):
    """
    Loads the trained LSTM model.
    """
    return tf.keras.models.load_model(model_path)

def load_scalers(scaler_X_path='scaler_X.pkl', scaler_y_path='scaler_y.pkl'):
    """
    Loads the MinMaxScalers from saved files.
    """
    scaler_X = joblib.load(scaler_X_path)
    scaler_y = joblib.load(scaler_y_path)
    return scaler_X, scaler_y

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

def predict_next_100_seconds(last_200_seconds, model_path='lstm_load_prediction.h5', 
                             scaler_X_path='scaler_X.pkl', scaler_y_path='scaler_y.pkl'):
    """
    Predicts the next 100 seconds of load based on the last 200 seconds using an LSTM model.
    
    Parameters:
    last_200_seconds (numpy array): A NumPy array of shape (200, 1).
    model_path (str): Path to the .h5 LSTM model file.
    scaler_X_path (str): Path to saved MinMaxScaler for input features.
    scaler_y_path (str): Path to saved MinMaxScaler for output predictions.
    
    Returns:
    numpy array: Predicted values for the next 100 seconds.
    """
    # Load the trained model
    model = load_trained_model(model_path)
    
    # Load pre-trained scalers
    scaler_X, scaler_y = load_scalers(scaler_X_path, scaler_y_path)
    
    # Ensure input data shape
    last_200_seconds = np.array(last_200_seconds).reshape(1, -1)
    
    # Transform input using the pre-fitted scaler
    last_200_seconds_scaled = scaler_X.transform(last_200_seconds).reshape(1, 200, 1)
    
    # Predict next 100 seconds
    prediction = model.predict(last_200_seconds_scaled)[0]
    
    # Inverse transform the predictions
    return scaler_y.inverse_transform(prediction.reshape(1, -1))[0]

# Example usage
# Ensure you have trained scalers and saved them using joblib before running predictions.

# Generate fluctuating load data for 200 time steps
base_demand = 25000  # 25,000 MW
input_data = fluctuating_load_pattern(base_demand).reshape(200, 1)  # Reshape to (200, 1)

# Predict
model_path = 'lstm_load_prediction.h5'
scaler_X_path = 'scaler_X.pkl'
scaler_y_path = 'scaler_y.pkl'

predicted_values = predict_next_100_seconds(input_data, model_path, scaler_X_path, scaler_y_path)
print(predicted_values)
