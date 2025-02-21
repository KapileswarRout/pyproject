import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import joblib

# Load the data (Modify the file path as needed)
data = pd.read_csv('electricity_data.csv')

# Ensure the timestamp is in seconds
data['timestamp'] = pd.to_datetime(data['timestamp'])
data['seconds'] = (data['timestamp'] - data['timestamp'].min()).dt.total_seconds()

# Sort the data by timestamp
data = data.sort_values(by='seconds')

# Feature Engineering: Using past 200 seconds data to predict next 100 seconds
def create_features(data, past_seconds=200, future_seconds=100):
    X, y = [], []
    for i in range(len(data) - past_seconds - future_seconds):
        X.append(data['load'].iloc[i:i+past_seconds].values.reshape(-1, 1))
        y.append(data['load'].iloc[i+past_seconds:i+past_seconds+future_seconds].values.reshape(-1, 1))
    return np.array(X), np.array(y)

X, y = create_features(data)

# Initialize and fit scalers
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X = scaler_X.fit_transform(X.reshape(X.shape[0], -1)).reshape(X.shape[0], X.shape[1], 1)
y = scaler_y.fit_transform(y.reshape(y.shape[0], -1)).reshape(y.shape[0], y.shape[1])

# Save scalers for later use in prediction
joblib.dump(scaler_X, 'scaler_X.pkl')
joblib.dump(scaler_y, 'scaler_y.pkl')

# Build LSTM Model
model = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(X.shape[1], 1)),
    LSTM(50, activation='relu'),
    Dense(100)  # Predicting next 100 seconds
])

# Use explicit Mean Squared Error (MSE) instead of "mse" string
optimizer = Adam(learning_rate=0.0005, clipnorm=1.0)
model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError())

# Early Stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Save the trained model
model.save('lstm_load_prediction.h5')


# **Load Model Function**
def load_trained_model(model_path='lstm_load_prediction.h5'):
    return load_model(model_path, compile=False)  # Load without recompiling

# **Prediction Function**
def predict_next_100_seconds(last_200_seconds, model_path='lstm_load_prediction.h5'):
    model = load_trained_model(model_path)

    # Load saved scalers
    scaler_X = joblib.load('scaler_X.pkl')
    scaler_y = joblib.load('scaler_y.pkl')

    # Reshape and scale input
    last_200_seconds = np.array(last_200_seconds).reshape(1, -1)
    last_200_seconds_scaled = scaler_X.transform(last_200_seconds).reshape(1, 200, 1)

    # Predict next 100 seconds
    prediction = model.predict(last_200_seconds_scaled)[0]

    # Inverse transform predictions
    return scaler_y.inverse_transform(prediction.reshape(1, -1))[0]

# **Example Usage**
latest_data = data['load'].iloc[-200:].values  # Take last 200 seconds of real data
predicted_load = predict_next_100_seconds(latest_data)
print("Predicted Load for Next 100 Seconds:", predicted_load)
