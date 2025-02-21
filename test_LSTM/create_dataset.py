import pandas as pd
import numpy as np

# Generate timestamps (in seconds)
num_points = 5000  # Adjust this for more data
timestamps = pd.date_range(start="2025-02-20", periods=num_points, freq="S")
seconds = np.arange(0, num_points)

# Generate load values around 25000 with small random variations
load_values = 25000 + np.random.normal(0, 500, num_points)  # Mean 25000, Std Dev 500

# Create DataFrame
data = pd.DataFrame({"timestamp": timestamps, "seconds": seconds, "load": load_values})

# Save to CSV
data.to_csv("electricity_data.csv", index=False)

print("Sample dataset created and saved as 'sample_electricity_data.csv'.")
