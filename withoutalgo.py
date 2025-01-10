import numpy as np
import matplotlib.pyplot as plt

# Load Demand Simulation (e.g., fluctuating between 400 and 700 MW)
time_steps = 24  # Simulating for 24 hours
load_demand = np.random.randint(400, 700, size=time_steps)

# Fixed Fractions (without optimization, fixed fractions for each energy source)
solar_fraction = 0.5  # 50% of solar capacity
wind_fraction = 0.6   # 60% of wind capacity
coal_fraction = 0.7   # 70% of coal capacity
gas_fraction = 0.5    # 50% of gas capacity

# Power Plant Capacities (MW)
solar_capacity = 100
wind_capacity = 150
coal_capacity = 200
gas_capacity = 100

# Calculate Energy Outputs based on fixed fractions
solar_output = solar_capacity * solar_fraction
wind_output = wind_capacity * wind_fraction
coal_output = coal_capacity * coal_fraction
gas_output = gas_capacity * gas_fraction

# Calculate Total Supply (constant in this case)
total_supply = solar_output + wind_output + coal_output + gas_output

# Calculate Imbalance (Cost) for each time step
imbalances = [abs(total_supply - demand) for demand in load_demand]

# Calculate Total Cost (Sum of Imbalances over all time steps)
total_cost_without_algo = sum(imbalances)

# Output the Results
print(f"Total Cost without optimization algorithm: {total_cost_without_algo:.2f}")

# Plot the Load Demand and Total Supply for visualization
plt.plot(load_demand, label='Load Demand', color='blue')
plt.axhline(y=total_supply, color='r', linestyle='--', label='Total Supply (Fixed)')
plt.xlabel('Time Steps (hours)')
plt.ylabel('Power (MW)')
plt.title('Load Demand vs. Total Supply (Fixed Fractions)')
plt.legend()
plt.show()
