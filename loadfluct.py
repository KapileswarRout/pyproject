import numpy as np
import matplotlib.pyplot as plt

renewable_capacity = {'solar': 100, 'wind': 150}  # MW
conventional_capacity = {'coal': 200, 'gas': 100}  # MW


# Define the fluctuating load pattern with inertia
def fluctuating_load_pattern(base_demand, num_steps=100, max_variation=0.2, inertia=0.05):
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

# Define the optimization function to minimize the mismatch between demand and supply
def optimization_function(solar_fraction, wind_fraction, coal_fraction, gas_fraction, load_demand, time_step):
    """
    Objective function to minimize load demand imbalance at a specific time step.
    solar_fraction, wind_fraction, coal_fraction, and gas_fraction are the fractions of the maximum capacity for each energy source.
    load_demand: The fluctuating load demand at the current time step.
    time_step: The current time step (index of the load demand).
    """
    # Define the capacities (MW)
    renewable_capacity = {'solar': 100, 'wind': 150}
    conventional_capacity = {'coal': 200, 'gas': 100}
    
    # Calculate the actual power contributions from solar, wind, coal, and gas
    solar_output = renewable_capacity['solar'] * solar_fraction
    wind_output = renewable_capacity['wind'] * wind_fraction
    coal_output = conventional_capacity['coal'] * coal_fraction
    gas_output = conventional_capacity['gas'] * gas_fraction

    # Calculate the total supply and the imbalance (mismatch with load demand)
    total_supply = solar_output + wind_output + coal_output + gas_output
    imbalance = np.abs(total_supply - load_demand[time_step])  # Absolute difference between total supply and demand

    # Return the imbalance as the objective to minimize
    return imbalance

# Run Bee Search Algorithm to find the global optimal solution for solar, wind, coal, and gas fractions at each time step
def bee_search_algorithm(load_demand, iterations=100, population_size=10):
    best_solution = None
    best_cost = float('inf')

    # Initialize a population of bees (solar_fraction, wind_fraction, coal_fraction, gas_fraction)
    population = np.random.rand(population_size, 4)  # Random initial population of solutions

    for iteration in range(iterations):
        for i in range(population_size):
            total_cost = 0
            for t in range(len(load_demand)):
                # Get solar, wind, coal, and gas generation fractions from bee i
                solar_fraction = population[i][0]
                wind_fraction = population[i][1]
                coal_fraction = population[i][2]
                gas_fraction = population[i][3]

                # Evaluate the cost (imbalance) for this solution at time step t
                total_cost += optimization_function(solar_fraction, wind_fraction, coal_fraction, gas_fraction, load_demand, t)

            # Update the best solution if needed
            if total_cost < best_cost:
                best_cost = total_cost
                best_solution = population[i]

        # Generate new population by random exploration
        population = np.random.rand(population_size, 4)

    return best_solution, best_cost

# Example: Load demand data with inertia and base demand of 500 MW
base_demand = 500
load_demand = fluctuating_load_pattern(base_demand)

# Run Bee Search Algorithm to find the optimal solution
best_solution_bsa, best_cost_bsa = bee_search_algorithm(load_demand)

# Print the result
print(f"Best solution found by BSA: Solar: {best_solution_bsa[0]:.4f}, Wind: {best_solution_bsa[1]:.4f}, Coal: {best_solution_bsa[2]:.4f}, Gas: {best_solution_bsa[3]:.4f} with total cost: {best_cost_bsa:.4f}")

# Plot the fluctuating load demand
plt.plot(load_demand)
plt.xlabel('Time Step')
plt.ylabel('Load Demand (MW)')
plt.title('Fluctuating Load Demand with Inertia')
plt.grid(True)
plt.show()
