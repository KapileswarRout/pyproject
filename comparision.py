import numpy as np
import matplotlib.pyplot as plt

# Step 1: Define grid parameters
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
base_demand = 500
load_demand = fluctuating_load_pattern(base_demand)

# Step 2: Renewable generation function
def renewable_generation(renewable_capacity):
    solar_output = renewable_capacity['solar'] * (np.random.rand() * 0.8 + 0.2)  # Random output between 20% to 100% of capacity
    wind_output = renewable_capacity['wind'] * (np.random.rand() * 0.7 + 0.3)   # Random output between 30% to 100% of capacity
    return solar_output, wind_output

# Step 3: Conventional generation function
def conventional_generation(remaining_demand, conventional_capacity):
    coal_output = min(conventional_capacity['coal'], remaining_demand)  # Coal generation based on remaining demand
    gas_output = remaining_demand - coal_output  # Gas generation for the remaining demand
    return coal_output, gas_output

# Step 4: Frequency deviation function
def grid_frequency(solar_output, wind_output, coal_output, gas_output, load_demand):
    total_generation = solar_output + wind_output + coal_output + gas_output
    frequency_deviation = total_generation - load_demand
    return frequency_deviation

# Step 5: Bee Search Algorithm (BSA) function
def bee_search_algorithm(renewable_capacity, conventional_capacity, load_demand, iterations):
    best_solution = np.random.rand(4) * [renewable_capacity['solar'], renewable_capacity['wind'], conventional_capacity['coal'], conventional_capacity['gas']]
    best_deviation = np.inf
    
    for _ in range(iterations):
        new_solution = best_solution + (np.random.rand(4) - 0.5) * best_solution * 0.05
        new_solution = np.clip(new_solution, 0, [renewable_capacity['solar'], renewable_capacity['wind'], conventional_capacity['coal'], conventional_capacity['gas']])
        
        solar_output, wind_output, coal_output, gas_output = new_solution
        frequency_deviation = grid_frequency(solar_output, wind_output, coal_output, gas_output, load_demand)
        
        if np.all(np.abs(frequency_deviation) < best_deviation):  # Use np.all() for array comparison
            best_solution = new_solution
            best_deviation = np.abs(frequency_deviation)
    
    return best_solution  # Return the best solution found


# Step 6: Particle Swarm Optimization (PSO) function
def particle_swarm_optimization(renewable_capacity, conventional_capacity, load_demand, iterations, swarm_size):
    swarm = np.random.rand(swarm_size, 4) * [renewable_capacity['solar'], renewable_capacity['wind'], conventional_capacity['coal'], conventional_capacity['gas']]
    velocities = np.zeros_like(swarm)
    best_positions = np.copy(swarm)
    best_deviation = np.inf * np.ones(swarm_size)
    
    global_best_position = best_positions[0]
    global_best_deviation = np.inf

    for _ in range(iterations):
        for i in range(swarm_size):
            solar_output, wind_output, coal_output, gas_output = swarm[i]
            frequency_deviation = grid_frequency(solar_output, wind_output, coal_output, gas_output, load_demand)
            
            # Ensure that frequency_deviation is a scalar value
            if isinstance(frequency_deviation, np.ndarray):
                frequency_deviation = frequency_deviation[0]  # Take the first element if it's an array

            # Update best deviation for each particle
            if abs(frequency_deviation) < best_deviation[i]:
                best_positions[i] = swarm[i]
                best_deviation[i] = abs(frequency_deviation)

            # Update global best solution
            if abs(frequency_deviation) < global_best_deviation:
                global_best_position = swarm[i]
                global_best_deviation = abs(frequency_deviation)
        
        inertia_weight = 0.9
        cognitive_weight = 1.5
        social_weight = 1.4
        
        for i in range(swarm_size):
            velocities[i] = inertia_weight * velocities[i] + \
                            cognitive_weight * np.random.rand() * (best_positions[i] - swarm[i]) + \
                            social_weight * np.random.rand() * (global_best_position - swarm[i])
            swarm[i] = swarm[i] + velocities[i]
            swarm[i] = np.clip(swarm[i], 0, [renewable_capacity['solar'], renewable_capacity['wind'], conventional_capacity['coal'], conventional_capacity['gas']])
    
    return global_best_position

# Step 7: Cost Calculation Function
import numpy as np

def calculate_total_cost(solar_output, wind_output, coal_output, gas_output):
    # Adjusted costs to reflect market prices more realistically
    cost_per_mw = {
        'solar': 50,  # $/MWh
        'wind': 40,   # $/MWh
        'coal': 20,   # $/MWh
        'gas': 30     # $/MWh
    }
    
    # Ensure inputs are NumPy arrays
    solar_output = np.asarray(solar_output)
    wind_output = np.asarray(wind_output)
    coal_output = np.asarray(coal_output)
    gas_output = np.asarray(gas_output)
    
    # Calculate total cost
    total_cost = np.sum(
        solar_output * cost_per_mw['solar'] + 
        wind_output * cost_per_mw['wind'] + 
        coal_output * cost_per_mw['coal'] + 
        gas_output * cost_per_mw['gas']
    )
    
    return total_cost

# Step 8: Main execution
print('Starting simulation...')

# Run BSA and PSO
best_solution_bsa = bee_search_algorithm(renewable_capacity, conventional_capacity, load_demand, 1000)
print('Best solution found by BSA:')
print(best_solution_bsa)

# Calculate total cost after BSA
solar_output_bsa, wind_output_bsa, coal_output_bsa, gas_output_bsa = best_solution_bsa
total_cost_bsa = calculate_total_cost(solar_output_bsa, wind_output_bsa, coal_output_bsa, gas_output_bsa)
print(f'Total cost after BSA: {total_cost_bsa:.2f}')

best_solution_pso = particle_swarm_optimization(renewable_capacity, conventional_capacity, load_demand, 1000, 50)
print('Best solution found by PSO:')
print(best_solution_pso)

# Calculate total cost after PSO
solar_output_pso, wind_output_pso, coal_output_pso, gas_output_pso = best_solution_pso
total_cost_pso = calculate_total_cost(solar_output_pso, wind_output_pso, coal_output_pso, gas_output_pso)
print(f'Total cost after PSO: {total_cost_pso:.2f}')

# Calculate total cost without using any algorithm
solar_output, wind_output = renewable_generation(renewable_capacity)
remaining_demand = load_demand - (solar_output + wind_output)
coal_output, gas_output = conventional_generation(remaining_demand, conventional_capacity)
total_cost_without_algo = calculate_total_cost(solar_output, wind_output, coal_output, gas_output)
print(f'Total cost without optimization algorithm: {total_cost_without_algo:.2f}')


# Plot the fluctuating load demand
plt.plot(load_demand)
plt.xlabel('Time Step')
plt.ylabel('Load Demand (MW)')
plt.title('Fluctuating Load Demand with Inertia')
plt.grid(True)
plt.show()