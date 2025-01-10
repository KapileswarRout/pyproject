import numpy as np

# Step 1: Define grid parameters
renewable_capacity = {'solar': 100, 'wind': 150}  # MW
conventional_capacity = {'coal': 200, 'gas': 100}  # MW
load_demand = 250  # MW

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
        new_solution = best_solution + (np.random.rand(4) - 0.5) * best_solution * 0.1
        new_solution = np.clip(new_solution, 0, [renewable_capacity['solar'], renewable_capacity['wind'], conventional_capacity['coal'], conventional_capacity['gas']])
        
        solar_output, wind_output, coal_output, gas_output = new_solution
        frequency_deviation = grid_frequency(solar_output, wind_output, coal_output, gas_output, load_demand)
        
        if abs(frequency_deviation) < best_deviation:
            best_solution = new_solution
            best_deviation = abs(frequency_deviation)
    
    return best_solution

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

            if abs(frequency_deviation) < best_deviation[i]:
                best_positions[i] = swarm[i]
                best_deviation[i] = abs(frequency_deviation)

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

# Step 7: Main execution
print('Starting simulation...')

# Run BSA and PSO
best_solution_bsa = bee_search_algorithm(renewable_capacity, conventional_capacity, load_demand, 100)
print('Best solution found by BSA:')
print(best_solution_bsa)

best_solution_pso = particle_swarm_optimization(renewable_capacity, conventional_capacity, load_demand, 100, 50)
print('Best solution found by PSO:')
print(best_solution_pso)


