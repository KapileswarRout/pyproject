import numpy as np
import pandas as pd
from dataset_helper import fluctuating_load_pattern, bee_search_algorithm, particle_swarm_optimization,renewable_capacity,conventional_capacity
def generate_dataset(method='PSO', samples=1000, num_steps=100):
    dataset = []
    
    for _ in range(samples):
        # Generate fluctuating load demand
        base_demand = 25000  # 25,000 MW
        load_demand = fluctuating_load_pattern(base_demand, num_steps)
        
        # Run optimization algorithm
        if method == 'PSO':
            best_solution = particle_swarm_optimization(renewable_capacity, conventional_capacity, load_demand, 1000, 50)
        else:
            best_solution = bee_search_algorithm(renewable_capacity, conventional_capacity, load_demand, 1000)
        
        # Store input-output pair
        dataset.append(np.concatenate([load_demand, best_solution]))
        print('Sample', _+1, 'completed')
    # Convert dataset to DataFrame
    columns = [f'load_t{i}' for i in range(num_steps)] + ['solar', 'wind', 'coal', 'gas']
    df = pd.DataFrame(dataset, columns=columns)
    
    # Save as CSV
    filename = f'dataset_{method}.csv'
    df.to_csv(filename, index=False)
    print(f'Dataset saved as {filename}')

# Generate datasets for both methods
generate_dataset(method='PSO')
generate_dataset(method='BSA')
