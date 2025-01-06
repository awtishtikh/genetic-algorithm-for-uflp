import numpy as np
import random
from data_uflp import ga_distances, ga_facility_costs
from tqdm import tqdm


# Assuming ga_facility_costs and ga_distances are loaded externally, but we'll also handle default generation
def generate_data(num_facilities, num_customers):
    """Generate random data for facility costs and customer distances."""
    facility_costs = np.random.randint(10, 100, size=num_facilities)
    distances = np.random.randint(1, 50, size=(num_customers, num_facilities))
    return facility_costs, distances


def calculate_cost(facility_state, facility_costs, distances):
    """Calculate the total cost for a given facility state."""
    if len(facility_state) != distances.shape[1]:
        raise ValueError("Mismatch between facility_state length and distances columns.")

    if len(facility_costs) != distances.shape[1]:
        raise ValueError("Mismatch between facility_costs length and distances columns.")

    if not any(facility_state):
        return float('inf')  # Invalid solution (no facility open)

    assignment_cost = 0
    for customer_idx in range(distances.shape[0]):
        # Assign each customer to the nearest open facility
        open_facilities = np.where(facility_state == 1)[0]
        if len(open_facilities) == 0:
            return float('inf')  # No facility is open

        customer_distances = distances[customer_idx, open_facilities]
        assignment_cost += np.min(customer_distances)

    facility_opening_cost = np.sum(facility_state * facility_costs)
    return facility_opening_cost + assignment_cost


def initialize_population(pop_size, num_facilities):
    """Generate an initial population of random solutions."""
    return [np.random.choice([0, 1], size=num_facilities) for _ in range(pop_size)]


def select_parents(population, costs):
    """Select two parents using tournament selection."""
    tournament_size = 3
    selected = random.sample(list(zip(population, costs)), tournament_size)
    selected.sort(key=lambda x: x[1])  # Sort by cost
    return selected[0][0], selected[1][0]


def crossover(parent1, parent2):
    """Perform one-point crossover."""
    point = random.randint(1, len(parent1) - 1)
    child1 = np.concatenate((parent1[:point], parent2[point:]))
    child2 = np.concatenate((parent2[:point], parent1[point:]))
    return child1, child2


def mutate(solution, mutation_rate):
    """Mutate the solution with a given mutation rate."""
    for i in range(len(solution)):
        if random.random() < mutation_rate:
            solution[i] = 1 - solution[i]  # Flip the bit
    return solution


def genetic_algorithm(num_facilities, num_customers, pop_size, generations, mutation_rate):
    """Solve UFLP using Genetic Algorithm."""
    # Load data
    facility_costs, distances = ga_facility_costs, ga_distances

    population = initialize_population(pop_size, num_facilities)
    global_best_solution = None
    global_best_cost = float("inf")
    best_generation = None

    for generation in tqdm(range(generations)):
        # Create new population
        new_population = []

        # Calculate costs
        costs = [calculate_cost(ind, facility_costs, distances) for ind in population]
        fitness = [5000 / (np.sqrt(cost) + 0.005) for cost in costs]

        # Selection process
        for idx, fit in enumerate(fitness):
            u = np.random.uniform(0, 1)
            selection_prob = fit / sum(fitness)
            if selection_prob >= u:
                new_population.append(population[idx])

        # Record the best solution
        best_idx = np.argmin(costs)
        best_solution = population[best_idx]
        best_cost = costs[best_idx]
        if best_cost <= global_best_cost:
            global_best_cost = best_cost
            global_best_solution = best_solution
            best_generation = generation

        print(f"Generation {generation}: Best Cost = {best_cost}")

        while len(new_population) < pop_size:
            parent1, parent2 = select_parents(population, costs)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            new_population.extend([child1, child2])

        # Ensure population size consistency
        population = new_population

    # Return the best solution and cost
    return global_best_solution, global_best_cost, best_generation, facility_costs, distances


if __name__ == "__main__":
    num_facilities = 20
    num_customers = 50
    pop_size = 50
    generations = 100
    mutation_rate = 0.1

    best_solution, best_cost, generation, costs, distances = genetic_algorithm(num_facilities, num_customers, pop_size,
                                                                               generations, mutation_rate)
    print("Best Facility State:", best_solution)
    print("Minimum Cost:", best_cost)
    print("Best Generation:", generation)
    print(costs)
    print(distances)
