import numpy as np

# Simulated Annealing optimization
class SimulatedAnnealing:
    def __init__(self, objective_function, bounds, iterations=50, temp=100, cooling_rate=0.90, min_temp=0.01):
        self.objective_function = objective_function
        self.bounds = bounds
        self.iterations = iterations
        self.initial_temp = temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.convergence_history = []

    def acceptance_probability(self, old_cost, new_cost, temperature):
        if new_cost < old_cost:
            return 1.0
        import math
        return math.exp((old_cost - new_cost) / temperature)

    def optimize_hyperparameters(self):
        param_bounds = self.bounds
        objective_function = self.objective_function
        max_iterations = self.iterations
        current_solution = np.array([np.random.uniform(b[0], b[1]) for b in param_bounds])
        current_cost = -objective_function(current_solution)

        best_solution = current_solution.copy()
        best_cost = current_cost

        self.convergence_history.append(-best_cost)
        history = [{'iteration': 0, 'best_score': -best_cost}]

        temperature = self.initial_temp
        iteration = 0

        while temperature > self.min_temp and iteration < max_iterations:
            neighbor = current_solution + np.random.uniform(-0.1, 0.1, len(param_bounds))
            neighbor = np.clip(neighbor,
                             [b[0] for b in param_bounds],
                             [b[1] for b in param_bounds])

            neighbor_cost = -objective_function(neighbor)

            if self.acceptance_probability(current_cost, neighbor_cost, temperature) > np.random.random():
                current_solution = neighbor
                current_cost = neighbor_cost

                if current_cost < best_cost:
                    best_solution = current_solution.copy()
                    best_cost = current_cost

            self.convergence_history.append(-best_cost)
            history.append({'iteration': iteration + 1, 'best_score': -best_cost})
            temperature *= self.cooling_rate
            iteration += 1
            
            if iteration % 5 == 0:
                print(f"SA Iteration {iteration}/{max_iterations}, Best Accuracy: {-best_cost:.4f}")

        return best_solution, -best_cost, history