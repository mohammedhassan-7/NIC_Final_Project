import numpy as np

# Tabu Search Optimization
class TabuSearch:
    def __init__(self, objective_function, bounds, iterations=50, tabu_tenure=2):
        self.objective_function = objective_function
        self.bounds = bounds
        self.iterations = iterations
        self.tabu_tenure = tabu_tenure
        self.tabu_list = {}
        self.history = []

    def _generate_neighbors(self, solution, step_size=0.1):
        neighbors = []
        for i in range(len(solution)):
            neighbor = solution.copy()
            neighbor[i] += step_size
            if neighbor[i] <= self.bounds[i][1]:
                neighbors.append(neighbor)

            neighbor = solution.copy()
            neighbor[i] -= step_size
            if neighbor[i] >= self.bounds[i][0]:
                neighbors.append(neighbor)
        return neighbors

    def optimize_hyperparameters(self):
        # Initialize with a random solution
        current_solution = np.array([np.random.uniform(b[0], b[1]) for b in self.bounds])
        current_cost = -self.objective_function(current_solution)

        best_solution = current_solution.copy()
        best_cost = current_cost
        self.convergence_history = [-best_cost]
        result_history = [{'iteration': 0, 'best_score': -best_cost}]

        for it in range(self.iterations):
            neighbors = self._generate_neighbors(current_solution)
            neighbor_scores = []

            # Decrement tabu tenure
            for sol in list(self.tabu_list.keys()):
                self.tabu_list[sol] -= 1
                if self.tabu_list[sol] == 0:
                    del self.tabu_list[sol]

            for neighbor in neighbors:
                neighbor_tuple = tuple(neighbor)
                if neighbor_tuple not in self.tabu_list:
                    neighbor_cost = -self.objective_function(neighbor)
                    neighbor_scores.append((neighbor, neighbor_cost))
                else:
                    # Aspiration Criterion
                    neighbor_cost = -self.objective_function(neighbor)
                    if neighbor_cost < best_cost:
                        neighbor_scores.append((neighbor, neighbor_cost))

            if not neighbor_scores:
                print("Stopping: No admissible neighbors left.")
                break

            # Choose the best neighbor
            neighbor_scores.sort(key=lambda x: x[1])
            best_neighbor, best_neighbor_cost = neighbor_scores[0]

            current_solution = best_neighbor
            current_cost = best_neighbor_cost

            # Update tabu list
            self.tabu_list[tuple(current_solution)] = self.tabu_tenure

            # Update best solution
            if current_cost < best_cost:
                best_solution = current_solution.copy()
                best_cost = current_cost

            self.convergence_history.append(-best_cost)
            self.history.append((it, current_solution, current_cost, best_solution, best_cost, self.tabu_list.copy()))
            result_history.append({'iteration': it + 1, 'best_score': -best_cost})

            if it % 5 == 0 or it == self.iterations - 1:
                print(f"Tabu Iteration {it+1}/{self.iterations}, Best Accuracy: {-best_cost:.4f}")

        return best_solution, -best_cost, result_history