import numpy as np
import random

# PSO optimization
class ParticleSwarmOptimization:
    def __init__(self, objective_function, bounds, num_particles=10, iterations=15, w=0.7, c1=1.5, c2=1.5):
        self.objective_function = objective_function
        self.bounds = bounds
        self.num_particles = num_particles
        self.iterations = iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.convergence_history = []

    def optimize_hyperparameters(self):
        param_bounds = self.bounds
        objective_function = self.objective_function
        n_particles = self.num_particles
        n_iterations = self.iterations
        n_params = len(param_bounds)
        positions = np.random.uniform(
            [b[0] for b in param_bounds],
            [b[1] for b in param_bounds],
            (n_particles, n_params)
        )
        velocities = np.random.uniform(-0.1, 0.1, (n_particles, n_params))

        personal_best_positions = positions.copy()
        personal_best_scores = np.array([objective_function(p) for p in positions])
        global_best_position = personal_best_positions[np.argmax(personal_best_scores)]
        global_best_score = np.max(personal_best_scores)

        self.convergence_history.append(global_best_score)
        history = [{'iteration': 0, 'best_score': global_best_score}]

        for iteration in range(n_iterations):
            for i in range(n_particles):
                r1, r2 = np.random.rand(2)
                velocities[i] = (self.w * velocities[i] +
                               self.c1 * r1 * (personal_best_positions[i] - positions[i]) +
                               self.c2 * r2 * (global_best_position - positions[i]))

                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i],
                                     [b[0] for b in param_bounds],
                                     [b[1] for b in param_bounds])

                score = objective_function(positions[i])

                if score > personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i].copy()

                if score > global_best_score:
                    global_best_score = score
                    global_best_position = positions[i].copy()

            self.convergence_history.append(global_best_score)
            history.append({'iteration': iteration + 1, 'best_score': global_best_score})
            print(f"PSO Iteration {iteration+1}/{n_iterations}, Best Accuracy: {global_best_score:.4f}")

        return global_best_position, global_best_score, history