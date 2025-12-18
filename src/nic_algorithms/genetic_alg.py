import random
import numpy as np

class GeneticAlgorithm:
    def __init__(self, objective_func, bounds, pop_size=20, generations=25, mutation_rate=0.2):
        self.obj_func = objective_func
        self.bounds = bounds
        self.pop_size = pop_size
        self.gens = generations
        self.mut_rate = mutation_rate
        
    def optimize_hyperparameters(self):
        # Init population
        pop = np.random.uniform([b[0] for b in self.bounds], [b[1] for b in self.bounds], (self.pop_size, len(self.bounds)))
        best_sol = None
        best_score = -1
        history = []

        for g in range(self.gens):
            print(f"GA Generation {g+1}/{self.gens}")
            scores = [self.obj_func(ind) for ind in pop]
            
            # Track best
            curr_best = np.max(scores)
            if curr_best > best_score:
                best_score = curr_best
                best_sol = pop[np.argmax(scores)]
            
            history.append({'iteration': g, 'best_score': best_score})
            print(f"GA Generation {g+1}/{self.gens}, Best Accuracy: {best_score:.4f}")

            # Selection (Tournament)
            new_pop = []
            for _ in range(self.pop_size):
                p1, p2 = pop[np.random.randint(0, self.pop_size)], pop[np.random.randint(0, self.pop_size)]
                parent1 = p1 if self.obj_func(p1) > self.obj_func(p2) else p2
                p3, p4 = pop[np.random.randint(0, self.pop_size)], pop[np.random.randint(0, self.pop_size)]
                parent2 = p3 if self.obj_func(p3) > self.obj_func(p4) else p4
                
                # Crossover
                cut = np.random.randint(0, len(self.bounds))
                child = np.concatenate((parent1[:cut], parent2[cut:]))
                
                # Mutation
                if random.random() < self.mut_rate:
                    idx = random.randint(0, len(self.bounds)-1)
                    child[idx] = random.uniform(self.bounds[idx][0], self.bounds[idx][1])
                
                new_pop.append(child)
            pop = np.array(new_pop)
            
        return best_sol, best_score, history