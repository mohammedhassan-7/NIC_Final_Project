import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.nic_algorithms.pso import ParticleSwarmOptimization as PSO
from src.nic_algorithms.genetic_alg import GeneticAlgorithm
from src.nic_algorithms.simulated_annealing import SimulatedAnnealing
from src.nic_algorithms.tabu import TabuSearch
from src import obj_function, TinyBERTClassifier, load_data, tokenize_texts

# ==========================================
# 1. Load DATA (Subset vs Full)
# ==========================================
print("Loading data...")
X_train, X_val, y_train, y_val, mlb, num_classes = load_data()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Number of classes: {num_classes}")
print(f"Using device: {device}")

# Create subset for optimization (1000 samples for speed)
subset_size = min(7000, len(X_train))
X_train_subset = X_train[:subset_size]
y_train_subset = y_train[:subset_size]
X_val_subset = X_val[:1400]
y_val_subset = y_val[:1400]

print(f"Subset size: {subset_size} train, {len(X_val_subset)} val")

# Tokenize subsets
print("Tokenizing texts...")
train_input_ids, train_attention_mask = tokenize_texts(X_train_subset, max_length=128)
val_input_ids, val_attention_mask = tokenize_texts(X_val_subset, max_length=128)

y_train_subset_tensor = torch.tensor(y_train_subset, dtype=torch.float32)
y_val_subset_tensor = torch.tensor(y_val_subset, dtype=torch.float32)

# Create datasets and dataloaders for hyperparameter optimization
def create_dataloaders(batch_size):
    train_dataset = TensorDataset(train_input_ids, train_attention_mask, y_train_subset_tensor)
    val_dataset = TensorDataset(val_input_ids, val_attention_mask, y_val_subset_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=int(batch_size), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=int(batch_size))
    
    return train_loader, val_loader

def fitness_wrapper(params):
    """Wrapper for optimization algorithms"""
    batch_size = int(params[2])
    train_loader, val_loader = create_dataloaders(batch_size)
    return obj_function(params, train_loader, val_loader, num_classes, device)

# Hyperparameter bounds: [learning_rate, dropout, batch_size]
bounds = [(1e-5, 1e-3), (0.1, 0.6), (8, 64)]

# ==========================================
# 2. RUN OPTIMIZATIONS (The "Search")
# ==========================================

# --- PSO ---
print(">>> Running PSO...")
pso = PSO(fitness_wrapper, bounds, num_particles=10, iterations=15)
best_pso, score_pso, hist_pso = pso.optimize_hyperparameters()
pd.DataFrame(hist_pso).to_csv("results/pso_history.csv", index=False)

# --- GA ---
print(">>> Running GA...")
ga = GeneticAlgorithm(fitness_wrapper, bounds, pop_size=10, generations=15)
best_ga, score_ga, hist_ga = ga.optimize_hyperparameters()
pd.DataFrame(hist_ga).to_csv("results/ga_history.csv", index=False)

# --- SA ---
print(">>> Running Simulated Annealing...")
sa = SimulatedAnnealing(fitness_wrapper, bounds, iterations=50, temp=1.0)
best_sa, score_sa, hist_sa = sa.optimize_hyperparameters()
pd.DataFrame(hist_sa).to_csv("results/sa_history.csv", index=False)

# --- Tabu ---
print(">>> Running Tabu Search...")
ts = TabuSearch(fitness_wrapper, bounds, iterations=50, tabu_tenure=7)
best_ts, score_ts, hist_ts = ts.optimize_hyperparameters()
pd.DataFrame(hist_ts).to_csv("results/tabu_history.csv", index=False)

# ==========================================
# 3. COMPARE & SELECT WINNER
# ==========================================
print("\n" + "="*80)
print("OPTIMIZATION RESULTS SUMMARY")
print("="*80)

# Display results for each algorithm
results = {
    "PSO": {"params": best_pso, "accuracy": score_pso},
    "GA": {"params": best_ga, "accuracy": score_ga},
    "SA": {"params": best_sa, "accuracy": score_sa},
    "Tabu": {"params": best_ts, "accuracy": score_ts}
}

for algo_name, result in results.items():
    lr, dropout, batch_size = result['params']
    print(f"\n{algo_name}:")
    print(f"  Learning Rate: {lr:.6f}")
    print(f"  Dropout: {dropout:.4f}")
    print(f"  Batch Size: {int(batch_size)}")
    print(f"  Validation Accuracy: {result['accuracy']:.4f}")

scores = {name: result['accuracy'] for name, result in results.items()}
winner_name = max(scores, key=scores.get)

print("\n" + "="*80)
print(f"🏆 WINNING ALGORITHM: {winner_name}")
print(f"   Best Validation Accuracy: {scores[winner_name]:.4f}")
print("="*80)

if winner_name == "PSO": best_params = best_pso
elif winner_name == "GA": best_params = best_ga
elif winner_name == "SA": best_params = best_sa
else: best_params = best_ts

# ==========================================
# 4. FINAL TRAINING (The "Real Deal")
# ==========================================
print(f"\nStarting FINAL training on FULL DATASET using best params: {best_params}")

# Unpack best params
lr = best_params[0]
dropout = best_params[1]
batch_size = int(best_params[2])

print(f"Learning Rate: {lr:.6f}, Dropout: {dropout:.4f}, Batch Size: {batch_size}")

# Tokenize full dataset
print("Tokenizing full dataset...")
X_train_full_input_ids, X_train_full_attention_mask = tokenize_texts(X_train, max_length=128)
X_val_full_input_ids, X_val_full_attention_mask = tokenize_texts(X_val, max_length=128)

y_train_full_tensor = torch.tensor(y_train, dtype=torch.float32)
y_val_full_tensor = torch.tensor(y_val, dtype=torch.float32)

# Create full dataloaders
train_dataset_full = TensorDataset(X_train_full_input_ids, X_train_full_attention_mask, y_train_full_tensor)
val_dataset_full = TensorDataset(X_val_full_input_ids, X_val_full_attention_mask, y_val_full_tensor)

train_loader_full = DataLoader(train_dataset_full, batch_size=batch_size, shuffle=True)
val_loader_full = DataLoader(val_dataset_full, batch_size=batch_size)

# Init Final Model
final_model = TinyBERTClassifier(num_classes, dropout=dropout).to(device)
optimizer = torch.optim.Adam(final_model.parameters(), lr=lr)
criterion = torch.nn.BCELoss()

# Train for 10 epochs on full data
print("Training on full dataset...")
for epoch in range(10):
    final_model.train()
    for batch in train_loader_full:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        optimizer.zero_grad()
        outputs = final_model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Final Training Epoch {epoch+1}/10 Complete")

# Save the deliverable
torch.save(final_model.state_dict(), "results/final_optimized_model.pth")
print("\nProject Complete. Model Saved to results/final_optimized_model.pth")