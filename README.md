# Nature-Inspired Optimization for Sentiment Analysis

**Authors:** [Anas Ahmad](https://github.com/anas-Ah25), [Mohammed Hassan](https://github.com/mohammedhassan-7)<br>
**Course:** Nature Inspired Computation  
**Date:** December 20, 2025

## Overview

This project uses metaheuristic algorithms to optimize a BiLSTM sentiment classifier on the IMDB movie review dataset. It implements a three-phase optimization pipeline: hyperparameter tuning, meta-optimization of algorithm parameters, and explainability optimization using LIME.

## Dataset

- **Source:** [IMDB Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/data) (50K samples)
- **Task:** Binary sentiment classification (positive/negative)
- **Preprocessing:** Tokenization (30K vocab), padding (150 tokens), stratified sampling

## Project Structure

```
phase 1/                    # Model hyperparameter optimization
phase 2/                    # Meta-optimization & XAI optimization
```

## Results

### Phase 1: Hyperparameter Optimization
- **Algorithms Evaluated:** Tabu Search, Simulated Annealing, Particle Swarm Optimization (PSO), Whale Optimization Algorithm (WOA), Firefly Algorithm (FA)
- **Search Space:** Hidden dimension (64-256), dropout (0.2-0.6), learning rate (0.0001-0.005)
- **Best Algorithm:** Particle Swarm Optimization (PSO)
- **Accuracy:** 76.4% → **78.9%** (after meta-optimization)

### Phase 2: Meta-Optimization
- **Meta-Optimizer:** Grey Wolf Optimizer (GWO)
- **Tuned Algorithms:** PSO, Firefly Algorithm
- **Parameter Tuning:** PSO (w, c1, c2), FA (β₀, γ, α)
- **Best Result:** PSO with tuned parameters → **78.9% accuracy**
- **Full Dataset Test:** PSO achieved **85.85%** accuracy on complete IMDB dataset (vs. FA at 85.72%)

### Phase 3: XAI Optimization

**Approach 1: Fidelity-Clarity Trade-off**
- **Objective:** Maximize R² fidelity while minimizing feature count
- **Best Algorithm:** Bat Algorithm
- **Result:** 91% fidelity with only 8 features (40% more concise)

**Approach 2: Composite Multi-Metric**
- **Objective:** Optimize LIME using weighted composite metric with compactness penalty:
  - Faithfulness (40%): Impact of feature removal on predictions
  - Fidelity (30%): R² score of linear approximation
  - Stability (30%): Consistency across multiple runs
  - Compactness Penalty: Penalizes explanations with too many features
- **Best Algorithm:** Grey Wolf Optimizer (GWO)
- **Result:** Superior overall explanation quality with balanced metrics

## Repository

https://github.com/mohammedhassan-7/nature-opt-sentiment-imdb
