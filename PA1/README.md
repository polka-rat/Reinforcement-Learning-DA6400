# PA1

## Setup

#### The environment uses Python 3.12
```bash
conda env create -f env.yml
conda activate rl_pa1
```
## Q1 - Gridworld and Value Iteration
- `gridworld.ipynb` - Main notebook with all the implementations

### Main Functions

#### Value Function
```python
gridWorld[i,j,k] = value_func(state,gamma, gridWorld, smoke_penalty)
```

#### Policy Iteration
```python
gridWorld=policy_iteration(gamma=0.95,theta= 0.01,smoke_penalty=-10)
```

### Outputs
All outputs are available within `gridworld.ipynb`, run notebook to observe all the different cases mentioned in the assignment

## Q2 - Acrobot TD Learning
- `q2.ipynb` - Main notebook with all implementations and experiments
- `analysis.ipynb` - Notebook for creating graphs and analysis.

### Main Functions

#### Training
```python
agent = SARSA(env, alpha=0.1, epsilon=0.1, decay=True, n_bins=10)
agent.train(timesteps=2000000, n=200)
```

#### Testing
```python
agent.test(qtable_path="logs/s_qtable_0.1_0.1.pkl", timesteps=1000)
```

#### Hyperparameter Sweep
```python
results = hyperparameter_sweep(
    env=env,
    algorithm_class=SARSA,
    algo="s",
    alpha_values=alpha_values,
    epsilon_values=epsilon_values,
    gamma=0.99,
    n=200,
    timesteps=2000000
)
alpha_values=[0.01, 0.05, 0.1, 0.3, 0.5],
epsilon_values=[0.01, 0.05, 0.1, 0.2, 0.3])
```

### Outputs

Logs saved in:
- `logs/` - Main experiments (including hyperparameter sweep runs)
- `decay_logs/` - Epsilon decay experiments
- `seed_logs/` - Multi-seed runs
- `c/` - Epsilon decay from 1.0 to 0.1 experiments
- `bins_logs/` - Discretization experiments
- `new_r_logs` - Logs for new reward function

Each run saves:
- Q-table: `{algo}_qtable_{alpha}_{epsilon}.pkl`
- Metrics: `{algo}_logs_{alpha}_{epsilon}.pkl`