# Smart Energy Grid Optimization using Reinforcement Learning

##  Project Overview
This project implements and compares three reinforcement learning algorithms—Deep Q-Network (DQN), Proximal Policy Optimization (PPO), and REINFORCE—to optimize energy distribution in a smart grid environment. The goal is to allocate limited energy supply across multiple regions with varying demand and priority levels, maximizing efficiency while minimizing shortages and waste.

---

##  Objectives
- Develop a custom reinforcement learning environment simulating a smart energy grid
- Train and evaluate DQN, PPO, and REINFORCE agents
- Compare performance across multiple hyperparameter configurations
- Analyze stability, convergence, and generalization of each algorithm
- Visualize results using reward curves, entropy, and loss metrics

---

##  Environment Description

### Agent
The agent represents an energy distribution controller responsible for allocating electricity across 5 regions. It dynamically adjusts allocations based on demand fluctuations and priority levels.

### Action Space
The action space is **discrete (6 actions)**:
- Actions `0–4`: Increase energy allocation to a specific region
- Action `5`: Rebalance energy evenly across all regions

### Observation Space
The observation is a **numerical vector** containing:
- Energy demand for each region (5 values)
- Current energy allocation (5 values)
- Region priority weights (5 values)
- Total available energy supply (1 value)
- Current timestep (1 value)

Total size: **17-dimensional vector**

### Reward Function
The reward function balances efficiency and fairness:

\[
R = \sum served - 2 \times \sum shortage - \sum waste + 0.5 \times \sum (served \times priority) - 0.05 \times fluctuation
\]

Where:
- **Served** = min(allocation, demand)
- **Shortage** = unmet demand
- **Waste** = excess allocation

---

## Algorithms Implemented

### Deep Q-Network (DQN)
- Multi-layer perceptron (MLP) policy
- Experience replay buffer
- Target network for stable learning
- Epsilon-greedy exploration strategy

### Proximal Policy Optimization (PPO)
- Actor-Critic architecture
- Clipped objective function
- Stable gradient updates
- Entropy regularization for exploration

### REINFORCE
- Monte Carlo policy gradient method
- Softmax policy network
- Entropy bonus for exploration
- No baseline (higher variance)

---

## Hyperparameter Tuning

Each algorithm was trained using **10 different hyperparameter configurations**, varying:
- Learning rate
- Discount factor (γ)
- Batch size
- Exploration parameters (DQN)
- Entropy coefficient (PPO & REINFORCE)

---

## Results Summary

| Algorithm   | Best Experiment | Average Reward |
|------------|----------------|---------------|
| DQN        | Exp 10         | **301.41**    |
| PPO        | Exp 10         | **258.28**    |
| REINFORCE  | Exp 7          | **176.63**    |

### Key Findings
- **DQN performed best overall** due to stable learning mechanisms
- **PPO showed strong and consistent performance**
- **REINFORCE was unstable** with high variance across experiments

---

## Visualizations

The following plots are generated:

-  Cumulative reward curves (all methods)
-  DQN loss (TD error approximation)
-  Policy entropy (PPO & REINFORCE)
-  Convergence plots (moving average rewards)
-  Generalization performance on unseen states

---

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/LaurelleJinelle/Jinelle_Nformi_rl_summative.git
cd Jinelle_Nformi_rl_summative
```
### 2. Create virtual environment
```bash
python -m venv dqn-env
source dqn-env/bin/activate   # Linux/Mac
dqn-env\Scripts\activate      # Windows
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```
### 4. Train models
```bash
python -m training.dqn_training
python -m training.ppo_training
python -m training.reinforce_training
```
### 5. Generate plots
```bash
python plot_script.py
```

