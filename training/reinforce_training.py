import os
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from environment.custom_env import EnergyGridEnv

# ----------------------------
# Create directories
# ----------------------------
os.makedirs("models/reinforce", exist_ok=True)
os.makedirs("logs/reinforce", exist_ok=True)

# ----------------------------
# Policy network
# ----------------------------
class PolicyNetwork(nn.Module):
    def __init__(self, obs_size, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(obs_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, n_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)

# ----------------------------
# Hyperparameter sets for REINFORCE
# ----------------------------
hyperparameter_sets = [
    {"lr": 1e-3, "gamma": 0.99, "entropy_coef": 0.01},
    {"lr": 5e-4, "gamma": 0.99, "entropy_coef": 0.01},
    {"lr": 1e-4, "gamma": 0.99, "entropy_coef": 0.01},
    {"lr": 1e-3, "gamma": 0.97, "entropy_coef": 0.01},
    {"lr": 5e-4, "gamma": 0.97, "entropy_coef": 0.01},
    {"lr": 1e-4, "gamma": 0.97, "entropy_coef": 0.01},
    {"lr": 1e-3, "gamma": 0.95, "entropy_coef": 0.01},
    {"lr": 5e-4, "gamma": 0.95, "entropy_coef": 0.01},
    {"lr": 1e-4, "gamma": 0.95, "entropy_coef": 0.01},
    {"lr": 5e-4, "gamma": 0.99, "entropy_coef": 0.01},
]

TIMESTEPS = 1500  # Reduce for testing; increase later

# ----------------------------
# Run experiments
# ----------------------------
for i, params in enumerate(hyperparameter_sets, start=1):
    print(f"\n===== REINFORCE Experiment {i} =====")
    print(f"Hyperparameters: {params}")

    env = EnergyGridEnv()
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n
    policy = PolicyNetwork(obs_size, n_actions)
    optimizer = optim.Adam(policy.parameters(), lr=params["lr"])

    all_rewards = []

    for episode in range(TIMESTEPS):
        obs, _ = env.reset()
        log_probs = []
        rewards = []
        distributions = []

        done = False
        truncated = False

        while not (done or truncated):
            obs_tensor = torch.FloatTensor(obs)
            action_probs = policy(obs_tensor)
            m = torch.distributions.Categorical(action_probs)
            action = m.sample()

            log_probs.append(m.log_prob(action))
            distributions.append(m)

            obs, reward, done, truncated, _ = env.step(action.item())
            rewards.append(reward)

        # Compute discounted returns
        discounted_returns = []
        R = 0
        for r in reversed(rewards):
            R = r + params["gamma"] * R
            discounted_returns.insert(0, R)

        discounted_returns = torch.FloatTensor(discounted_returns)
        discounted_returns = (discounted_returns - discounted_returns.mean()) / (discounted_returns.std() + 1e-8)

        # Policy gradient loss with entropy bonus
        entropy = torch.stack([d.entropy() for d in distributions]).sum()
        loss = -torch.sum(torch.stack(log_probs) * discounted_returns) - params["entropy_coef"] * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        all_rewards.append(sum(rewards))

    # Save policy model
    model_path = f"models/reinforce/reinforce_exp_{i}.pt"
    torch.save(policy.state_dict(), model_path)

    # Test policy
    obs, _ = env.reset()
    test_rewards = []
    done = False
    truncated = False
    for step in range(50):
        obs_tensor = torch.FloatTensor(obs)
        action_probs = policy(obs_tensor)
        action = torch.argmax(action_probs).item()
        obs, reward, done, truncated, _ = env.step(action)
        test_rewards.append(reward)
        if done or truncated:
            obs, _ = env.reset()
    env.close()

    avg_reward = np.mean(test_rewards)
    print(f"REINFORCE model saved to {model_path}")
    print(f"Average reward for Experiment {i}: {avg_reward:.2f}")

    # Save reward curve
    plt.figure()
    plt.plot(np.cumsum(test_rewards))
    plt.xlabel("Step")
    plt.ylabel("Cumulative Reward")
    plt.title(f"REINFORCE Experiment {i} - Cumulative Reward")
    plt.savefig(f"logs/reinforce/reinforce_exp_{i}_reward_curve.png")
    plt.close()