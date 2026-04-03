import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from stable_baselines3 import DQN, PPO
from environment.custom_env import EnergyGridEnv
import torch.nn as nn

best_dqn_model = "models/dqn/dqn_exp_10.zip"
best_ppo_model = "models/ppo/ppo_exp_10.zip"
best_reinforce_model = "models/reinforce/reinforce_exp_7.pt"
plots_dir = "plots"
os.makedirs(plots_dir, exist_ok=True)

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

def test_dqn(model_path, steps=100):
    env = EnergyGridEnv()
    model = DQN.load(model_path)
    obs, _ = env.reset()
    rewards, losses = [], []

    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(action)
        rewards.append(reward)
        td_values = model.q_net(torch.FloatTensor(obs).unsqueeze(0))
        losses.append(td_values.var().item())
        if done or truncated:
            obs, _ = env.reset()
    env.close()
    return rewards, losses

def test_ppo(model_path, steps=100):
    env = EnergyGridEnv()
    model = PPO.load(model_path)
    obs, _ = env.reset()
    rewards, entropy_vals = [], []

    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(action)
        rewards.append(reward)

        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        dist = model.policy.get_distribution(obs_tensor)
        entropy_vals.append(dist.entropy().mean().item())
        if done or truncated:
            obs, _ = env.reset()
    env.close()
    return rewards, entropy_vals

def test_reinforce(model_path, steps=50, n_runs=5):

    env = EnergyGridEnv()
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # Load policy network
    policy = PolicyNetwork(obs_size, n_actions)
    policy.load_state_dict(torch.load(model_path))
    policy.eval()

    rewards_all = []
    entropy_all = []

    for run in range(n_runs):
        obs, _ = env.reset()
        rewards, entropy_vals = [], []

        for _ in range(steps):
            obs_tensor = torch.FloatTensor(obs)
            action_probs = policy(obs_tensor)

            # Sample action probabilistically
            action = torch.multinomial(action_probs, num_samples=1).item()

            # Step environment
            obs, reward, done, truncated, _ = env.step(action)
            rewards.append(reward)

            # Compute policy entropy for this step
            entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-8)).item()
            entropy_vals.append(entropy)

            if done or truncated:
                obs, _ = env.reset()

        rewards_all.append(rewards)
        entropy_all.append(entropy_vals)

    env.close()

    # Average over multiple runs
    avg_rewards = np.mean(rewards_all, axis=0)
    avg_entropy = np.mean(entropy_all, axis=0)

    return avg_rewards, avg_entropy

def test_generalization(model_type, model_path, runs=5, steps=100):
    all_rewards = []
    for _ in range(runs):
        if model_type == "DQN":
            rewards, _ = test_dqn(model_path, steps)
        elif model_type == "PPO":
            rewards, _ = test_ppo(model_path, steps)
        elif model_type == "REINFORCE":
            rewards, _ = test_reinforce(model_path, steps)
        all_rewards.append(np.sum(rewards))
    return all_rewards

dqn_rewards, dqn_losses = test_dqn(best_dqn_model)
ppo_rewards, ppo_entropy = test_ppo(best_ppo_model)
reinforce_rewards, reinforce_entropy = test_reinforce(best_reinforce_model)

# Plot cumulative rewards subplots
fig, axs = plt.subplots(3, 1, figsize=(12, 12))
axs[0].plot(np.cumsum(dqn_rewards))
axs[0].set_title("DQN Cumulative Rewards")
axs[0].set_xlabel("Steps")
axs[0].set_ylabel("Cumulative Reward")

axs[1].plot(np.cumsum(ppo_rewards))
axs[1].set_title("PPO Cumulative Rewards")
axs[1].set_xlabel("Steps")
axs[1].set_ylabel("Cumulative Reward")

axs[2].plot(np.cumsum(reinforce_rewards))
axs[2].set_title("REINFORCE Cumulative Rewards")
axs[2].set_xlabel("Steps")
axs[2].set_ylabel("Cumulative Reward")

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "cumulative_rewards_subplots.png"))
plt.close()

# DQN loss curve
plt.figure(figsize=(10,6))
plt.plot(dqn_losses)
plt.title("DQN TD Variance (Approx. Loss)")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.savefig(os.path.join(plots_dir, "dqn_loss_curve.png"))
plt.close()

# Policy entropy curves
plt.figure(figsize=(10,6))
plt.plot(ppo_entropy, label="PPO")
plt.plot(reinforce_entropy, label="REINFORCE")
plt.title("Policy Entropy")
plt.xlabel("Steps")
plt.ylabel("Entropy")
plt.legend()
plt.savefig(os.path.join(plots_dir, "policy_entropy_curve.png"))
plt.close()

# Episodes to converge (moving average)
def moving_average(x, w=10):
    return np.convolve(x, np.ones(w)/w, mode='valid')

plt.figure(figsize=(12,6))
plt.plot(moving_average(dqn_rewards), label="DQN")
plt.plot(moving_average(ppo_rewards), label="PPO")
plt.plot(moving_average(reinforce_rewards), label="REINFORCE")
plt.title("Moving Average Rewards (Convergence)")
plt.xlabel("Steps")
plt.ylabel("Avg Reward (window=10)")
plt.legend()
plt.savefig(os.path.join(plots_dir, "episodes_to_converge.png"))
plt.close()

# Generalization plots
dqn_gen = test_generalization("DQN", best_dqn_model)
ppo_gen = test_generalization("PPO", best_ppo_model)
reinforce_gen = test_generalization("REINFORCE", best_reinforce_model)

plt.figure(figsize=(10,6))
plt.bar(["DQN","PPO","REINFORCE"], [np.mean(dqn_gen), np.mean(ppo_gen), np.mean(reinforce_gen)],
        yerr=[np.std(dqn_gen), np.std(ppo_gen), np.std(reinforce_gen)], capsize=5)
plt.title("Generalization Test - Sum of Rewards on Unseen Initial States")
plt.ylabel("Total Reward")
plt.savefig(os.path.join(plots_dir, "generalization_test.png"))
plt.close()

print(f"All plots saved in '{plots_dir}' folder.")