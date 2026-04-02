import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from environment.custom_env import EnergyGridEnv

# ----------------------------
# Create directories for models and logs
# ----------------------------
os.makedirs("models/dqn", exist_ok=True)
os.makedirs("logs/dqn", exist_ok=True)

# ----------------------------
# Define 10 different hyperparameter sets
# ----------------------------
hyperparameter_sets = [
    {"learning_rate": 1e-3, "gamma": 0.99, "batch_size": 32, "eps_start": 1.0, "eps_end": 0.05, },
    {"learning_rate": 5e-4, "gamma": 0.95, "batch_size": 64, "eps_start": 1.0, "eps_end": 0.1,},
    {"learning_rate": 1e-4, "gamma": 0.99, "batch_size": 32, "eps_start": 0.9, "eps_end": 0.05,},
    {"learning_rate": 1e-3, "gamma": 0.95, "batch_size": 64, "eps_start": 0.9, "eps_end": 0.1,},
    {"learning_rate": 5e-4, "gamma": 0.99, "batch_size": 32, "eps_start": 1.0, "eps_end": 0.05,},
    {"learning_rate": 1e-4, "gamma": 0.95, "batch_size": 64, "eps_start": 1.0, "eps_end": 0.1,},
    {"learning_rate": 5e-4, "gamma": 0.97, "batch_size": 32, "eps_start": 0.9, "eps_end": 0.05,},
    {"learning_rate": 1e-3, "gamma": 0.97, "batch_size": 64, "eps_start": 1.0, "eps_end": 0.05,},
    {"learning_rate": 1e-4, "gamma": 0.97, "batch_size": 32, "eps_start": 0.9, "eps_end": 0.1,},
    {"learning_rate": 5e-4, "gamma": 0.97, "batch_size": 64, "eps_start": 1.0, "eps_end": 0.1,},
]

# ----------------------------
# Training parameters
# ----------------------------
TIMESTEPS = 50000

# ----------------------------
# Loop through all experiments
# ----------------------------
for i, params in enumerate(hyperparameter_sets, start=1):
    print(f"\n===== Experiment {i} =====")
    print(f"Hyperparameters: {params}")

    # Initialize environment
    env = EnergyGridEnv()
    env = Monitor(env, f"logs/dqn/experiment_{i}")  # separate log folder per experiment

    # Add epsilon to hyperparameters dictionary
    eps_start = params.get("eps_start", 1.0)
    eps_end = params.get("eps_end", 0.05)

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=params["learning_rate"],
        gamma=params["gamma"],
        batch_size=params["batch_size"],
        buffer_size=50000,
        learning_starts=1000,
        train_freq=4,
        target_update_interval=1000,
        exploration_fraction=0.2,   # fraction of timesteps to decay epsilon
        exploration_initial_eps=eps_start,
        exploration_final_eps=eps_end,
        verbose=0,
    )

    # Train model
    model.learn(total_timesteps=TIMESTEPS)

    # Save model
    model_path = f"models/dqn/dqn_exp_{i}"
    model.save(model_path)
    print(f"Model saved to {model_path}.zip")

    # Test agent for a few steps and record rewards
    obs, _ = env.reset()  # discard info
    done = False
    episode_rewards = []

    for step in range(50):
        action, _ = model.predict(obs, deterministic=True)  # obs is now just the array
        obs, reward, done, truncated, info = env.step(action)  # unpack correctly
        episode_rewards.append(reward)
        env.render()
        if done or truncated:
            obs, _ = env.reset()
    env.close()

    # Calculate and print average reward
    avg_reward = np.mean(episode_rewards)
    print(f"Average reward for Experiment {i}: {avg_reward:.2f}")

    # Plot cumulative rewards for this experiment
    plt.figure()
    plt.plot(np.cumsum(episode_rewards))
    plt.xlabel("Step")
    plt.ylabel("Cumulative Reward")
    plt.title(f"DQN Experiment {i} - Cumulative Reward")
    plt.savefig(f"logs/dqn/dqn_exp_{i}_reward_curve.png")
    plt.close()