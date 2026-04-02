import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from environment.custom_env import EnergyGridEnv

# ----------------------------
# Create directories for models and logs
# ----------------------------
os.makedirs("models/ppo", exist_ok=True)
os.makedirs("logs/ppo", exist_ok=True)

# ----------------------------
# Define 10 hyperparameter sets for PPO
# ----------------------------
hyperparameter_sets = [
    {"learning_rate": 3e-4, "gamma": 0.99, "n_steps": 128, "ent_coef": 0.0, "batch_size": 64},
    {"learning_rate": 1e-3, "gamma": 0.99, "n_steps": 256, "ent_coef": 0.01, "batch_size": 64},
    {"learning_rate": 5e-4, "gamma": 0.97, "n_steps": 128, "ent_coef": 0.01, "batch_size": 64},
    {"learning_rate": 1e-3, "gamma": 0.97, "n_steps": 256, "ent_coef": 0.0, "batch_size": 128},
    {"learning_rate": 3e-4, "gamma": 0.95, "n_steps": 128, "ent_coef": 0.01, "batch_size": 64},
    {"learning_rate": 5e-4, "gamma": 0.95, "n_steps": 256, "ent_coef": 0.0, "batch_size": 128},
    {"learning_rate": 1e-3, "gamma": 0.99, "n_steps": 128, "ent_coef": 0.01, "batch_size": 64},
    {"learning_rate": 3e-4, "gamma": 0.97, "n_steps": 256, "ent_coef": 0.0, "batch_size": 128},
    {"learning_rate": 5e-4, "gamma": 0.97, "n_steps": 128, "ent_coef": 0.01, "batch_size": 64},
    {"learning_rate": 1e-3, "gamma": 0.95, "n_steps": 256, "ent_coef": 0.01, "batch_size": 128},
]

TIMESTEPS = 200000

# ----------------------------
# Run experiments
# ----------------------------
for i, params in enumerate(hyperparameter_sets, start=1):
    print(f"\n===== PPO Experiment {i} =====")
    print(f"Hyperparameters: {params}")

    # Initialize environment
    env = EnergyGridEnv()
    env = Monitor(env, f"logs/ppo/experiment_{i}")

    # Initialize PPO model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=params["learning_rate"],
        gamma=params["gamma"],
        n_steps=params["n_steps"],
        batch_size=params["batch_size"],
        ent_coef=params["ent_coef"],
        verbose=0
    )

    # Train model
    model.learn(total_timesteps=TIMESTEPS)

    # Save model
    model_path = f"models/ppo/ppo_exp_{i}"
    model.save(model_path)
    print(f"PPO model saved to {model_path}.zip")

    # Test agent briefly
    obs, _ = env.reset() 

    done = False
    truncated = False
    episode_rewards = []

    for step in range(50):
        # Get action from the trained model
        action, _ = model.predict(obs, deterministic=True)
        
        # Step the environment
        obs, reward, done, truncated, info = env.step(action)
        episode_rewards.append(reward)
        
        # Render environment (optional)
        env.render()
        
        # Reset environment if episode ends
        if done or truncated:
            obs, _ = env.reset()
    env.close()

    # Compute and print average reward
    avg_reward = sum(episode_rewards) / len(episode_rewards)
    print(f"Average reward: {avg_reward:.2f}")

    # Plot cumulative rewards
    plt.figure()
    plt.plot(np.cumsum(episode_rewards))
    plt.xlabel("Step")
    plt.ylabel("Cumulative Reward")
    plt.title(f"PPO Experiment {i} - Cumulative Reward")
    plt.savefig(f"logs/ppo/ppo_exp_{i}_reward_curve.png")
    plt.close()