import numpy as np
import time
from stable_baselines3 import DQN
from environment.custom_env import EnergyGridEnv

BEST_MODEL_PATH = "models/dqn/dqn_exp_10.zip"


def run(num_episodes=5, step_delay=0.5):
    print("\nRunning DQN...")
    env = EnergyGridEnv()
    model = DQN.load(BEST_MODEL_PATH)

    all_rewards = []

    for episode in range(1, num_episodes + 1):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False

        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            env.render()
            time.sleep(step_delay)  # <-- slows down each step

        all_rewards.append(episode_reward)
        print(f"Episode {episode} Reward: {episode_reward:.2f}")

    env.close()
    avg_reward = np.mean(all_rewards)
    print(f"\nAverage Reward over {num_episodes} episodes: {avg_reward:.2f}")


if __name__ == "__main__":
    print("===== Running Best Models =====")
    run(num_episodes=5, step_delay=0.5)
    print("\nDone.")