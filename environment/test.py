from custom_env import EnergyGridEnv
import time

env = EnergyGridEnv()
obs, _ = env.reset()

for _ in range(20):
    action = env.action_space.sample()
    obs, reward, done, truncated, _ = env.step(action)
    env.render()
    print(f"Reward: {reward:.2f}, Step: {env.current_step}")
    time.sleep(0.5)

env.close()