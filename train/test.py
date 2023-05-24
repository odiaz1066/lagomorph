import gymnasium as gym
import sys
sys.path.append(".")
import envs

env = gym.make("Lagomorph/CartPoleGoal-v1", render_mode="human")
env.reset()
for step in range(100000):
    env.step(env.action_space.sample())
    env.render()
    if step % 100 == 0:
        env.reset()
env.close()
