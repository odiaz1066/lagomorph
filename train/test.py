import gymnasium as gym
import sys
sys.path.append(".")
import wrappers
from matplotlib import pyplot as plt

env = gym.make("BreakoutNoFrameskip-v4", render_mode="rgb_array")
env = gym.wrappers.ResizeObservation(env, (84, 84))
env = gym.wrappers.GrayScaleObservation(env)
env = wrappers.RotateObservation(env)
env.reset()
for step in range(100000):
    obs, _, _, _, _ = env.step(env.action_space.sample())
    plt.imshow(obs)
    plt.show()
    env.render()
    if step % 100 == 0:
        env.reset()
env.close()
