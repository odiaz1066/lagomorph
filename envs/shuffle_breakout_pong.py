"""
This environment is basically a wrapper for both BreakoutNoFrameskip-v4 and PongNoFrameskip-v4.
Every 10 episodes, the environment will switch between Breakout and Pong.
"""

import gymnasium as gym

class ShuffleBreakoutPongEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "obs_types": ["rgb", "grayscale", "ram"]
    }

    def __init__(self, render_mode="human", obs_type="rgb"):
        pass