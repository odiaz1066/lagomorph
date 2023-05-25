import numpy as np
import gymnasium as gym

class RotateObservation(gym.ObservationWrapper):
    def __init__(self, env, frequency=0.5, degrees=90):
        super().__init__(env)
        self.frequency = frequency
        assert degrees % 90 == 0, "degrees must be a multiple of 90"
        self.rotations = degrees // 90
        
    def observation(self, obs):
        chance = np.random.uniform()
        if chance < self.frequency:
            obs = np.rot90(obs, self.rotations)
            return obs
        else:
            return obs