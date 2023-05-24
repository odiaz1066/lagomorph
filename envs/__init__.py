from envs.cartpole_goal import CartPoleGoalEnv
from gymnasium.envs.registration import register

register(
    id="Lagomorph/CartPoleGoal-v1",
    entry_point="envs:CartPoleGoalEnv",
    max_episode_steps=500,
    reward_threshold=475.0,
)