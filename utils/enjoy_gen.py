import argparse
from distutils.util import strtobool
import random
import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
from typing import Callable
import sys
sys.path.append(".")
import envs

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="dqn_atari",
        help="the name of this experiment (e.g., ppo, dqn_atari)")
    parser.add_argument("--graphics", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether the environment uses visual environments or not")
    parser.add_argument("--seed", type=int, default=42,
        help="seed of the experiment")
    parser.add_argument("--hf-entity", type=str, default="cleanrl",
        help="the user or org name of the model repository from the Hugging Face Hub")
    parser.add_argument("--hf-repository", type=str, default="",
        help="the huggingface repo (e.g., cleanrl/BreakoutNoFrameskip-v4-dqn_atari-seed1)")
    parser.add_argument("--env-id", type=str, default="BreakoutNoFrameskip-v4",
        help="the id of the environment")
    parser.add_argument("--eval-episodes", type=int, default=10,
        help="the number of evaluation episodes")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")
    args = parser.parse_args()
    # fmt: on
    return args

class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x)
    
class VisualQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            #nn.Linear(512, env.single_action_space.n),
            nn.Linear(512, env.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x / 255.0)
    
def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if args.graphics:
            from stable_baselines3.common.atari_wrappers import (
                ClipRewardEnv,
                EpisodicLifeEnv,
                FireResetEnv,
                MaxAndSkipEnv,
                NoopResetEnv,
            )
            env = NoopResetEnv(env, noop_max=30)
            env = MaxAndSkipEnv(env, skip=4)
            env = EpisodicLifeEnv(env)
            if "FIRE" in env.unwrapped.get_action_meanings():
                env = FireResetEnv(env)
            env = ClipRewardEnv(env)
            env = gym.wrappers.ResizeObservation(env, (84, 84))
            env = gym.wrappers.GrayScaleObservation(env)
            env = gym.wrappers.FrameStack(env, 4)
        env.action_space.seed(seed)

        return env

    return thunk

def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    epsilon: float = 0.05,
    capture_video: bool = True,
):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, 0, capture_video, run_name)])
    model = Model(envs).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    obs, _ = envs.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = model(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()
        next_obs, _, _, _, infos = envs.step(actions)
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" in info:
                    print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                    episodic_returns += [info["episode"]["r"]]
        obs = next_obs

    return episodic_returns

if __name__ == "__main__":
    args = parse_args()
    if args.graphics:
        Model = VisualQNetwork
    else:
        Model = QNetwork
    
    if not args.hf_repository:
        args.hf_repository = f"{args.hf_entity}/{args.env_id}-{args.exp_name}-seed{args.seed}"
    print(f"loading saved models from {args.hf_repository}...")
    model_path = f"./models/{args.exp_name}.pt"
    evaluate(
        model_path,
        make_env,
        args.env_id,
        eval_episodes=args.eval_episodes,
        run_name=f"eval/{args.exp_name}",
        Model=Model,
        capture_video=args.capture_video,
    )
