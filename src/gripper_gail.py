# %% 
# from stable_baselines3 import PPO, SAC
from sb3_contrib import PPO
from stable_baselines3.ppo import MlpPolicy as PPOMlpPolicy
from stable_baselines3.sac import MlpPolicy as SACMlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import matplotlib.pyplot as plt
import numpy as np

from sb3_contrib import PPO
from stable_baselines3.ppo import MlpPolicy
import gym

from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3.common.vec_env import DummyVecEnv

from imitation.data import types
import wandb
from imitation.util import logger as imit_logger
from imitation.scripts.common import wb
import pybulletgym
import pickle
import os
import logging
import sys
from imitation.scripts.train_adversarial import save

from imitation.util import util
with open('../jjh_data/expert_models/gripper_v1/final.pkl', 'rb') as f:
    rollouts = types.load(f)
#%%
from imitation.algorithms.adversarial.gail import GAIL
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.ppo import MlpPolicy 
import gym
import seals

if __name__ == "__main__":

    log_format_strs = ["wandb", "stdout"]
    def make_env(env_id, rank, seed=0):
        def _init():
            env = gym.make(env_id)
            env.seed(seed + rank)
            env = Monitor(env)
            return env
        return _init
    print(sys.argv)

    log_dir = os.path.join(
                "output",
                sys.argv[0].split(".")[0],
                util.make_unique_timestamp(),
            )
    os.makedirs(log_dir, exist_ok=True)
    print(sys.argv)
    if len(sys.argv) <2:
        name = None
    else:
        name = 'gail_' + sys.argv[1]

    wandb.init(project='great', sync_tensorboard=True, dir=log_dir, name=name)
    num_cpu=16

    custom_logger = imit_logger.configure(
        folder=os.path.join(log_dir, "log"),
        format_strs=log_format_strs,
    )
    venv = SubprocVecEnv( [make_env("Gripper-v1", i) for i in range(num_cpu)])
    learner = PPO(
        env=venv,
        policy=MlpPolicy,
        batch_size=128,
        ent_coef=0.001,
        learning_rate=0.0003,
        n_steps=int(50*2),
        # n_epochs=10,
        tensorboard_log='./logs/',
        device='cuda'
    )
    reward_net = BasicRewardNet(
        venv.observation_space, venv.action_space, normalize_input_layer=None,
    )
    gail_trainer = GAIL(
        demonstrations=rollouts,
        demo_batch_size=2*50*num_cpu,
        gen_replay_buffer_capacity=4*50*num_cpu,
        n_disc_updates_per_round=5,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
        custom_logger=custom_logger
    )

    eval_env = DummyVecEnv([lambda: gym.make("Gripper-v1")] * 1)
    # eval_env.render(mode='human')

    checkpoint_interval=10
    def cb(round_num):
        if checkpoint_interval > 0 and round_num % checkpoint_interval == 0:
            save(gail_trainer, os.path.join(log_dir, "checkpoints", f"{round_num:05d}"))
            obs = eval_env.reset()
            for i in range(500):
                action, _states = gail_trainer.gen_algo.predict(obs, deterministic=False)
                obs, _, _, _= eval_env.step(action)
                # eval_env.render(mode='human')

    gail_trainer.train(1000000, callback=cb)  # Note: set to 300000 for better results
    learner_rewards_after_training, _ = evaluate_policy(
        learner, venv, 100, return_episode_rewards=True
    )
    # %%
