from email.errors import InvalidHeaderDefect
from imitation.algorithms.adversarial.gail import GAIL
from imitation.algorithms.adversarial.airl import AIRL 

from imitation.algorithms.adversarial.irdd import IRDD 
from imitation.rewards.reward_nets import BasicRewardNet, BasicShapedRewardNet, NormalizedRewardNet, ScaledRewardNet, ShapedScaledRewardNet
from imitation.util.networks import RunningNorm
# from stable_baselines3 import PPO
from sb3_contrib import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.ppo import MlpPolicy 
import gym
import seals
import torch
from imitation.data import types
import wandb
from imitation.util import logger as imit_logger
from imitation.scripts.common import wb
import pybulletgym
import pickle
import os
import logging

from imitation.scripts.train_adversarial import save
with open('../jjh_data/peginhole_v1_0/rollouts/final3.pkl', 'rb') as f:
    rollouts = types.load(f)

if __name__ == '__main__':

    log_format_strs = ["wandb", "stdout"]
    def make_env(env_id, rank, seed=0):
        def _init():
            env = gym.make(env_id)
            env.seed(seed + rank)
            env = Monitor(env)
            return env
        return _init

    log_dir = "./log/"
    wandb.init(project='good', sync_tensorboard=True, dir=log_dir)

    # if "wandb" in log_format_strs:
    #     wb.wandb_init(log_dir=log_dir)
    custom_logger = imit_logger.configure(
        folder=os.path.join(log_dir, "log"),
        format_strs=log_format_strs,
    )
    #venv = DummyVecEnv([lambda: gym.make("Gripper-v0")] * 4)
    venv = SubprocVecEnv( [make_env("Gripper-v0", i) for i in range(16)])
    learner = PPO(
        env=venv,
        policy=MlpPolicy,
        # batch_size=64,
        # n_steps=512,
        ent_coef=0.01,
        learning_rate=0.0003,
        n_epochs=20,
        # n_epochs=1,
        n_steps=int(2048/32),
        tensorboard_log='./logs/',
        device='cpu',
    )
    print(learner.n_epochs)
    reward_fn = lambda s, a, ns, d: torch.norm(ns[...,1:3], dim=-1, keepdim=False) 
    reward_net = ScaledRewardNet(
        venv.observation_space, venv.action_space, reward_fn =reward_fn, normalize_input_layer=None,
    )
    reward_net = BasicShapedRewardNet(
        venv.observation_space, venv.action_space, normalize_input_layer=None,
    )
    reward_net = NormalizedRewardNet(
        base=reward_net, normalize_output_layer=RunningNorm,
    )
    constraint_net = ShapedScaledRewardNet(
        venv.observation_space, venv.action_space,reward_fn =reward_fn, normalize_input_layer=None,
    )
    gail_trainer = IRDD(
        demonstrations=rollouts,
        demo_batch_size=512,
        gen_replay_buffer_capacity=1024,
        n_disc_updates_per_round=10,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
        # disc_opt_kwargs={"lr":0.001},
        log_dir=log_dir,
        constraint_net=constraint_net,
        # const_disc_opt_kwargs={"lr":0.001}
        custom_logger=custom_logger
    )

    # learner_rewards_before_training, _ = evaluate_policy(
    #     learner, venv, 100, return_episode_rewards=True
    # )
    # print(learner_rewards_before_training)
    checkpoint_interval=5
    def cb(round_num):
        if checkpoint_interval > 0 and round_num % checkpoint_interval == 0:
            save(gail_trainer, os.path.join(log_dir, "checkpoints", f"{round_num:05d}"))

    gail_trainer.train(int(1e6), callback=cb)  # Note: set to 300000 for better results
    learner_rewards_after_training, _ = evaluate_policy(
        learner, venv, 100, return_episode_rewards=True
    )
    print(learner_rewards_after_training )
