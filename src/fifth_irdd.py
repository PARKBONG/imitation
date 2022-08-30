from email.errors import InvalidHeaderDefect
from imitation.algorithms.adversarial.gail import GAIL
from imitation.algorithms.adversarial.airl import AIRL 

from imitation.algorithms.adversarial.irdd import IRDD 
from imitation.algorithms.adversarial.irdd2 import IRDD2
from imitation.algorithms.adversarial.irdd3 import IRDD3
from imitation.rewards.reward_nets import BasicRewardNet, BasicShapedRewardNet, NormalizedRewardNet, ScaledRewardNet, ShapedScaledRewardNet, PredefinedRewardNet, DropoutRewardNet
from imitation.util.networks import RunningNorm
from stable_baselines3 import PPO,SAC
# from sb3_contrib import PPO
from sb3_contrib import PPO2
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
import time
import sys
from imitation.util import util
# from imitation.scripts.train_adversarial import save
import torch.nn as nn
import torch as th
import numpy as np
import math
from imitation.rewards.reward_nets import RewardNetWrapper
import seaborn as sns
import matplotlib.pyplot as plt
from imitation.policies import serialize
from scipy import ndimage
with open('../jjh_data/expert_models/gripper_v1/final.pkl', 'rb') as f:
    rollouts = types.load(f)

def save(trainer, save_path):
    """Save discriminator and generator."""
    # We implement this here and not in Trainer since we do not want to actually
    # serialize the whole Trainer (including e.g. expert demonstrations).
    os.makedirs(save_path, exist_ok=True)
    th.save(trainer.reward_train, os.path.join(save_path, "reward_train.pt"))
    th.save(trainer.reward_test, os.path.join(save_path, "reward_test.pt"))

    if hasattr(trainer, "primary_train"):
        saving_net_train = trainer.primary_train
        saving_net_test = trainer.primary_test
        while isinstance(saving_net_train, RewardNetWrapper) and hasattr(saving_net_train, "base"):
            saving_net_train = saving_net_train.base
        while isinstance(saving_net_test, RewardNetWrapper) and hasattr(saving_net_test, "base"):
            saving_net_test = saving_net_test.base
        th.save(saving_net_train.mlp, os.path.join(save_path, "primary_train.pt"))
        th.save(saving_net_test.mlp, os.path.join(save_path, "primary_test.pt"))

    if hasattr(trainer, "constraint_train"):
        saving_net_train = trainer.constraint_train
        saving_net_test = trainer.constraint_test
        while isinstance(saving_net_train, RewardNetWrapper) and hasattr(saving_net_train, "base"):
            saving_net_train = saving_net_train.base
        while isinstance(saving_net_test, RewardNetWrapper) and hasattr(saving_net_test, "base"):
            saving_net_test = saving_net_test.base
        th.save(saving_net_train.mlp, os.path.join(save_path, "constraint_test.pt"))
        th.save(saving_net_test.mlp, os.path.join(save_path, "constraint_train.pt"))
    serialize.save_stable_model(
        os.path.join(save_path, "gen_policy"),
        trainer.gen_algo,
    )

with open('../jjh_data/expert_models/gripper_v1/final.pkl', 'rb') as f:
    rollouts = types.load(f)

model = SAC.load("../jjh_data/expert_models/gripper_v1/model.zip")
eval_env = DummyVecEnv([lambda: gym.make("Gripper-v1")] * 2)
old_obs = eval_env.reset()


obs_batch = []
obs_action = []
next_obs_batch = []
pos_xs = []
pos_ys = []
r_batch = []
for _ in range(10):
    a = time.time()
    
    for i in range(50):
        action, _states = model.predict(old_obs, deterministic=False)

        obs, rewards, dones, _= eval_env.step(action)
        if dones.any():
            continue
        obs_batch.append(old_obs)
        next_obs_batch.append(obs)
        obs_action.append(action)
        r_batch.append(rewards)
        old_obs = obs
        # print(i, end=' ')
        pos_x = old_obs[...,1]
        pos_y = old_obs[...,2]
        pos_xs.append(pos_x)
        pos_ys.append(pos_y)
    print(time.time() - a)



state = np.array(obs_batch)
next_state = np.array(next_obs_batch)
action = np.array(obs_action)

r_batch = np.array(r_batch) 
r_batch = r_batch.reshape(-1,1)
pos_xs = np.array(pos_xs)
pos_ys = np.array(pos_ys)

pos_xs = pos_xs.reshape(-1,1)
pos_ys = pos_ys.reshape(-1,1)
# Get sqil reward
done = np.zeros_like(r_batch)

    # print(eval_env.envs[0].observation_space.shape)
state = state.reshape(-1, eval_env.envs[0].observation_space.shape[0])
next_state = next_state.reshape(-1, eval_env.envs[0].observation_space.shape[0])
action = action.reshape(-1, eval_env.envs[0].action_space.shape[0])
    
del eval_env
plt.scatter(pos_xs,pos_ys,s=20,c=r_batch, marker = 'o', cmap="YlGnBu_r" )
plt.show()
def visualize_reward(reward_net, log_dir, round_num, tag='', use_wandb=False, ):

    with torch.no_grad():
        state = torch.FloatTensor(obs_batch).to(model.device)
        action = torch.FloatTensor(obs_action).to(model.device)
        next_state = torch.FloatTensor(next_obs_batch).to(model.device)
        done = torch.Tensor(done, dtype=torch.bool).to(model.device)
    with torch.no_grad():
        irl_reward = reward_net(state, action, next_state, done)

        irl_reward = irl_reward.cpu().numpy()
    score = irl_reward

    score = irl_reward
    plt.scatter(pos_xs,pos_ys,s=20,c=score, marker = 'o', cmap="YlGnBu_r" )
    ax.invert_yaxis()
    plt.axis('off')
    ax.invert_yaxis()
    plt.axis('off')
    if use_wandb:
        wandb.log({f"rewards_map/{tag}": wandb.Image(plt)}, step=round_num)
    savedir = os.path.join(log_dir,"maps")
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    print(savedir)
    plt.savefig(savedir + '/%s_%s.png' % (itr, tag))
    print('Save Itr', itr)
    plt.close()


if __name__ == '__main__':
    
    log_format_strs = ["stdout"]
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
        name = 'irdd_' + sys.argv[1]
    
    # wandb.init(project='great', sync_tensorboard=True, dir=log_dir, name=name)
    # if "wandb" in log_format_strs:
    #     wb.wandb_init(log_dir=log_dir)
    custom_logger = imit_logger.configure(
        folder=os.path.join(log_dir, "log"),
        format_strs=log_format_strs,
    )
    #venv = DummyVecEnv([lambda: gym.make("Gripper-v0")] * 4)
    num_cpu=4
    venv = SubprocVecEnv( [make_env("Gripper-v1", i) for i in range(num_cpu)])
    learner = PPO2(
        env=venv,
        policy=MlpPolicy,
        batch_size=64,
        # n_steps=512,
        ent_coef=0.01,
        learning_rate=0.0003,
        #n_epochs=80,
        # n_epochs=20,
        n_steps=int(2 * 50 * num_cpu),
        policy_kwargs={'optimizer_class':th.optim.Adam},
        tensorboard_log='./logs/',
        device='cpu',
    )
    print(learner.n_epochs)
    def reward_fn(s, a, ns, d):
        #return torch.norm(s[...,2:3], dim=-1, keepdim=False)  
        # print(torch.norm(s[...,2:3], dim=-1, keepdim=False).shape  )
        # print(s[...,2].shape)
        return s[...,-3:]
    #reward_fn = lambda s, a, ns, d: torch.norm(ns[...,1:3], dim=-1, keepdim=False) 
    reward_net = BasicShapedRewardNet(
        venv.observation_space, venv.action_space, normalize_input_layer=None,
        # potential_hid_sizes=[8, 8],
        # reward_hid_sizes=[8, 8],
    )
    reward_net = BasicRewardNet(
        venv.observation_space, venv.action_space, normalize_input_layer=None,#RunningNorm,
        hid_sizes=[8, 8],

    )
    constraint_net = BasicRewardNet(
        venv.observation_space, venv.action_space, normalize_input_layer=None,#RunningNorm,
        hid_sizes=[8, 8],

    )
    primary_net = PredefinedRewardNet(
        venv.observation_space, venv.action_space, reward_fn=reward_fn, combined_size=3,normalize_input_layer=None, #RunningNorm, #RunningNorm,
        hid_sizes=[8, 8],
        # potential_hid_sizes=[8, 8],
    )
    # reward_net = ShapedScaledRewardNet(
    #     venv.observation_space, venv.action_space,reward_fn =reward_fn, normalize_input_layer=None,
    #     potential_hid_sizes=[8, 8],
    # )
    gail_trainer = IRDD3(
        demonstrations=rollouts,
        demo_batch_size=2*50*num_cpu,
        gen_replay_buffer_capacity=4 * 50 * num_cpu,
        n_disc_updates_per_round=1,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
        # disc_opt_kwargs={"lr":0.001},
        log_dir=log_dir,
        primary_net=primary_net,
        constraint_net=constraint_net,
        disc_opt_cls=th.optim.Adam,
        # const_disc_opt_kwargs={"lr":0.001}
        custom_logger=custom_logger
    )

    # learner_rewards_before_training, _ = evaluate_policy(
    #     learner, venv, 100, return_episode_rewards=True
    # )
    # print(learner_rewards_befor

    checkpoint_interval=10
    def cb(round_num):
        if checkpoint_interval > 0 and round_num % checkpoint_interval == 0:
            save(gail_trainer, os.path.join(log_dir, "checkpoints", f"{round_num:05d}"))
            # visualize_reward(gail_trainer.gen_algo, lambda *args: gail_trainer.reward_train(*args)-gail_trainer.primary_train(*args), "CartPole-Const-v0",log_dir, round_num, "constraint", True, )
            visualize_reward(gail_trainer.primary_train, log_dir,  round_num, "primary", True, )
            visualize_reward(gail_trainer.constraint_train, log_dir,  str(round_num)+"total", True, )
    gail_trainer.train(int(10e6), callback=cb)  # Note: set to 300000 for better results
    learner_rewards_after_training, _ = evaluate_policy(
        learner, venv, 100, return_episode_rewards=True
    )
    print(learner_rewards_after_training )
