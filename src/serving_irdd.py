from email.errors import InvalidHeaderDefect
from imitation.algorithms.adversarial.gail import GAIL
from imitation.algorithms.adversarial.airl import AIRL 

from imitation.algorithms.adversarial.irdd import IRDD 
from imitation.algorithms.adversarial.irdd2 import IRDD2
from imitation.algorithms.adversarial.irdd3 import IRDD3
from imitation.rewards.reward_nets import BasicRewardNet, BasicShapedRewardNet, NormalizedRewardNet, ScaledRewardNet, ShapedScaledRewardNet, PredefinedRewardNet, DropoutRewardNet
from imitation.util.networks import RunningNorm
# from stable_baselines3 import PPO
from sb3_contrib import PPO
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
import sys
from imitation.util import util
# from imitation.scripts.train_adversarial import save
import torch.nn as nn
import torch as th
import numpy as np
import time
import math
from imitation.rewards.reward_nets import RewardNetWrapper

from imitation.policies import serialize
from scipy import ndimage
with open('../jjh_data/expert_models/serving/final.pkl', 'rb') as f:
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

def visualize_reward(model, reward_net, env_id, log_dir, round_num, tag='', use_wandb=False, ):
    import seaborn as sns
    import matplotlib.pyplot as plt
    env = gym.make(env_id)
    grid_size = 0.1
    rescale = 1./grid_size
    boundary_low = -2.5
    boundary_high = 2.5
    barrier_range = [0.2, 0.6]
    barrier_y = 0.3
    cart_height = 1.0 / (12 ** 0.5)
    pole_height = 0.9
    plate_height = 0.2
        
    for itr in range(1):
        state = env.reset()

        obs_batch = []
        obs_action = []
        next_obs_batch = []

        num_y = 0
        for pos in np.arange(-1.2, 1.2, 0.1):
            num_y += 1
            num_x = 0
            for ang in np.arange(-0.3, 0.3, 0.025):
                num_x += 1
                obs = np.zeros(11)
                
                cart_x = pos
                plate_ang = ang
                #plate_x = cart_x  np.sin(plate_ang) * (cart_height + plate_height/2)# - cart_height*2.0 - plate_height - pole_height/2)
                plate_x = cart_x + np.sin(plate_ang) * (1.0 * cart_height + plate_height/2)
                pole_x = cart_x + np.sin(plate_ang) * (pole_height/2 - cart_height + plate_height/2)
                
                # <state type="xvel" body="cart"/>
                obs[1] = cart_x# <state type="xpos" body="cart"/>
                obs[2] = plate_x# <state type="xpos" body="plate"/>
                # <state type="xvel" body="plate"/>
                obs[4] = plate_ang # <state type="apos" body="plate"/>
                # <state type="avel" body="plate"/>
                obs[6] = 1.0 # <state type="xpos" body="goal"/>
                obs[7] = pole_x# <state type="xpos" body="pole"/>
                # <state type="xvel" body="pole"/>
                obs[9] = plate_ang*2 # <state type="apos" body="pole"/>
                # <state type="avel" body="pole"/>
                obs_batch.append(obs)

                action, _ = model.predict(obs, deterministic=True)
                # next_state, reward, done, _ = env.step(action)

                obs_action.append(action)
                # next_obs_batch.append(next_state)
                next_obs_batch.append(obs)

        obs_batch = np.array(obs_batch)
        next_obs_batch = np.array(next_obs_batch)
        obs_action = np.array(obs_action)

        # Get sqil reward
        with torch.no_grad():
            state = torch.FloatTensor(obs_batch).to(model.device)
            action = torch.FloatTensor(obs_action).to(model.device)
            next_state = torch.FloatTensor(next_obs_batch).to(model.device)

        done = torch.zeros_like(state[...,-1:])

        with torch.no_grad():
            irl_reward = reward_net(state, action, next_state, done)

            irl_reward = irl_reward.cpu().numpy()
        score = irl_reward

        score = irl_reward
        flights = score.copy().reshape([num_x, num_y])
        ax = sns.heatmap(score.reshape([num_x, num_y]), cmap="YlGnBu_r")
        # ax.scatter((target[0]-boundary_low)*rescale, (target[1]-boundary_low)
        #            * rescale, marker='*', s=150, c='r', edgecolors='k', linewidths=0.5)
        # ax.scatter((0.3-boundary_low + np.random.uniform(low=-0.05, high=0.05))*rescale, (0.-boundary_low +
        #                                                                                   np.random.uniform(low=-0.05, high=0.05))*rescale, marker='o', s=120, c='white', linewidths=0.5, edgecolors='k')
        # ax.plot([(barrier_range[0] - boundary_low)*rescale, (barrier_range[1] - boundary_low)*rescale], [(barrier_y - boundary_low)*rescale, (barrier_y - boundary_low)*rescale],
        #         color='k', linewidth=10)
        ax.invert_yaxis()
        plt.axis('off')
        # plt.show()
        smooth_scale = 10
        z = ndimage.zoom(flights, smooth_scale)
        contours = np.linspace(np.min(score), np.max(score), 9)
        cntr = ax.contour(np.linspace(0, num_x, num_x * smooth_scale),
                        np.linspace(0, num_y, num_y * smooth_scale),
                        z, levels=contours[:-1], colors='red')
        ax.invert_yaxis()
        plt.axis('off')
        if use_wandb:
            wandb.log({f"rewards_map/{tag}": wandb.Image(plt)}, step=round_num)
        print(score.reshape([num_x, num_y]))
        savedir = os.path.join(log_dir,"maps")
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        print(savedir)
        plt.savefig(savedir + '/%s_%s.png' % (itr, tag))
        print('Save Itr', itr)
        plt.close()


if __name__ == '__main__':
    
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
        name = 'irdd_' + sys.argv[1]
    
    wandb.init(project='prelim', sync_tensorboard=True, dir=log_dir, name=name)
    # if "wandb" in log_format_strs:
    #     wb.wandb_init(log_dir=log_dir)
    custom_logger = imit_logger.configure(
        folder=os.path.join(log_dir, "log"),
        format_strs=log_format_strs,
    )
    #venv = DummyVecEnv([lambda: gym.make("Gripper-v0")] * 4)
    venv = SubprocVecEnv( [make_env("Serving-v0", i) for i in range(16)])
    learner = PPO2(
        env=venv,
        policy=MlpPolicy,
        batch_size=64,
        # n_steps=512,
        ent_coef=0.01,
        learning_rate=0.0003,
        #n_epochs=80,
        n_epochs=30,
        target_kl=0.02,
        # n_steps=int(2048/32),
        policy_kwargs={'optimizer_class':th.optim.Adam},
        tensorboard_log='./logs/',
        device='cpu',
    )
    print(learner.n_epochs)
    def reward_fn(s, a, ns, d):
        #return torch.norm(s[...,2:3], dim=-1, keepdim=False)  
        # print(torch.norm(s[...,2:3], dim=-1, keepdim=False).shape  )
        # print(s[...,2].shape)
        # return torch.cat([s[...,2:3],ns[...,2:3]],dim=1)
        # print(torch.cat([s[...,0:1], s[...,6:7]], dim=-1).shape)
        # print(s[...,0:1].shape)
        # exit()
        #return torch.cat([s[...,1:2], s[...,6:7]], dim=-1)
        return s[...,6:9]    #reward_fn = lambda s, a, ns, d: torch.norm(ns[...,1:3], dim=-1, keepdim=False) 
    reward_net = BasicShapedRewardNet(
        venv.observation_space, venv.action_space, normalize_input_layer=None,
        # potential_hid_sizes=[8, 8],
        # reward_hid_sizes=[8, 8],
    )
    reward_net = BasicRewardNet(
        venv.observation_space, venv.action_space, normalize_input_layer=None,#RunningNorm,
        # hid_sizes=[8, 8],

    )
    constraint_net = BasicRewardNet(
        venv.observation_space, venv.action_space, normalize_input_layer=None,#RunningNorm,
        # hid_sizes=[8, 8],

    )
    primary_net = PredefinedRewardNet(
            venv.observation_space, venv.action_space, reward_fn=reward_fn, combined_size=3, use_action=True, normalize_input_layer=None, #RunningNorm, #RunningNorm,
        # hid_sizes=[8, 8],
        # potential_hid_sizes=[8, 8],
    )
    # reward_net = ShapedScaledRewardNet(
    #     venv.observation_space, venv.action_space,reward_fn =reward_fn, normalize_input_layer=None,
    #     potential_hid_sizes=[8, 8],
    # )
    gail_trainer = IRDD3(
        demonstrations=rollouts,
        demo_batch_size=2000,
        gen_replay_buffer_capacity=4000,
        n_disc_updates_per_round=20,
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
    # print(learner_rewards_before_training)

    eval_env = DummyVecEnv([lambda: gym.make("Serving-v0")] * 1)
    eval_env.render(mode='human')
    checkpoint_interval=3
    def cb(round_num):
        if checkpoint_interval > 0 and round_num % checkpoint_interval == 0:
            save(gail_trainer, os.path.join(log_dir, "checkpoints", f"{round_num:05d}"))
            obs = eval_env.reset()
            for i in range(200):
                action, _states = gail_trainer.gen_algo.predict(obs, deterministic=False)
                obs, _, _, _= eval_env.step(action)
                eval_env.render(mode='human')
                time.sleep(0.05)
            visualize_reward(gail_trainer.gen_algo, lambda *args: gail_trainer.reward_train(*args)-gail_trainer.primary_train(*args), "CartPole-Const-v0",log_dir,  int(gail_trainer._disc_step), "constraint", True, )
            visualize_reward(gail_trainer.gen_algo, gail_trainer.primary_train, "CartPole-Const-v0",log_dir,  int(gail_trainer._disc_step), "primary", True, )
            # visualize_reward(gail_trainer.gen_algo, gail_trainer.constraint_train, "CartPole-Const-v0",log_dir,  str(round_num)+"total", True, )
    gail_trainer.train(int(10e6), callback=cb)  # Note: set to 300000 for better results
    learner_rewards_after_training, _ = evaluate_policy(
        learner, venv, 100, return_episode_rewards=True
    )
    print(learner_rewards_after_training )
