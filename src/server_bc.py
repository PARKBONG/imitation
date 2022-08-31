from email.errors import InvalidHeaderDefect
from imitation.algorithms.adversarial.gail import GAIL
from imitation.algorithms.adversarial.airl import AIRL 
from imitation.algorithms.bc import BC
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

import hydra
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from imitation.policies import serialize
from scipy import ndimage
def load_rollouts(dir):
    with open(dir, 'rb') as f:
        rollouts = types.load(f)
    return rollouts

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

    cart_width = 4.0 / (12 ** 0.5)
    cart_height = 1.0 / (12 ** 0.5)
    plate_width = 0.5
    plate_height = 0.2
    pole_width = 0.15
    pole_height = 0.8
    anchor_height = 0.1
    for itr in range(1):
        state = env.reset()

        obs_batch = []
        obs_action = []
        next_obs_batch = []

        plate_ang = 0.0
        num_y = 0
        for pos in np.arange(-1, 1, 0.05):
            num_y += 1
            num_x = 0
            for ang in np.arange(-1, 1, 0.05):
                num_x += 1
                obs = np.zeros(11)
                """
                    <state type="xpos" body="goal"/>    ## 0
                <state type="xpos" body="plate"/>   ## 1
                <state type="ypos" body="plate"/>   ## 2
                <state type="xvel" body="plate"/>   ## 3
                <state type="apos" body="plate"/>   ## 4   
                <state type="avel" body="plate"/>   ## 5
                <state type="xpos" body="pole"/>    ## 6
                <state type="ypos" body="pole"/>    ## 7
                <state type="xvel" body="pole"/>    ## 8
                <state type="apos" body="pole"/>    ## 9
                <state type="avel" body="pole"/>    ## 10
                """
                plate_x = pos
                
                pole_ang = ang
                mid_pole_x = plate_x - np.sin(plate_ang)*(plate_height/2)
                pole_x = mid_pole_x - (np.cos(pole_ang) - np.cos(plate_ang)) * (pole_width/2)
                
                obs[0] = 1.0
                obs[1] = pos
                obs[6] = pos
                obs[9] = ang
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

def visualize_reward_gt(env_id, log_dir, round_num=-1, tag='', use_wandb=False, ):
    import seaborn as sns
    import matplotlib.pyplot as plt
    grid_size = 0.1
    rescale = 1./grid_size

    cart_width = 4.0 / (12 ** 0.5)
    cart_height = 1.0 / (12 ** 0.5)
    plate_width = 0.5
    plate_height = 0.2
    pole_width = 0.15
    pole_height = 0.8
    anchor_height = 0.1
    for itr in range(1):
        obs_batch = []
        obs_action = []
        next_obs_batch = []
        rewards = []
        plate_ang = 0.0
        num_y = 0
        for pos in np.arange(-1, 1, 0.05):
            num_y += 1
            num_x = 0
            for ang in np.arange(-1, 1, 0.05):
                num_x += 1
                obs = np.zeros(11)
                """
                    <state type="xpos" body="goal"/>    ## 0
                <state type="xpos" body="plate"/>   ## 1
                <state type="ypos" body="plate"/>   ## 2
                <state type="xvel" body="plate"/>   ## 3
                <state type="apos" body="plate"/>   ## 4   
                <state type="avel" body="plate"/>   ## 5
                <state type="xpos" body="pole"/>    ## 6
                <state type="ypos" body="pole"/>    ## 7
                <state type="xvel" body="pole"/>    ## 8
                <state type="apos" body="pole"/>    ## 9
                <state type="avel" body="pole"/>    ## 10
                """
                plate_x = pos
                
                pole_ang = ang
                mid_pole_x = plate_x - np.sin(plate_ang)*(plate_height/2)
                pole_x = mid_pole_x - (np.cos(pole_ang) - np.cos(plate_ang)) * (pole_width/2)
                
                obs[0] = 0.7
                obs[1] = pos
                obs[6] = pos
                obs[9] = ang
                
                ucost = 1e-5*(ang**2)
                # print(self.contact)
                xcost = np.abs(pos - 0.7)
                xcost2 = float(np.abs(pos - 0.7) < 0.005)
                obs_batch.append(obs)
                reward = 1 * 2 -  1 *xcost + 1*xcost2*2 - 1 * ucost
                rewards.append(reward)
                # next_obs_batch.append(next_state)
                next_obs_batch.append(obs)
        irl_reward = np.array(rewards)
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
@hydra.main(config_path="config", config_name="common")
def main(cfg: DictConfig):
    n_envs = int(cfg.n_envs)
    total_steps = int(cfg.total_steps)
    is_wandb = bool(cfg.is_wandb)
    device = cfg.device
    
    env_id = cfg.env.env_id
    r_gamma = float(cfg.env.r_gamma)
    
    gen_lr = float(cfg.gen.lr)
    ent_coef = float(cfg.gen.ent_coef)
    target_kl = int(cfg.gen.target_kl)
    batch_size = int(cfg.gen.batch_size)
    n_epochs = int(cfg.gen.n_epochs)
    
    disc_lr = float(cfg.disc.lr)
    demo_batch_size = int(cfg.disc.demo_batch_size)
    gen_replay_buffer_capacity = int(cfg.disc.gen_replay_buffer_capacity)
    n_disc_updates_per_round = int(cfg.disc.n_disc_updates_per_round)
    hid_size = int(cfg.disc.hid_size)
    rollouts = load_rollouts(os.path.join(to_absolute_path('.'), "../jjh_data/expert_models/","serving_imit","final.pkl"))
    
    tensorboard_log = os.path.join(to_absolute_path('logs'), f"{cfg.gen.model}_{cfg.env.env_id}")

    log_format_strs = ["stdout"]
    if is_wandb:
        log_format_strs.append("wandb")
        
    def make_env(env_id, rank, seed=0):
        def _init():
            env = gym.make(env_id)
            env.seed(seed + rank)
            env = Monitor(env)
            return env
        return _init

    log_dir = os.path.join(
                "output",
                sys.argv[0].split(".")[0],
                util.make_unique_timestamp(),
            )
    os.makedirs(log_dir, exist_ok=True)
    
    if cfg.comment == "None":
        comment = ""
    else:
        comment = f"_{str(cfg.comment)}"
    name = 'irdd' + comment
    wandb.init(project='server', sync_tensorboard=True, dir=log_dir, name=name)
    # if "wandb" in log_format_strs:
    #     wb.wandb_init(log_dir=log_dir)
    custom_logger = imit_logger.configure(
        folder=os.path.join(log_dir, "log"),
        format_strs=log_format_strs,
    )
    #venv = DummyVecEnv([lambda: gym.make("Gripper-v0")] * 4)
    venv = SubprocVecEnv( [make_env(env_id, i) for i in range(1)])

    bc_trainer = BC(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        demonstrations=rollouts,
    )
    eval_env = DummyVecEnv([lambda: gym.make(env_id)] * 1)
    eval_env.render(mode='human')
    learner_rewards_before_training, _ = evaluate_policy(
        bc_trainer.policy, eval_env, 100, return_episode_rewards=True
    ) 
    bc_trainer.train(n_epochs=20)
    obs = eval_env.reset()
    for i in range(200):
        action, _states = bc_trainer.policy.predict(obs, deterministic=False)
        obs, _, _, _= eval_env.step(action)
        eval_env.render(mode='human')
        time.sleep(0.05)
            # visualize_reward(gail_trainer.gen_algo, lambda *args: gail_trainer.reward_train(*args)-gail_trainer.primary_train(*args), env_id,log_dir,  int(gail_trainer._disc_step), "constraint", True, )
    learner_rewards_after_training, _ = evaluate_policy(
        bc_trainer.policy, eval_env, 100, return_episode_rewards=True
    )
    print(learner_rewards_after_training )
    import matplotlib.pyplot as plt
    plt.hist(
        [learner_rewards_before_training, learner_rewards_after_training],
        label=["untrained", "trained"],
    )
    plt.legend()
    plt.show()
if __name__ == '__main__':
    #main(env_id=env_id)
    main()