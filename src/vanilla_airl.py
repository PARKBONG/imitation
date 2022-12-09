from imitation.algorithms.adversarial.ird import IRD 
from imitation.rewards.reward_nets import BasicRewardNet, BasicShapedRewardNet, NormalizedRewardNet, ScaledRewardNet, ShapedScaledRewardNet, PredefinedRewardNet, DropoutRewardNet
from imitation.util.networks import RunningNorm
from sb3_contrib import PPO2, SACLagrangian
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.sac import MlpPolicy 
import gym
from imitation.data import types
import wandb
from imitation.util import logger as imit_logger
import os
import sys
from imitation.util import util
import torch as th
import time
from sb3_contrib.common.monitor import ConstMonitor
from stable_baselines3.common.utils import safe_mean, should_collect_more_steps
from imitation.algorithms.adversarial.airl import AIRL
import hydra
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from util import visualize_reward, save, visualize_reward_gt, plot_reward
import numpy as np
import imitation
import matplotlib.pyplot as plt

from imitation.policies import base
def load_rollouts(dir):
    with open(dir, 'rb') as f:
        rollouts = types.load(f)
    return rollouts

def make_env(env_id, rank, seed=0):
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        env = ConstMonitor(env)
        return env
    return _init

def reward_fn(s, a, ns, d):
    return s[...,[8]]    
combined_size  = 1
@hydra.main(config_path="config", config_name="common")
def main(cfg: DictConfig):
    
    normalize_layer = {"None":None, "RunningNorm":RunningNorm}
    opt_cls = {"None":None, "Adam":th.optim.Adam, "AdamW": th.optim.AdamW}
    
    n_envs = int(cfg.n_envs)
    total_steps = int(cfg.total_steps)
    is_wandb = bool(cfg.is_wandb)
    device = cfg.device
    render = bool(cfg.render)
    
    env_id = cfg.env.env_id
    r_gamma = float(cfg.env.r_gamma)
    
    gen_lr = float(cfg.gen.lr)
    ent_coef = float(cfg.gen.ent_coef)
    target_kl = float(cfg.gen.target_kl)
    batch_size = int(cfg.gen.batch_size)
    n_epochs = int(cfg.gen.n_epochs)
    n_steps = int(cfg.gen.n_steps)

    rew_opt = cfg.disc.reward_net_opt
    primary_opt = cfg.disc.primary_net_opt
    constraint_opt = cfg.disc.constraint_net_opt

    
    disc_lr = float(cfg.disc.lr)
    demo_batch_size = int(cfg.disc.demo_batch_size)
    gen_replay_buffer_capacity = int(cfg.disc.gen_replay_buffer_capacity)
    n_disc_updates_per_round = int(cfg.disc.n_disc_updates_per_round)
    hid_size = int(cfg.disc.hid_size)
    normalize = cfg.disc.normalize
    rollouts = load_rollouts(os.path.join(to_absolute_path('.'), "../jjh_data/expert_models/","cheetah","final.pkl"))
    
    NORMALIZE_RUNNING_POLICY_KWARGS = {
        "features_extractor_class": base.NormalizeFeaturesExtractor,
        "features_extractor_kwargs": {
            "normalize_class": imitation.util.networks.RunningNorm,
        },
        'optimizer_class': th.optim.Adam,
        'net_arch': [256,256],
    }


    tensorboard_log = os.path.join(to_absolute_path('logs'), f"{cfg.gen.model}_{cfg.env.env_id}")

    log_format_strs = ["stdout"]
    if is_wandb:
        log_format_strs.append("wandb")
        
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
    name = 'ird' + comment
    wandb.init(project='test_bench', sync_tensorboard=True,entity='jesus_accept', dir=log_dir, config=cfg, name=name)
    # if "wandb" in log_format_strs:
    #     wb.wandb_init(log_dir=log_dir)
    custom_logger = imit_logger.configure(
        folder=os.path.join(log_dir, "log"),
        format_strs=log_format_strs,
    )
    #venv = DummyVecEnv([lambda: gym.make("Gripper-v0")] * 4)
    venv = SubprocVecEnv( [make_env(env_id, i) for i in range(n_envs)])
    learner = SAC(
        env=venv,
        policy=MlpPolicy,
        batch_size=16384,
        gradient_steps=10,
        buffer_size=int(1e5),
        # ent_coef=ent_coef,
        # learning_rate=gen_lr,
        # target_kl=target_kl,
        # n_epochs=n_epochs,
        # n_steps=n_steps,
        # # policy_kwargs=NORMALIZE_RUNNING_POLICY_KWARGS,#{'optimizer_class':th.optim.Adam, },#'net_arch':[64,64]},
        # policy_kwargs={'optimizer_class':th.optim.Adam, 'net_arch':[256,256]},
        # tensorboard_log='./logs/',
        device=device,
    )
    reward_net = BasicRewardNet(
        venv.observation_space, venv.action_space, normalize_input_layer=normalize_layer[normalize],#RunningNorm,
        hid_sizes=[hid_size, hid_size],
    )
    constraint_net = BasicRewardNet(
        venv.observation_space, venv.action_space, normalize_input_layer=normalize_layer[normalize],#RunningNorm,
        hid_sizes=[hid_size, hid_size],
    )
    primary_net = PredefinedRewardNet(
            venv.observation_space, venv.action_space, reward_fn=reward_fn, combined_size=combined_size, use_action=True, normalize_input_layer=normalize_layer[normalize], #RunningNorm, #RunningNorm,
        hid_sizes=[hid_size, hid_size],
    )
    reward_net = NormalizedRewardNet(reward_net, normalize_output_layer=RunningNorm)
    constraint_net = NormalizedRewardNet(constraint_net, normalize_output_layer=RunningNorm)
    primary_net = NormalizedRewardNet(primary_net, normalize_output_layer=RunningNorm)
    gail_trainer = AIRL(
        demonstrations=rollouts,
        demo_batch_size=demo_batch_size,
        gen_replay_buffer_capacity=gen_replay_buffer_capacity,
        n_disc_updates_per_round=n_disc_updates_per_round,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
        disc_opt_kwargs={"lr":disc_lr},
        log_dir=log_dir,
        disc_opt_cls=opt_cls[rew_opt],
        gen_train_timesteps=int(1e5),
        # primary_disc_opt_cls=opt_cls[primary_opt],
        # primary_disc_opt_kwargs={"lr":disc_lr},
        # const_disc_opt_cls=opt_cls[constraint_opt],
        # const_disc_opt_kwargs={"lr":disc_lr},
        custom_logger=custom_logger
    )
    
    eval_env = DummyVecEnv([lambda: gym.make(env_id)] * 1)
    if render:
        eval_env.render(mode='human')
    checkpoint_interval=10
    safe_mean_interval=3
    def cb(round_num):
        if checkpoint_interval > 0 and round_num % safe_mean_interval == 0:
            ivc = safe_mean([ep_info["c"] for ep_info in gail_trainer.gen_algo.ep_info_buffer])
            gail_trainer.logger.record('mean/gen/rollout/ep_const_mean', ivc)
    gail_trainer.train(int(total_steps), callback=cb)  
    

if __name__ == '__main__':
    main()