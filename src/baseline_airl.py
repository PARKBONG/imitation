from imitation.algorithms.adversarial.ird import IRD 
from imitation.rewards.reward_nets import BasicRewardNet, BasicShapedRewardNet, NormalizedRewardNet, ScaledRewardNet, ShapedScaledRewardNet, PredefinedRewardNet, DropoutRewardNet
from imitation.util.networks import RunningNorm
from sb3_contrib import PPO2
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.ppo import MlpPolicy 
from stable_baselines3 import PPO
from sb3_contrib.common.monitor import ConstMonitor
from sb3_contrib import PPO2, SACLagrangian
import gym
from imitation.data import types
import wandb
from imitation.util import logger as imit_logger
import os
import sys
from imitation.util import util
import torch as th
import time
from util import visualize_reward, save, visualize_reward_gt, plot_reward, get_cmap
import numpy as np
import matplotlib.pyplot as plt
from imitation.algorithms.adversarial.airl3 import AIRL3
from imitation.algorithms.adversarial.airl import AIRL
import hydra
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from util import visualize_reward, save, visualize_reward_gt

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
    return s[...,[0,1,2,3,4]]    
combined_size  = 5
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
    rollouts = load_rollouts(os.path.join(to_absolute_path('.'), "../jjh_data/expert_models/","cheetah_vel","final.pkl"))
    
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
    from imitation.policies import base
    import imitation
    NORMALIZE_RUNNING_POLICY_KWARGS = {
        "features_extractor_class": base.NormalizeFeaturesExtractor,
        "features_extractor_kwargs": {
            "normalize_class": imitation.util.networks.RunningNorm,
        },
        'optimizer_class': th.optim.Adam,
        # 'net_arch': [128, 128]
    }

 
    if cfg.comment == "None":
        comment = ""
    else:
        comment = f"_{str(cfg.comment)}"
    name = 'ird' + comment
    wandb.init(project='test_bench', sync_tensorboard=True, dir=log_dir, config=cfg, name=name)
    # if "wandb" in log_format_strs:
    #     wb.wandb_init(log_dir=log_dir)
    custom_logger = imit_logger.configure(
        folder=os.path.join(log_dir, "log"),
        format_strs=log_format_strs,
    )
    #venv = DummyVecEnv([lambda: gym.make("Gripper-v0")] * 4)
    venv = SubprocVecEnv( [make_env(env_id, i) for i in range(n_envs)])
    learner = PPO(
        env=venv,
        policy=MlpPolicy,
        batch_size=batch_size,
        ent_coef=ent_coef,
        learning_rate=gen_lr,
        target_kl=target_kl,
        n_epochs=n_epochs,
        n_steps=n_steps,
        policy_kwargs=NORMALIZE_RUNNING_POLICY_KWARGS,#'net_arch':[64,64]},
        tensorboard_log='./logs/',
        device=device,
    )
    reward_net = BasicRewardNet(
        venv.observation_space, venv.action_space, normalize_input_layer=normalize_layer[normalize],#RunningNorm,
        hid_sizes=[hid_size, hid_size],
    )
    reward_net = NormalizedRewardNet(reward_net, normalize_output_layer=RunningNorm)
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
        custom_logger=custom_logger
    )
   
    eval_env = DummyVecEnv([lambda: gym.make(env_id)] * 1)
    if render:
        eval_env.render(mode='human')
    checkpoint_interval=10
    safe_mean_interval=3
    # visualize_reward_gt(env_id='',log_dir=log_dir)
  

    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    model = SACLagrangian.load(os.path.join(to_absolute_path('.'), "../jjh_data/expert_models/","cheetah_vel","model.zip"))
    old_obs = eval_env.reset()
    obs_batch = []
    obs_action = []
    next_obs_batch = []
    from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, VectorizedActionNoise
    r_batch = []
    c_batch = []

    for _ in range(4):
        a = time.time()
        total_rewards = 0
        total_violations = 0
        for i in range(1000):
            action, _states = model.predict(old_obs, deterministic=False)
            obs, rewards, dones, infos= eval_env.step(action)
            if dones.any():
                continue
            obs_batch.append(old_obs)
            next_obs_batch.append(obs)
            obs_action.append(action)
            for rew in rewards:
                total_rewards += rew
            for info in infos:
                total_violations += info['constraint']
            c_batch.append(infos[0]['constraint'])
            r_batch.append(rewards)
            old_obs = obs
        print(total_rewards, total_violations)
        
        print(time.time() - a) 
    n_actions = eval_env.action_space.shape[-1]
    mean = np.zeros(n_actions)
    sigma=0.1 * np.ones(n_actions)
    action_noise = NormalActionNoise(mean=mean, sigma=sigma)
    vec_action_noise = VectorizedActionNoise(base_noise = action_noise, n_envs=1)

    for _ in range(4):
        a = time.time()
        total_rewards = 0
        total_violations = 0
        for i in range(1000):
            action, _states = model.predict(old_obs, deterministic=False)
            action = np.clip(action + vec_action_noise(), -1, 1)
            obs, rewards, dones, infos= eval_env.step(action)
            if dones.any():
                continue
            obs_batch.append(old_obs)
            next_obs_batch.append(obs)
            obs_action.append(action)
            for rew in rewards:
                total_rewards += rew
            for info in infos:
                total_violations += info['constraint']
            c_batch.append(infos[0]['constraint'])
            r_batch.append(rewards)
            old_obs = obs
        print(total_rewards, total_violations)
        
        print(time.time() - a) 
    
    model = SACLagrangian.load(os.path.join(to_absolute_path('.'), "../jjh_data/expert_models/","cheetah","model.zip"))
    for _ in range(4):
        a = time.time()
        total_rewards = 0
        total_violations = 0
        for i in range(1000):
            action, _states = model.predict(old_obs, deterministic=False)
            obs, rewards, dones, infos= eval_env.step(action)
            if dones.any():
                continue
            obs_batch.append(old_obs)
            next_obs_batch.append(obs)
            obs_action.append(action)
            for rew in rewards:
                total_rewards += rew
            for info in infos:
                total_violations += info['constraint']
            c_batch.append(infos[0]['constraint'])
            r_batch.append(rewards)
            old_obs = obs
        print(total_rewards, total_violations)
        
        print(time.time() - a) 
    
    plot_grid = (eval_env.observation_space.shape[-1]//5 + 1, 5)
    savedir = os.path.join(log_dir,"maps")
    r_batch = np.squeeze(np.array(r_batch),axis=-1)
    c_batch = np.array(c_batch)
    obs_batch = np.array(obs_batch)
    obs_batch = np.reshape(obs_batch, (-1, eval_env.observation_space.shape[-1]))
    obs_action = np.reshape(obs_action, (-1, eval_env.action_space.shape[-1]))
    next_obs_batch = np.reshape(next_obs_batch, (-1, eval_env.observation_space.shape[-1]))
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    for ob in range(eval_env.observation_space.shape[-1]):
        ax = plt.subplot2grid(plot_grid, (ob//5, ob%5))
        from itertools import cycle
        cycol = cycle('bgrcmykw')
        cmap = get_cmap(len(next_obs_batch)//4000 + 1)
        for i in range(len(next_obs_batch)//4000):

            ax.scatter(next_obs_batch[i*4000: (i+1)*4000,ob], c_batch[i*4000: (i+1)*4000], c=next(cycol), marker='o', s=2, alpha=0.1)
        # labels = [item.get_text() for item in ax.get_xticklabels()]
        # ax.scatter(next_obs_batch[...,ob], c_batch, marker='o', s=2, alpha=0.5)
        # labels = [item.get_text() for item in ax.get_xticklabels()]
        # empty_string_labels = ['']*len(labels)
        # ax.set_xticklabels(empty_string_labels)  

    cycol = cycle('bgrcmykw')
    ax = plt.subplot2grid(plot_grid, (eval_env.observation_space.shape[-1]//5, eval_env.observation_space.shape[-1]%5))
    cmap = get_cmap(len(next_obs_batch)//4000 + 1)
    for i in range(len(next_obs_batch)//4000):

        ax.scatter(np.max(np.abs(next_obs_batch[i*4000: (i+1)*4000,10:]), axis=-1), c_batch[i*4000: (i+1)*4000], c=next(cycol), marker='o', s=1, alpha=0.3)
    plt.savefig(savedir + '/test.png')
    sa_pair = (obs_batch, obs_action, next_obs_batch)
    from stable_baselines3.common.utils import safe_mean
    
    def cb(round_num):

        if checkpoint_interval > 0 and round_num % safe_mean_interval == 0:
            ivc = safe_mean([ep_info["c"] for ep_info in gail_trainer.gen_algo.ep_info_buffer])
            gail_trainer.logger.record('mean/gen/rollout/ep_const_mean', ivc)
            
        if checkpoint_interval > 0 and round_num % checkpoint_interval == 0:
            save(gail_trainer, os.path.join(log_dir, "checkpoints", f"{round_num:05d}"))
            obs = eval_env.reset()
            total_violation=0
            for i in range(1000):
                action, _states = gail_trainer.gen_algo.predict(obs, deterministic=False)
                obs, _, _, infos= eval_env.step(action)
                for info in infos:
                    total_violation+=info['constraint']
                if render:
                    eval_env.render(mode='human')
                    time.sleep(0.005)
            # gail_trainer.logger.record('eval/const_mean', total_violation/(8000*4))
            # plot_reward(gail_trainer.gen_algo,lambda *args: 1.0 * gail_trainer.constraint_train.predict_processed(*args) + 1.0 * gail_trainer.primary_train.predict_processed(*args), eval_env, log_dir,  int(round_num), "total_rand", is_wandb, sa_pair)
            # plot_reward(gail_trainer.gen_algo,lambda *args: 1.0 * gail_trainer.constraint_train.predict_processed(*args), eval_env, log_dir,  int(round_num), "constraint_rand", is_wandb, sa_pair )
            # plot_reward(gail_trainer.gen_algo,lambda *args: 1.0 * gail_trainer.primary_train.predict_processed(*args), eval_env, log_dir,  int(round_num), "primary_rand", is_wandb, sa_pair)
            # # visualize_reward(gail_trainer.gen_algo,lambda *args: 1.0 * gail_trainer.reward_train(*args) - 1.0 * gail_trainer.primary_train(*args), env_id,log_dir,  int(round_num), "constraint", is_wandb, )
            # visualize_reward(gail_trainer.gen_algo, gail_trainer.primary_train, env_id,log_dir,  int(round_num), "primary", is_wandb, )
            # visualize_reward(gail_trainer.gen_algo, gail_trainer.reward_train, env_id,log_dir,  int(round_num), "total", is_wandb, )

    gail_trainer.train(int(total_steps), callback=cb)  
    

if __name__ == '__main__':
    main()