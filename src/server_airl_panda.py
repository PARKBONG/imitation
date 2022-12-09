from imitation.algorithms.adversarial.ird import IRD 
from imitation.rewards.reward_nets import BasicRewardNet, BasicShapedRewardNet, NormalizedRewardNet, ScaledRewardNet, ShapedScaledRewardNet, PredefinedRewardNet, DropoutRewardNet
from imitation.util.networks import RunningNorm
from sb3_contrib import PPO2
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.ppo import MlpPolicy 
import gym
from imitation.data import types
import wandb
from imitation.util import logger as imit_logger
import os
import sys
from imitation.util import util
import torch as th
import time

from imitation.algorithms.adversarial.airl3 import AIRL3 
import hydra
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from util import visualize_reward, save, visualize_reward_gt

from torch.nn import functional as F
from imitation.rewards import reward_nets

import pybulletgym2

class SigmoidRewardNet(reward_nets.RewardNet):
    def __init__(self, base: reward_nets.RewardNet):
        """Builds LogSigmoidRewardNet to wrap `reward_net`."""
        # TODO(adam): make an explicit RewardNetWrapper class?
        super().__init__(
            observation_space=base.observation_space,
            action_space=base.action_space,
            normalize_images=base.normalize_images,
        )
        self.base = base
        self.scaler = th.nn.Parameter(th.tensor(0.0, requires_grad=True),requires_grad=True)

    def forward(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
    ) -> th.Tensor:
        logits = self.base.forward(state, action, next_state, done)
        return F.sigmoid(logits) * self.scaler * 1000
    
def load_rollouts(dir):
    with open(dir, 'rb') as f:
        rollouts = types.load(f)
    return rollouts

def make_env(env_id, rank, seed=0):
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        env = Monitor(env)
        return env
    return _init

def reward_fn(s, a, ns, d):
    return s[...,[0,4]]    
combined_size  = 2

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
    rollouts = load_rollouts(os.path.join(to_absolute_path('.'), "../jjh_data/expert_models/","panda","final.pkl"))
    
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
    wandb.init(project='test_bench', sync_tensorboard=True, dir=log_dir, config=cfg, name=name)
    # if "wandb" in log_format_strs:
    #     wb.wandb_init(log_dir=log_dir)
    custom_logger = imit_logger.configure(
        folder=os.path.join(log_dir, "log"),
        format_strs=log_format_strs,
    )
    #venv = DummyVecEnv([lambda: gym.make("Gripper-v0")] * 4)
    venv = SubprocVecEnv( [make_env(env_id, i) for i in range(n_envs)])
    learner = PPO2(
        env=venv,
        policy=MlpPolicy,
        batch_size=batch_size,
        ent_coef=ent_coef,
        learning_rate=gen_lr,
        target_kl=target_kl,
        n_epochs=n_epochs,
        n_steps=n_steps,
        policy_kwargs={'optimizer_class':th.optim.Adam},
        tensorboard_log='./logs/',
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
    
    # reward_net = SigmoidRewardNet(reward_net)
    # reward_net = NormalizedRewardNet(reward_net, normalize_output_layer=RunningNorm)
    # constraint_net = SigmoidRewardNet(constraint_net)
    # primary_net = SigmoidRewardNet(primary_net)
    # constraint_net = NormalizedRewardNet(constraint_net, normalize_output_layer=RunningNorm)
    # primary_net = NormalizedRewardNet(primary_net, normalize_output_layer=RunningNorm)
    gail_trainer = AIRL3(
        demonstrations=rollouts,
        demo_batch_size=demo_batch_size,
        gen_replay_buffer_capacity=gen_replay_buffer_capacity,
        n_disc_updates_per_round=n_disc_updates_per_round,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
        disc_opt_kwargs={"lr":disc_lr},
        log_dir=log_dir,
        primary_net=primary_net,
        constraint_net=constraint_net,
        disc_opt_cls=opt_cls[rew_opt],
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
    visualize_reward_gt(env_id='',log_dir=log_dir)
    
    def cb(round_num):
        if checkpoint_interval > 0 and round_num % checkpoint_interval == 0:
            save(gail_trainer, os.path.join(log_dir, "checkpoints", f"{round_num:05d}"))
            obs = eval_env.reset()
            for i in range(300):
                action, _states = gail_trainer.gen_algo.predict(obs, deterministic=False)
                obs, _, _, _= eval_env.step(action)
                if render:
                    eval_env.render(mode='human')
                    time.sleep(0.005)
            visualize_reward(gail_trainer.gen_algo,lambda *args: gail_trainer._running_norm( 1.0 * gail_trainer.reward_train(*args) - 1.0 * gail_trainer.primary_train(*args)), env_id,log_dir,  int(round_num), "constraint", is_wandb, )
            visualize_reward(gail_trainer.gen_algo, lambda *args: gail_trainer._running_norm(gail_trainer.primary_train(*args)), 
                             env_id,log_dir,  int(round_num), "primary", is_wandb, )
            visualize_reward(gail_trainer.gen_algo, lambda *args: gail_trainer._running_norm(gail_trainer.reward_train(*args)),
                             env_id,log_dir,  int(round_num), "total", is_wandb, )
            
    gail_trainer.train(int(total_steps), callback=cb)  
    

if __name__ == '__main__':
    main()