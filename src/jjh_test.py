# %% 
from stable_baselines3 import PPO, SAC
from stable_baselines3.ppo import MlpPolicy as PPOMlpPolicy
from stable_baselines3.sac import MlpPolicy as SACMlpPolicy
import gym
import pybulletgym
from stable_baselines3.common.evaluation import evaluate_policy
import seals

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import matplotlib.pyplot as plt
import numpy as np

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        #env = ConstMonitor(env)
        env = Monitor(env)
        return env
    set_random_seed(seed)
    return _init
#%%
env = gym.make("GripperPegInHole2DPyBulletEnv-v1")

#venv = SubprocVecEnv([lambda: gym.make("GripperPegInHole2DPyBulletEnv-v1")] * 8, start_method="forkserver")

venv = DummyVecEnv( [make_env("GripperPegInHole2DPyBulletEnv-v1", i) for i in range(4)] )
expert = SAC.load("/root/imitation/src/imitation/quickstart/rl/policies/000000840000/model.zip", env=env)
#%%
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper

rollouts = rollout.rollout(
    expert,
    DummyVecEnv([lambda: RolloutInfoWrapper(gym.make("GripperPegInHole2DPyBulletEnv-v1"))] * 8),
    # venv,
    rollout.make_sample_until(min_timesteps=None, min_episodes=16),
)

# %%

learner_rewards_before_training, _ = evaluate_policy(
    expert, env, 10, return_episode_rewards=True
)
print(np.mean(learner_rewards_before_training))


#%%
from imitation.algorithms.adversarial.gail import GAIL
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

import gym
import seals


venv = DummyVecEnv([lambda: gym.make("GripperPegInHole2DPyBulletEnv-v1")] * 1)
learner = PPO(
    env=venv,
    policy=PPOMlpPolicy,
    batch_size=64,
    ent_coef=0.0,
    learning_rate=0.0003,
    n_epochs=2,
)
reward_net = BasicRewardNet(
    venv.observation_space, venv.action_space, normalize_input_layer=RunningNorm
)
gail_trainer = GAIL(
    demonstrations=rollouts,
    demo_batch_size=16,
    gen_replay_buffer_capacity=16,
    n_disc_updates_per_round=4,
    venv=venv,
    gen_algo=learner,
    reward_net=reward_net,
)

# learner_rewards_before_training, _ = evaluate_policy(
#     learner, venv, 5, return_episode_rewards=True
# )
gail_trainer.train(20000)  # Note: set to 300000 for better results
learner_rewards_after_training, _ = evaluate_policy(
    learner, venv, 5, return_episode_rewards=True
)

print(np.mean(learner_rewards_after_training))
print(np.mean(learner_rewards_before_training))
# %%
