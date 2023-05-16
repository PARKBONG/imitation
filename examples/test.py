
import numpy as np
import gym
from stable_baselines3 import  SAC
from sb3_contrib import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
# from stable_baselines3.sac import MlpPolicy2
from sb3_contrib.sac_custom import MlpPolicy

from imitation.algorithms.adversarial.airl import AIRL
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env

from hydra.utils import get_original_cwd, to_absolute_path
import os


from imitation.data import types
rng = np.random.default_rng(0)

def load_rollouts(dir):
    with open(dir, 'rb') as f:
        rollouts = types.load(f)
    return rollouts

env = gym.make("Serving-v3")
expert = SAC(policy=MlpPolicy, env=env)
expert.learn(1000)
rollouts = load_rollouts(os.path.join(to_absolute_path('.'), "../jjh_data/expert_models/","serving-oneway","final.pkl"))
    
venv = make_vec_env("Serving-v3", n_envs=8)
learner = SAC(env=venv, policy=MlpPolicy)
reward_net = BasicShapedRewardNet(
    venv.observation_space,
    venv.action_space,
    normalize_input_layer=RunningNorm,
)
airl_trainer = AIRL(
    demonstrations=rollouts,
    demo_batch_size=1024,
    gen_replay_buffer_capacity=2048,
    n_disc_updates_per_round=4,
    venv=venv,
    gen_algo=learner,
    reward_net=reward_net,
    allow_variable_horizon=True,
)
airl_trainer.train(20000)
rewards, _ = evaluate_policy(learner, venv, 100, return_episode_rewards=True)
print("Rewards:", rewards)