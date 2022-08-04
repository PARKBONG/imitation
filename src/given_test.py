from imitation.algorithms.adversarial.gail import GAIL, RewardNetFromDiscriminatorLogit
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from stable_baselines3.sac import MlpPolicy as SACMlpPolicy
import gym
import pybulletgym
import seals
from imitation.rewards.reward_wrapper import RewardVecEnvWrapper
import torch
venv = DummyVecEnv([lambda: gym.make("GripperPegInHole2DPyBulletEnv-v1")] * 1)
basic_reward = BasicRewardNet( venv.observation_space, venv.action_space, normalize_input_layer=RunningNorm)
reward_fn = RewardNetFromDiscriminatorLogit(base=basic_reward)
reward_net = torch.load("/root/imitation/jjh_data/expert_models/peginhole_v1_imit/reward_test.pt")

# reward_fn.load_state_dict(reward_net['model_state_dict'])
venv = RewardVecEnvWrapper(venv, reward_net)

learner = SAC(
    env=venv,
    policy=SACMlpPolicy,
)

# learner_rewards_before_training, _ = evaluate_policy(
#     learner, venv, 100, return_episode_rewards=True
# )
learner.learn(20000)  # Note: set to 300000 for better results
# learner_rewards_after_training, _ = evaluate_policy(
#     learner, venv, 100, return_episode_rewards=True
# )