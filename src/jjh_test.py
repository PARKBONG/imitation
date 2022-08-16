# %% 
# from stable_baselines3 import PPO, SAC
from sb3_contrib import PPO
from stable_baselines3.ppo import MlpPolicy as PPOMlpPolicy
from stable_baselines3.sac import MlpPolicy as SACMlpPolicy
import gym
import pybulletgym
from stable_baselines3.common.evaluation import evaluate_policy
import seals

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import matplotlib.pyplot as plt
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
import gym
import seals

env = gym.make("seals/CartPole-v0")
expert = PPO(
    policy=MlpPolicy,
    env=env,
    seed=0,
    batch_size=64,
    ent_coef=0.0,
    learning_rate=0.0003,
    n_epochs=10,
    n_steps=64,
    device='cpu'
)
expert.learn(100000)  # Note: set to 100000 to train a proficient expert
#%%
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3.common.vec_env import DummyVecEnv

rollouts = rollout.rollout(
    expert,
    DummyVecEnv([lambda: RolloutInfoWrapper(gym.make("seals/CartPole-v0"))] * 5),
    rollout.make_sample_until(min_timesteps=None, min_episodes=60),
)

#%%
import pickle
with open('filename.pkl', 'wb') as f:
	pickle.dump(rollouts, f, protocol=pickle.HIGHEST_PROTOCOL)

# %%

learner_rewards_before_training, _ = evaluate_policy(
    expert, env, 10, return_episode_rewards=True
)
print(np.mean(learner_rewards_before_training))

#%%
import pickle
with open('filename.pkl', 'rb') as f:
	rollouts = pickle.load(f)
#%%
from imitation.algorithms.adversarial.gail import GAIL
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from stable_baselines3.ppo import MlpPolicy 
import gym
import seals


venv = DummyVecEnv([lambda: gym.make("seals/CartPole-v0")] * 8)
learner = PPO(
    env=venv,
    policy=MlpPolicy,
    batch_size=64,
    ent_coef=0.0,
    learning_rate=0.0003,
    n_epochs=10,
    device='cpu'
)
reward_net = BasicRewardNet(
    venv.observation_space, venv.action_space, normalize_input_layer=RunningNorm
)
gail_trainer = GAIL(
    demonstrations=rollouts,
    demo_batch_size=1024,
    gen_replay_buffer_capacity=2048,
    n_disc_updates_per_round=4,
    venv=venv,
    gen_algo=learner,
    reward_net=reward_net,
)

learner_rewards_before_training, _ = evaluate_policy(
    learner, venv, 100, return_episode_rewards=True
)
gail_trainer.train(20000)  # Note: set to 300000 for better results
learner_rewards_after_training, _ = evaluate_policy(
    learner, venv, 100, return_episode_rewards=True
)
# %%
