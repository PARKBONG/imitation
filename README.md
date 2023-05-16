## Installation:

### Installing PyPI release

```
pip install imitation
```

### Install latest commit

```
git clone http://github.com/HumanCompatibleAI/imitation
cd imitation
pip install -e .
```
## Quickstart:

### Generating a new rollouts

```bash
python3 -m imitation.scripts.eval_policy with [env] \
policy_type=[policy_type] \
policy_path=[model_dir] \
rollout_save_path=[rollout_dir] \
render

python3 -m imitation.scripts.eval_policy with half_cheetah \
policy_type=ppo_lag \
policy_path=jjh_data/expert_models/cheetah/ \
rollout_save_path=jjh_data/expert_models/cheetah/final.pkl \
render

python3 -m imitation.scripts.eval_policy with twodmaze \
policy_type=ppo_lag \
policy_path=jjh_data/expert_models/twodmaze-ppo/ \
rollout_save_path=jjh_data/expert_models/twodmaze-ppo/final.pkl \
render
python3 -m imitation.scripts.eval_policy with twodmaze \
policy_type=sac_lag \
policy_path=jjh_data/expert_models/twodmaze/ \
rollout_save_path=jjh_data/expert_models/twodmaze/final.pkl \
render

python3 -m imitation.scripts.eval_policy with serving_oneway \
policy_type=sac_lag \
policy_path=jjh_data/expert_models/serving-oneway/ \
rollout_save_path=jjh_data/expert_models/serving-oneway/final.pkl \
render
python3 -m imitation.scripts.eval_policy with serving \
policy_type=sac_lag \
policy_path=jjh_data/expert_models/serving-nov/ \
rollout_save_path=jjh_data/expert_models/serving-nov/final.pkl \
render


python3 -m imitation.scripts.eval_policy with cartpole \
policy_type=ppo \
policy_path=jjh_data/expert_models/cartpole/ \
rollout_save_path=jjh_data/expert_models/cartpole/final.pkl \
render


python3 -m imitation.scripts.eval_policy with serving \
policy_type=sac_lag \
policy_path=jjh_data/expert_models/serving-fixture/ \
rollout_save_path=jjh_data/expert_models/serving-fixture/final.pkl \
render

python3 -m imitation.scripts.eval_policy with serving \
policy_type=sac \
policy_path=jjh_data/expert_models/serving/ \
rollout_save_path=jjh_data/expert_models/serving/final.pkl \
render
python3 -m imitation.scripts.eval_policy with serving \
policy_type=ppo \
policy_path=jjh_data/expert_models/serving_imit/ \
rollout_save_path=jjh_data/expert_models/serving_imit/final.pkl \
render
python3 -m imitation.scripts.eval_policy with serving \
policy_type=sac \
policy_path=jjh_data/expert_models/serving_mj/ \
rollout_save_path=jjh_data/expert_models/serving/final.pkl \
render
```

```bash
# Train PPO agent on pendulum and collect expert demonstrations. Tensorboard logs saved in quickstart/rl/
python -m imitation.scripts.train_rl with pendulum common.fast train.fast rl.fast fast common.log_dir=quickstart/rl/

# Train GAIL from demonstrations. Tensorboard logs saved in output/ (default log directory).
python -m imitation.scripts.train_adversarial gail with pendulum common.fast demonstrations.fast train.fast rl.fast fast demonstrations.rollout_path=quickstart/rl/rollouts/final.pkl

# Train AIRL from demonstrations. Tensorboard logs saved in output/ (default log directory).
python -m imitation.scripts.train_adversarial airl with pendulum common.fast demonstrations.fast train.fast rl.fast fast demonstrations.rollout_path=quickstart/rl/rollouts/final.pkl
```
Tips:
  * Remove the "fast" options from the commands above to allow training run to completion.
  * `python -m imitation.scripts.train_rl print_config` will list Sacred script options. These configuration options are documented in each script's docstrings.

For more information on how to configure Sacred CLI options, see the [Sacred docs](https://sacred.readthedocs.io/en/stable/).


## Python Interface Quickstart:

See [examples/quickstart.py](examples/quickstart.py) for an example script that loads CartPole-v1 demonstrations and trains BC, GAIL, and AIRL models on that data.


### Density reward baseline

We also implement a density-based reward baseline. You can find an [example notebook here](examples/density_baseline_demo.ipynb).

# Citations (BibTeX)
```
@misc{wang2020imitation,
  author = {Wang, Steven and Toyer, Sam and Gleave, Adam and Emmons, Scott},
  title = {The {\tt imitation} Library for Imitation Learning and Inverse Reinforcement Learning},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/HumanCompatibleAI/imitation}},
}
```

# Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md).
