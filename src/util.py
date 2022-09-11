
import os
from imitation.rewards.reward_nets import RewardNetWrapper
from imitation.rewards.reward_nets import RewardNet
import torch as th
from imitation.policies import serialize
from scipy import ndimage
import numpy as np
import wandb

def save(trainer, save_path):
    """Save discriminator and generator."""
    # We implement this here and not in Trainer since we do not want to actually
    # serialize the whole Trainer (including e.g. expert demonstrations).
    os.makedirs(save_path, exist_ok=True)
    if isinstance(trainer._reward_net, RewardNet):
        th.save(trainer.reward_train, os.path.join(save_path, "reward_train.pt"))
        th.save(trainer.reward_test, os.path.join(save_path, "reward_test.pt"))

    if hasattr(trainer, "primary_train"):
        saving_net_train = trainer.primary_train
        th.save(saving_net_train, os.path.join(save_path, "primary_train.pt"))
        saving_net_test = trainer.primary_test
        while isinstance(saving_net_test, RewardNetWrapper) and hasattr(saving_net_test, "base"):
            saving_net_test = saving_net_test.base
        if hasattr(saving_net_train, "mlp"):
            th.save(saving_net_test.mlp, os.path.join(save_path, "primary_test.pt"))

    if hasattr(trainer, "constraint_train") and isinstance(trainer._constraint_net, RewardNet):
        saving_net_train = trainer.constraint_train
        th.save(saving_net_train, os.path.join(save_path, "constraint_train.pt"))
        saving_net_test = trainer.constraint_test
        while isinstance(saving_net_test, RewardNetWrapper) and hasattr(saving_net_test, "base"):
            saving_net_test = saving_net_test.base
        if hasattr(saving_net_train, "mlp"):
            th.save(saving_net_train.mlp, os.path.join(save_path, "constraint_test.pt"))
    serialize.save_stable_model(
        os.path.join(save_path, "gen_policy"),
        trainer.gen_algo,
    )

def visualize_reward(model, reward_net, env_id, log_dir, round_num, tag='', use_wandb=False, goal=1.0):
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
    grid_size = 0.05
    rescale= int(1/grid_size)
    boundary_low = -2.1
    boundary_high = 2.1
    for goal in [-1, 1]:
        obs_batch = []
        obs_action = []
        next_obs_batch = []
        target = [0., goal]

        plate_ang = 0.0
        num_y = 0
        for pos in np.arange(boundary_low, boundary_high, grid_size):
            num_y += 1
            num_x = 0
            for ang in np.arange(boundary_low, boundary_high, grid_size):
                num_x += 1
                obs = np.zeros(9)
                """
                <state type="xpos" body="goal"/>    ## 0
                <state type="xpos" body="plate"/>   ## 1
                <state type="xvel" body="plate"/>   ## 2
                <state type="apos" body="plate"/>   ## 3   
                <state type="avel" body="plate"/>   ## 4
                <state type="xpos" body="pole"/>    ## 5
                <state type="xvel" body="pole"/>    ## 6
                <state type="apos" body="pole"/>    ## 7
                <state type="avel" body="pole"/>    ## 8
                """
                plate_x = pos
                
                pole_ang = ang
                mid_pole_x = plate_x - np.sin(plate_ang)*(plate_height/2)
                pole_x = mid_pole_x - (np.cos(pole_ang) - np.cos(plate_ang)) * (pole_width/2)
                
                obs[0] = goal
                obs[1] = pos
                obs[5] = pos
                # obs[7] = np.tanh(ang)
                obs[7] = ang
                obs_batch.append(obs)

                action, _ = model.predict(obs, deterministic=True)
                # next_state, reward, done, _ = env.step(action)

                obs_action.append(action)
                next_obs_batch.append(obs)

        obs_batch = np.array(obs_batch)
        next_obs_batch = np.array(next_obs_batch)
        obs_action = np.array(obs_action)

        # Get sqil reward
        with th.no_grad():
            state = th.FloatTensor(obs_batch).to(model.device)
            action = th.FloatTensor(obs_action).to(model.device)
            next_state = th.FloatTensor(next_obs_batch).to(model.device)

        done = th.zeros_like(state[...,-1:])

        with th.no_grad():
            irl_reward = reward_net(state, action, next_state, done)

            irl_reward = irl_reward.cpu().numpy()
        score = irl_reward

        score = irl_reward
        flights = score.copy().reshape([num_x, num_y])
        ax = sns.heatmap(score.reshape([num_x, num_y]), cmap="YlGnBu_r")
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

        ax.scatter((target[0]-boundary_low)*rescale, (target[1]-boundary_low)
                    * rescale, marker='*', s=100, c='r', edgecolors='k', linewidths=0.5)
        if use_wandb:
            wandb.log({f"rewards_map({goal})/{tag}": wandb.Image(plt)}, step=round_num)
        print(score.reshape([num_x, num_y]))
        savedir = os.path.join(log_dir,"maps")
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        print(savedir)
        plt.savefig(savedir + '/%s_%s.png' % (goal, tag))
        print('Save Itr', goal)
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
        for pos in np.arange(-1.5, 1.5, 0.05):
            num_y += 1
            num_x = 0
            for ang in np.arange(-1.5, 1.5, 0.05):
                num_x += 1
                obs = np.zeros(9)
                """
                    <state type="xpos" body="goal"/>    ## 0
                <state type="xpos" body="plate"/>   ## 1
                <state type="xvel" body="plate"/>   ## 2
                <state type="apos" body="plate"/>   ## 3   
                <state type="avel" body="plate"/>   ## 4
                <state type="xpos" body="pole"/>    ## 5
                <state type="xvel" body="pole"/>    ## 6
                <state type="apos" body="pole"/>    ## 7
                <state type="avel" body="pole"/>    ## 8
                """
                plate_x = pos
                
                pole_ang = ang
                mid_pole_x = plate_x - np.sin(plate_ang)*(plate_height/2)
                pole_x = mid_pole_x - (np.cos(pole_ang) - np.cos(plate_ang)) * (pole_width/2)
                
                obs[0] = 1.0
                obs[1] = pos
                obs[5] = pos
                obs[7] = ang
                
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
        ax.invert_yaxis()
        plt.axis('off')
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