
import os
from imitation.rewards.reward_nets import RewardNetWrapper
from imitation.rewards.reward_nets import RewardNet
import torch as th
from imitation.policies import serialize
from scipy import ndimage
import numpy as np
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

    plate_ang = 0.0
    num_y = 0
    for pos in np.arange(-2.1, 2.1, 0.05):
        num_y += 1
        num_x = 0
        for ang in np.arange(-2.1, 2.1, 0.05):
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
            
            obs[1] = pos
            obs[5] = pos
            # obs[7] = np.tanh(ang)
            obs[7] = ang
            obs_batch.append(obs)
            next_obs_batch.append(obs)

    obs_batch = np.array(obs_batch)
    next_obs_batch = np.array(next_obs_batch)

    irl_reward = np.zeros([num_x, num_y])
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
    print(score.reshape([num_x, num_y]))
    savedir = os.path.join("./temp")
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    print(savedir)
    plt.savefig(savedir + '/%s.png' % (itr))
    print('Save Itr', itr)
    plt.close()