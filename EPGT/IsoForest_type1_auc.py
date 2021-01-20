import gym
import numpy as np
import torch
import random
import pybullet_envs
import time
from sklearn.metrics import roc_curve, auc
from sklearn import datasets, manifold
from LSTM_PAD_type1 import MDPset

import matplotlib.pyplot as plt
import copy

from deep_rl import *

from hyperparameter_test import *

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
import os

# args = copy.deepcopy(args_half_cheetah_type1)
# args = copy.deepcopy(args_hopper_type1)
args = copy.deepcopy(args_walker_type1)
# args = copy.deepcopy(args_ant_type1)

# mdps = MDPset(mdp_num=100, # args.mdp_num,
#               template_mdp=args.template_mdp,
#               inlier_mdps=args.inlier_mdps,  # 'HalfCheetahBulletmdp-v0',
#               outlier_mdps=args.outlier_mdps,  # ['HalfCheetahBrother_0_06_BulletEnv-v0'],
#               # outlier_mdps=[HalfCheetah_Type2Anomaly_BulletEnv(noise_std=0.01)],  # ['HalfCheetahBrother_0_06_BulletEnv-v0'],
#               outlier_rate=args.outlier_rate)

mdp_num = 10000
outlier_num = 100
mdp_inds = np.array([mdp_index for mdp_index in range(mdp_num)])
outlier_inds = np.random.choice(mdp_inds, outlier_num, replace=False)
inlier_inds = np.array(list(set(mdp_inds) - set(outlier_inds)))
mdp_label = [1 if mdp_index in outlier_inds else 0 for mdp_index in mdp_inds]

print(outlier_inds)

state_dim = gym.make(args.inlier_mdps[0]).observation_space.shape[0]
action_dim = gym.make(args.inlier_mdps[0]).action_space.shape[0]

args.trajectory_len = 10

repeat_time = 5 # 1000

roc_auc_buf = torch.zeros((repeat_time, ))
# cross_distance_trajectory_space_sum = np.zeros((mdp_num, mdp_num))

for repeat in range(repeat_time):

    state_action_time_series = np.zeros((mdp_num, args.trajectory_len, state_dim + action_dim))
    action_buf = np.random.uniform(-1., 1., (args.trajectory_len, action_dim))

    for mdp_index in range(mdp_num):

        if mdp_index in outlier_inds:
            outlier_mdp_sample = np.random.choice(args.outlier_mdps)
            mdp = outlier_mdp_sample
        else:
            inlier_mdp_sample = np.random.choice(args.inlier_mdps)
            mdp = gym.make(inlier_mdp_sample)

        state_noise = np.random.normal(0, args.state_noise_std, (state_dim, ))
        state = mdp.reset() + state_noise
        # print(state)
        for step in range(args.trajectory_len):
            action = action_buf[step]
            nxt_state, reward, done, _ = mdp.step(action)

            state_noise = np.random.normal(0, args.state_noise_std, (state_dim,))
            nxt_state += state_noise
            state_action_time_series[mdp_index, step] = np.concatenate((state, action), axis=0)
            state = nxt_state
            # print(action)
            # print(state_action_time_series.shape)

    # print(state_action_time_series.shape)
        # print(critic.hidden(state_action_time_series).shape)

    mdp_time_series = state_action_time_series.reshape(mdp_num, -1)

    # plt.plot(mdp_time_series[mdps.inlier_index].T, color='blue')
    # plt.plot(mdp_time_series[mdps.outlier_index].T, color='red')
    # plt.show()

    # clf = IsolationForest()
    # clf = OneClassSVM()
    clf = LocalOutlierFactor(algorithm='auto', contamination=0.01) # , novelty=True)

    clf.fit(mdp_time_series)
    outlier_score = -clf._decision_function(mdp_time_series) # decision_function(mdp_time_series)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(mdp_label, outlier_score)
    roc_auc = auc(false_positive_rate, true_positive_rate)

    roc_auc_buf[repeat] = roc_auc
    print(repeat, roc_auc)

roc_auc_ave = roc_auc_buf.mean()
roc_auc_std = roc_auc_buf.std(unbiased=False)
print(roc_auc_ave)
print(roc_auc_std)


# cross_distance_trajectory_space = np.zeros((args.mdp_num, args.mdp_num))
    # for idx1 in range(args.mdp_num):
    #     for idx2 in range(idx1):
    #         cross_distance_trajectory_space[idx1, idx2] = ((state_action_time_series[idx1] - state_action_time_series[idx2])**2).sum()
    #
    # for idx1 in range(args.mdp_num):
    #     for idx2 in range(idx1+1, args.mdp_num):
    #         cross_distance_trajectory_space[idx1, idx2] = cross_distance_trajectory_space[idx2, idx1]
    #
    # cross_distance_trajectory_space_sum += cross_distance_trajectory_space
    # print(cross_distance)

# cross_distance_trajectory_space_sum /= args.mdp_num
#
# inlier_index = mdps.inlier_index
# outlier_index = mdps.outlier_index

# np.save(os.path.join('./data_reserve', args.exp + '_trajectory_cross_distance.npy'),
#         [cross_distance_trajectory_space_sum, mdps.inlier_index, mdps.outlier_index])

# cross_distance_trajectory_space_sum, inlier_index, outlier_index = np.load('./data_reserve/walker_trajectory_cross_distance_1.npy', allow_pickle=True)

# plt.figure()
# plt.imshow(cross_distance_trajectory_space_sum, cmap='gray')
# cbar = plt.colorbar()
# cbar.set_label('Distance', fontsize=18)
# cbar.ax.tick_params(labelsize=15)
# plt.scatter(inlier_index, inlier_index, color='blue')
# plt.scatter(outlier_index, outlier_index, color='red')
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.xlabel('MDP index', fontsize=18)
# plt.ylabel('MDP index', fontsize=18)
# plt.savefig(os.path.join('figure', 'RA+iForest-' + args.exp + '-type' + str(args.type) + '-trajectory-heat.png'))
# plt.show()


# plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.4f' % roc_auc)
# plt.show()
# print(outlier_score)
# print(roc_auc)

