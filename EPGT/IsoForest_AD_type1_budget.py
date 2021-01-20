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

args = copy.deepcopy(args_half_cheetah_type1)
# args = copy.deepcopy(args_hopper_type1)
# args = copy.deepcopy(args_walker_type1)
# args = copy.deepcopy(args_ant_type1)

mdps = MDPset(mdp_num=100, # args.mdp_num,
              template_mdp=args.template_mdp,
              inlier_mdps=args.inlier_mdps,  # 'HalfCheetahBulletmdp-v0',
              outlier_mdps=args.outlier_mdps,  # ['HalfCheetahBrother_0_06_BulletEnv-v0'],
              # outlier_mdps=[HalfCheetah_Type2Anomaly_BulletEnv(noise_std=0.01)],  # ['HalfCheetahBrother_0_06_BulletEnv-v0'],
              outlier_rate=args.outlier_rate)

state_dim = mdps.template_mdp.observation_space.shape[0]
action_dim = mdps.template_mdp.action_space.shape[0]

state_normalizer = RescaleNormalizer()
reward_normalizer = RescaleNormalizer()

args.trajectory_len = 10

sample_time = 10 # 100
max_quiry_time = 10
roc_auc_vs_quiry_time = np.zeros((max_quiry_time,))

for quiry_time in range(1, max_quiry_time+1, 1):
    roc_auc_buf = torch.zeros((sample_time,))
    state_action_time_series = np.zeros((quiry_time, args.mdp_num, args.trajectory_len, state_dim + action_dim))

    for sample_index in range(sample_time):
        action_buf = np.random.uniform(-1., 1., (args.trajectory_len, action_dim))

        for repeat in range(quiry_time):
            for mdp_index in range(args.mdp_num):
                mdp = mdps.mdp_buf[mdp_index]
                state_noise = np.random.normal(0, args.state_noise_std, (state_dim, ))
                state = mdp.reset() + state_noise
                # print(state)
                for step in range(args.trajectory_len):
                    action = action_buf[step]
                    nxt_state, reward, done, _ = mdp.step(action)

                    state_noise = np.random.normal(0, args.state_noise_std, (state_dim,))
                    nxt_state += state_noise
                    state_action_time_series[repeat, mdp_index, step] = np.concatenate((state, action), axis=0)
                    state = nxt_state
                    # print(action)
                    # print(state_action_time_series.shape)

        mdp_time_series = state_action_time_series.mean(axis=0).reshape(args.mdp_num, -1)


        clf = IsolationForest()
        # clf = OneClassSVM()
        # clf = LocalOutlierFactor(algorithm='auto', contamination=0.1, novelty=True)

        clf.fit(mdp_time_series)
        outlier_score = -clf.decision_function(mdp_time_series)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(mdps.mdp_label, outlier_score)
        roc_auc = auc(false_positive_rate, true_positive_rate)

        roc_auc_buf[sample_index] = roc_auc
        print(sample_index, roc_auc)


    roc_auc_vs_quiry_time[quiry_time-1] = roc_auc_buf.mean()
    print(quiry_time, roc_auc_buf.mean())


np.savetxt('./data_reserve/' + args.exp + '_iForest_quiry_time_vs_roc_auc.txt', roc_auc_vs_quiry_time)


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

