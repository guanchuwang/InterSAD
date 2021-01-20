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
              outlier_num=args.outlier_num)

state_dim = mdps.template_mdp.observation_space.shape[0]
action_dim = mdps.template_mdp.action_space.shape[0]

state_normalizer = RescaleNormalizer()
reward_normalizer = RescaleNormalizer()

sample_time = 10 # 100
max_trajectory_lengh = 10
quiry_time = 1
roc_auc_vs_trajectory_lengh = np.zeros((max_trajectory_lengh,))

for trajectory_len in range(1, max_trajectory_lengh+1, 1):
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
                for step in range(trajectory_len):
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

    roc_auc_vs_trajectory_lengh[trajectory_len-1] = roc_auc_buf.mean()
    print(trajectory_len, roc_auc_buf.mean())


np.savetxt('./data_reserve/' + args.exp + '_iForest_trajectory_length_vs_roc_auc.txt', roc_auc_vs_trajectory_lengh)


