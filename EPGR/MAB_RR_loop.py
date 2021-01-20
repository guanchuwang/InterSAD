import gym
import numpy as np
import torch
import random
import pybullet_envs
import time
from sklearn.metrics import roc_curve, auc
from sklearn import datasets, manifold

import matplotlib.pyplot as plt
import copy

import log_utils, os, pickle

from deep_rl import *

from LSTM_PAD_type2 import Virtual_Userset, Virtual_User
from hyperparameter_test import *

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

anomaly_prob = [0.1, -0.1]

with open('./user_characteristic_buf_10000_seed0.pkl', 'rb') as r:
# with open('./user_characteristic_buf_100.pkl', 'rb') as r:
    user_characteristic_buf = pickle.load(r)

userset = Virtual_Userset(template_mdp='VirtualTB-v0',
                            user_num=10000, # args.user_num,
                            user_characteristic_buf=user_characteristic_buf,
                            anomaly_num=100,
                            anomaly_prob=anomaly_prob,)

state_dim = userset.template_mdp.observation_space.shape[0]
action_dim = userset.template_mdp.action_space.shape[0]

trajectory_len = 10
round_time = 1000000
repeat_time = 5

roc_auc_buf = torch.zeros((repeat_time,))

for repeat in range(repeat_time):

    reward_time_series = np.zeros((round_time, userset.user_num))
    mean_reward = np.zeros((userset.user_num,))
    user_step_buf = np.zeros((userset.user_num,), dtype=np.int)

    action_time_series = np.random.uniform(-1., 1., (round_time, action_dim))

    k_threshold = 1.
    user_index = 0
    reward_bound = 20 - 0
    delta = 0.9999
    active_flag = [True for user_index in range(userset.user_num)]

    for round_index in range(round_time):

        user = userset.user_select(user_index)
        step = user_step_buf[user_index]

        if step % trajectory_len == 0:
            user.reset()

        action_array = action_time_series[round_index]
        nxt_state, reward, done, _ = user.step(action_array)
        reward_tensor = torch.tensor(reward, dtype=torch.float).unsqueeze(dim=0)

        reward_time_series[step, user_index] = reward
        # print(reward)

        if all(user_step_buf > 1):

            mean_reward[user_index] = reward_time_series[0:step, user_index].mean()
            mean_mean_reward = mean_reward.mean()
            sigma_mean_reward = np.sqrt((mean_reward - mean_mean_reward).var())
            theta = mean_mean_reward + k_threshold * sigma_mean_reward

            beta = reward_bound * np.sqrt(k_threshold/(2*user_step_buf[user_index])*(-np.log(delta)))

            harmonic_mean_step = userset.user_num*1./(1./user_step_buf).sum()
            beta_theta = reward_bound * np.sqrt(k_threshold/(2*harmonic_mean_step)*(-np.log(delta)))

            # print(mean_reward[user_index], theta, beta, beta_theta)
            if not (theta < mean_reward[user_index] < theta + beta + beta_theta) and \
                not (theta - beta + beta_theta < mean_reward[user_index] < theta):

                active_flag[user_index] = False
            else:

                active_flag[user_index] = True

        user_step_buf[user_index] = step + 1
        user_index = (user_index + 1) % userset.user_num

        # if round_index % 1000 == 0:
        #     print('Active number: ', np.array(active_flag).sum())

        if not any(active_flag):
            break


    outlier_score = mean_reward-theta
    mdp_label = userset.user_label
    false_positive_rate, true_positive_rate, thresholds = roc_curve(mdp_label, outlier_score)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    print(roc_auc)
    roc_auc_buf[repeat] = roc_auc

roc_auc_ave = roc_auc_buf.mean()
roc_auc_std = roc_auc_buf.std(unbiased=False)
print(float(roc_auc_ave))
print(float(roc_auc_std))











