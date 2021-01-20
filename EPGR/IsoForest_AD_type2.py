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

from deep_rl import *

from LSTM_PAD_type2 import Encoder, Actor, Critic, Virtual_Userset, Virtual_User
from hyperparameter_test import *

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

args = copy.deepcopy(args_virtual_taobao_type2)

anomaly_prob = [0.1, -0.1]
with open('./user_characteristic_buf_10000_seed0.pkl', 'rb') as r:
    user_characteristic_buf = pickle.load(r)

userset = Virtual_Userset(template_mdp=args.template_mdp,
                            user_num=10000, # args.user_num,
                            user_characteristic_buf=user_characteristic_buf,
                            anomaly_num=100,
                            anomaly_prob=anomaly_prob,)

state_dim = userset.template_mdp.observation_space.shape[0]
action_dim = userset.template_mdp.action_space.shape[0]

args.trajectory_len = 10

quiry_time = 1
repeat_time = 5

roc_auc_buf = torch.zeros((repeat_time, ))

# action_reward_time_series = torch.zeros((userset.user_num, args.trajectory_len, action_dim+1)) # state_dim +
# mdp_anomaly_label = np.zeros((repeat_time, userset.user_num)).astype(dtype=np.int)
# outlier_score_buf = np.zeros((repeat_time, userset.user_num))

for repeat in range(repeat_time):

    reward_time_series = np.zeros((userset.user_num, args.trajectory_len))
    action_time_series = np.random.uniform(-1., 1., (args.trajectory_len, action_dim))

    for user_index in range(userset.user_num):

        user = userset.user_select(user_index)
        state = user.reset()
        # print(state)
        for step in range(args.trajectory_len):
            action_array = action_time_series[step]
            nxt_state, reward, done, user_anomaly = user.step(action_array)
    #
            reward_time_series[user_index, step] = reward
            state = nxt_state

        # mdp_anomaly_label[sample_index, quiry_index, user_index] = 1 if np.array(user_anomaly, dtype=np.int).sum() > 0 else 0

    # reward_predict_time_series, _, _ = critic(trajectory_time_series[sample_index, quiry_index].unsqueeze(dim=0))
    # reward_predict_time_series[reward_predict_time_series < 0.] = 0.
    # reward_predict_time_series[reward_predict_time_series > 0.] = torch.floor(reward_predict_time_series[reward_predict_time_series > 0.] + 1)
    # feature = (reward_time_series - reward_predict_time_series.squeeze(dim=2).detach().numpy())**2

    feature = reward_time_series
    # print(reward_time_series[sample_index, quiry_index])
    # print(reward_predict_time_series.squeeze(dim=2))
    # print(reward_time_series)

    # clf = IsolationForest()
    # clf = OneClassSVM()
    clf = LocalOutlierFactor(algorithm='auto', contamination=0.01)

    clf.fit(feature)
    # outlier_score = -clf.decision_function(feature)
    outlier_score = -clf._decision_function(feature)

    mdp_label = userset.user_label # mdp_anomaly_label[sample_index].max(axis=0)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(mdp_label, outlier_score)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    print(roc_auc)
    roc_auc_buf[repeat] = roc_auc

roc_auc_ave = roc_auc_buf.mean()
roc_auc_std = roc_auc_buf.std(unbiased=False)
print(float(roc_auc_ave))
print(float(roc_auc_std))







