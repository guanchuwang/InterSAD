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

from InterSADR_train import Encoder, Actor, Critic, Virtual_Userset, Virtual_User
from hyperparameter_test import *

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

# load_dir = 'virtual_taobao-type2/virtual_taobao-type2-20201025-224609'
load_dir = 'virtual_taobao-type2/virtual_taobao-type2-20201028-202226'

args = copy.deepcopy(args_virtual_taobao_type2)

anomaly_prob = [0.1, -0.1]

# export_dir = 'virtual_taobao-type2/virtual_taobao-type2-20201028-202226/' + str(anomaly_prob[0]) + '-seed0

with open('./user_characteristic_buf_1000_seed0.pkl', 'rb') as r:
    user_characteristic_buf = pickle.load(r)

max_learn_step = 600
args.trajectory_len = 10

# learn_step_buf = np.concatenate((np.arange(0,1000,100), np.arange(1000,max_learn_step,500)), axis=0)
learn_step_buf = np.arange(0,max_learn_step,100)
learn_step_num = len(learn_step_buf)

roc_auc_buf = torch.zeros((learn_step_num,))
# var_buf = np.zeros((learn_step_num, userset.user_num))
var_vs_roc_buf = []

repeat_time = 100
for index in range(repeat_time):

    userset = Virtual_Userset(template_mdp=args.template_mdp,
                              user_num=100,  # args.user_num,
                              user_characteristic_buf=user_characteristic_buf,
                              anomaly_num=10,
                              anomaly_prob=anomaly_prob, )

    state_dim = userset.template_mdp.observation_space.shape[0]
    action_dim = userset.template_mdp.action_space.shape[0]

    # action_reward_time_series = torch.zeros((userset.user_num, args.trajectory_len, action_dim+1)) # state_dim +
    reward_time_series = np.zeros((learn_step_num, userset.user_num, args.trajectory_len))
    reward_predict_time_series = torch.zeros((learn_step_num, userset.user_num, args.trajectory_len))
    trajectory_time_series = torch.zeros((learn_step_num, userset.user_num, args.trajectory_len, state_dim + action_dim))
    action_time_series = torch.zeros((learn_step_num, userset.user_num, args.trajectory_len, action_dim))
    var_vs_roc = []

    for learn_step_index in range(learn_step_num):

        learn_step = learn_step_buf[learn_step_index]
        actor = torch.load(load_dir + '/actor/actor_' + str(learn_step) + '.pkl')
        critic = torch.load(load_dir + '/critic/critic_' + str(learn_step) + '.pkl')

        for user_index in range(userset.user_num):

            user = userset.user_select(user_index)
            state = user.reset()
            # print(state)
            for step in range(args.trajectory_len):
                state_tensor = torch.tensor(state, dtype=torch.float)
                action_tensor = actor(state_tensor).detach()
                action_array = action_tensor.detach().numpy()
                nxt_state, reward, done, user_anomaly = user.step(action_array)
                reward_tensor = torch.tensor(reward, dtype=torch.float).unsqueeze(dim=0)
        #
                reward_time_series[learn_step_index, user_index, step] = reward
                action_time_series[learn_step_index, user_index, step] = action_tensor
                trajectory_time_series[learn_step_index, user_index, step] = torch.cat((state_tensor, action_tensor), dim=0)

                state = nxt_state

        feature = reward_time_series[learn_step_index]
        var_feature = feature.sum(axis=1).var(axis=0)
        # print(reward_time_series)
        # print(reward_predict_time_series.squeeze(dim=2))
        # print(reward_time_series)

        # clf = IsolationForest()
        clf = OneClassSVM()
        clf.fit(feature)
        outlier_score = -clf.decision_function(feature)

        mdp_label = userset.user_label # mdp_anomaly_label[sample_index].max(axis=0)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(mdp_label, outlier_score)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        #
        roc_auc_buf[learn_step_index] = roc_auc
        # var_buf[learn_step_index] = var_feature - var_feature.mean()
        # var_buf[index, learn_step_index] = var_feature.var(axis=0).sum()
        var_vs_roc.append([roc_auc, var_feature, learn_step])

        print(learn_step, roc_auc, var_feature)

        # with open(os.path.join(load_dir, str(anomaly_prob[0]), 'reward_time_series_' + str(learn_step)) + '.pkl', 'wb') as output:
        #     pickle.dump(feature, output)

    var_vs_roc_np = np.array(var_vs_roc)
    var_vs_roc_np[:, 1] = var_vs_roc_np[:, 1]/var_vs_roc_np[:, 1].sum()
    var_vs_roc_buf.append(var_vs_roc_np)

var_vs_roc_buf_array = np.array(var_vs_roc_buf)
print(var_vs_roc_buf_array.shape)
np.save(os.path.join(load_dir, "roc_var.npy"), var_vs_roc_buf_array)

var_vs_roc_buf_np = np.concatenate(var_vs_roc_buf, axis=0).T
var_vs_roc_mean = np.array(var_vs_roc_buf).mean(axis=0).T
re_index = var_vs_roc_mean[0].argsort()
var_vs_roc_mean = var_vs_roc_mean[:, re_index]

print(var_vs_roc_buf_np.shape)
time = np.array(learn_step_buf)
# plt.scatter(var_vs_roc_buf_np[0], var_vs_roc_buf_np[1])
plt.plot(var_vs_roc_mean[0], var_vs_roc_mean[1])

for learn_step in learn_step_buf:
    mask = var_vs_roc_buf_np[2] == learn_step
    plt.scatter(var_vs_roc_buf_np[0, mask], var_vs_roc_buf_np[1, mask])

# plt.boxplot(list(var_buf))
# plt.xlabel('step')
# plt.ylabel('ROC/AUC')
plt.xlim([0.3, 1.01])
# # plt.legend(loc='lower right')
# plt.savefig(load_dir + '/' + args.exp + '-type' + str(args.type) + '.png')
plt.show()
