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


# load_dir = 'virtual_taobao-type2/virtual_taobao-type2-20201025-224609'
load_dir = 'virtual_taobao-type2/virtual_taobao-type2-20201028-202226'

args = copy.deepcopy(args_virtual_taobao_type2)

anomaly_prob = [0.3, -0.3]
with open('./user_characteristic_buf_100000_seed7.pkl', 'rb') as r:
    user_characteristic_buf = pickle.load(r)

learn_step = 1000

args.trajectory_len = 10

actor = torch.load(load_dir + '/actor/actor_' + str(learn_step) + '.pkl')
critic = torch.load(load_dir + '/critic/critic_' + str(learn_step) + '.pkl')


quiry_time = 1
repeat_time = 5

# action_reward_time_series = torch.zeros((userset.user_num, args.trajectory_len, action_dim+1)) # state_dim +

roc_auc_buf = torch.zeros((repeat_time,))

for repeat in range(repeat_time):

    userset = Virtual_Userset(template_mdp=args.template_mdp,
                              user_num=10000,  # args.user_num,
                              user_characteristic_buf=user_characteristic_buf,
                              anomaly_num=100,
                              anomaly_prob=anomaly_prob, )

    state_dim = userset.template_mdp.observation_space.shape[0]
    action_dim = userset.template_mdp.action_space.shape[0]

    reward_time_series = np.zeros((userset.user_num, args.trajectory_len))
    reward_predict_time_series = torch.zeros((userset.user_num, args.trajectory_len))
    trajectory_time_series = torch.zeros((userset.user_num, args.trajectory_len, state_dim + action_dim))
    action_time_series = torch.zeros((userset.user_num, args.trajectory_len, action_dim))

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
            reward_time_series[user_index, step] = reward
            action_time_series[user_index, step] = action_tensor
            trajectory_time_series[user_index, step] = torch.cat((state_tensor, action_tensor), dim=0)

            state = nxt_state

    # reward_predict_time_series, _, _ = critic(trajectory_time_series[sample_index, quiry_index].unsqueeze(dim=0))
    # reward_predict_time_series[reward_predict_time_series < 0.] = 0.
    # reward_predict_time_series[reward_predict_time_series > 0.] = torch.floor(reward_predict_time_series[reward_predict_time_series > 0.] + 1)
    # feature = (reward_time_series - reward_predict_time_series.squeeze(dim=2).detach().numpy())**2

    feature = reward_time_series
    # print(reward_time_series)
    # print(reward_predict_time_series.squeeze(dim=2))
    # print(reward_time_series)

    # clf = IsolationForest()
    clf = OneClassSVM()
    clf.fit(feature)
    outlier_score = -clf.decision_function(feature)

    mdp_label = userset.user_label
    false_positive_rate, true_positive_rate, thresholds = roc_curve(mdp_label, outlier_score)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    print(roc_auc)
    roc_auc_buf[repeat] = roc_auc

roc_auc_ave = roc_auc_buf.mean()
roc_auc_std = roc_auc_buf.std(unbiased=False)
print(float(roc_auc_ave))
print(float(roc_auc_std))


#
# mdp_trajectory = reward_time_series.reshape(100, -1) # .detach().numpy()
#
# cross_distance_trajectory_space = np.zeros((userset.user_num, userset.user_num))
# for idx1 in range(userset.user_num):
#     for idx2 in range(idx1):
#         cross_distance_trajectory_space[idx1, idx2] = ((reward_time_series[idx1] - reward_time_series[idx2])**2).sum()
#
# for idx1 in range(userset.user_num):
#     for idx2 in range(idx1+1, userset.user_num):
#         cross_distance_trajectory_space[idx1, idx2] = cross_distance_trajectory_space[idx2, idx1]
#
# plt.figure()
# plt.imshow(cross_distance_trajectory_space)
# plt.colorbar()
# # plt.savefig(load_dir + '/' + args.exp + '-type' + str(args.type) + '-trajectory-heat.png')
# plt.show()
#

# cross_distance = np.zeros((userset.user_num, userset.user_num))
# for idx1 in range(userset.user_num):
#     for idx2 in range(idx1):
#         cross_distance[idx1, idx2] = ((critic_cell[idx1] - critic_cell[idx2])**2).sum()
#
# for idx1 in range(userset.user_num):
#     for idx2 in range(idx1+1, userset.user_num):
#         cross_distance[idx1, idx2] = cross_distance[idx2, idx1]
#
# # print(cross_distance)
#
# plt.figure()
# plt.imshow(cross_distance)
# plt.colorbar()
# # plt.savefig(load_dir + '/' + args.exp + '-type' + str(args.type) + '-embedding-heat.png')
# plt.show()


#
#
# plt.scatter(inlier_critic[:, 0], inlier_critic[:, 1], label='Inlier MDP')
# plt.scatter(outlier_critic[:, 0], outlier_critic[:, 1], c='red', label='Outlier MDP')
# plt.savefig(load_dir + '/' + args.exp + '-type' + str(args.type) + '2D-mapping.png')
#
# plt.show()

#
# # plt.xlabel('t')
# # plt.ylabel(r'$Q(s_t, a_t)$')
#
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.xlabel('x', fontsize=15)
# plt.ylabel('y', fontsize=15)
# plt.grid()
# plt.legend(fontsize=15)
# plt.legend(loc='lower right')

# plt.savefig(load_dir + '/' + args.exp + '-type' + str(args.type) \
#             + '-step-' + str(learn_step) + '.png')
#
# plt.show()
#

# plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.4f' % roc_auc)
# plt.show()
# print(outlier_score)

#
#
# if __name__ == '__main__':
#     random_seed(10)
#     main()
