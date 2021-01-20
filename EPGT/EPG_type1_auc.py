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
import os

from deep_rl import *

from LSTM_PAD_type1 import Encoder, Decoder, Actor, Critic, MDPset
from hyperparameter_test import *

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

# halfcheetah
# load_dir = 'halfcheetah-type1/halfcheetah-type1-20201103-111722' # state_noise_std = 0.05 dim=16 good

# walker
# load_dir = 'walker-type1/walker-type1-20201021-150642' # state_noise_std = 0.05 dim=14

# ant
load_dir = 'ant-type1/ant-type1-20201021-154917' # state_noise_std = 0.05 dim=18


if load_dir[0:3] == 'hal':
    args = copy.deepcopy(args_half_cheetah_type1)
elif load_dir[0:3] == 'hop':
    args = copy.deepcopy(args_hopper_type1)
elif load_dir[0:3] == 'wal':
    args = copy.deepcopy(args_walker_type1)
elif load_dir[0:3] == 'ant':
    args = copy.deepcopy(args_ant_type1)

# mdps = MDPset(mdp_num=args.mdp_num,
#               template_mdp=args.template_mdp,
#               inlier_mdps=args.inlier_mdps,  # 'HalfCheetahBulletmdp-v0',
#               outlier_mdps=args.outlier_mdps,  # ['HalfCheetahBrother_0_06_BulletEnv-v0'],
#               outlier_num=args.outlier_num)
#               # outlier_mdps=[HalfCheetah_Type2Anomaly_BulletEnv(noise_std=0.01)],  # ['HalfCheetahBrother_0_06_BulletEnv-v0'],
#               # outlier_rate=0.05)

mdp_num = 10000
outlier_num = 100
mdp_inds = np.array([mdp_index for mdp_index in range(mdp_num)])
outlier_inds = np.random.choice(mdp_inds, outlier_num, replace=False)
inlier_inds = np.array(list(set(mdp_inds) - set(outlier_inds)))
mdp_label = [1 if mdp_index in outlier_inds else 0 for mdp_index in mdp_inds]

print(outlier_inds)

state_dim = gym.make(args.inlier_mdps[0]).observation_space.shape[0]
action_dim = gym.make(args.inlier_mdps[0]).action_space.shape[0]

learn_step = 2000
args.trajectory_len = 10
args.repeat_time = 5 #10

roc_auc_buf = torch.zeros((args.repeat_time, ))

actor = torch.load(load_dir + '/actor/actor_' + str(learn_step) + '.pkl')
critic = torch.load(load_dir + '/critic/critic_' + str(learn_step) + '.pkl')

outlier_score = torch.tensor([0. for mdp_index in mdp_inds])

critic_cell_buf = np.zeros((args.repeat_time, mdp_num, args.encode_dim*args.trajectory_len))
state_action_time_series_buf = torch.zeros((args.repeat_time, mdp_num, args.trajectory_len, state_dim + action_dim))

for repeat in range(args.repeat_time):
    state_action_time_series = torch.zeros((mdp_num, args.trajectory_len, state_dim + action_dim))

    for mdp_index in range(mdp_num):
        if mdp_index in outlier_inds:
            outlier_mdp_sample = np.random.choice(args.outlier_mdps)
            mdp = outlier_mdp_sample
        else:
            inlier_mdp_sample = np.random.choice(args.inlier_mdps)
            mdp = gym.make(inlier_mdp_sample)

        state_noise = np.random.normal(0, args.state_noise_std, (state_dim, ))
        state = mdp.reset() + state_noise # state_normalizer(mdp.reset() + state_noise)
        # print(state)
        for step in range(args.trajectory_len):
            state_tensor = torch.tensor(state, dtype=torch.float)
            action_tensor = actor(state_tensor)
            action_array = action_tensor.detach().numpy()
            nxt_state, reward, done, _ = mdp.step(action_array)

            state_noise = np.random.normal(0, args.state_noise_std, (state_dim,))

            # print(state_noise)
            nxt_state += state_noise # state_normalizer(nxt_state + np.random.normal(0, 0.05, (state_dim, )))
            state_action_time_series[mdp_index, step] = torch.cat((state_tensor, action_tensor), dim=0)
            state = nxt_state
            # print(action_tensor)

        # print(torch.cat((state_tensor, action_tensor), dim=0))

            # print(state_action_time_series.shape)
    # print(state_action_time_series.shape)
    # print(critic.hidden(state_action_time_series).shape)

    critic_encoding, critic_hidden, critic_cell = critic.hidden(state_action_time_series) # shape: (mdp_num, hidden_size)

    # print(critic_cell.shape)
    # critic_cell_buf[repeat] = critic_cell.squeeze(dim=0).detach().numpy()

    # print(critic_encoding.shape)
    # feature = critic_encoding.min(dim=1).values.detach().numpy()
    # feature = critic_encoding.mean(dim=1).detach().numpy()
    feature = critic_encoding.reshape(mdp_num, -1).detach().numpy()

    clf = IsolationForest()
    # clf = OneClassSVM()
    # clf = LocalOutlierFactor(algorithm='auto', contamination=0.01)

    clf.fit(feature)
    outlier_score = -clf.decision_function(feature)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(mdp_label, outlier_score)
    roc_auc = auc(false_positive_rate, true_positive_rate)

    roc_auc_buf[repeat] = roc_auc
    print(repeat, roc_auc)

roc_auc_ave = roc_auc_buf.mean()
roc_auc_std = roc_auc_buf.std(unbiased=False)
print(roc_auc_ave)
print(roc_auc_std)


# plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.4f' % roc_auc)
# plt.show()
# print(outlier_score)


# critic_cell, _, _ = critic.hidden(state_action_time_series_buf.mean(dim=0))
# critic_cell = critic_cell.reshape(100, -1).detach().numpy()

# print(critic_cell[mdps.inlier_index])
# print(critic_cell[mdps.outlier_index])

# plt.plot(critic_cell[mdps.inlier_index].T, color='blue')
# plt.plot(critic_cell[mdps.outlier_index].T, color='red')
# plt.show()

# outlier_critic = critic_buf[mdps.outlier_index]
# inlier_critic = critic_buf[mdps.inlier_index]
# time = torch.arange(0, args.trajectory_len, 1).unsqueeze(dim=0)
# plt.scatter(x=time.repeat((inlier_critic.shape[0], 1)), y=inlier_critic,
#             c='black', label='Inlier MDP')
# plt.scatter(x=time.repeat((outlier_critic.shape[0], 1)), y=outlier_critic,
#             c='red', label='Outlier MDP')

# print(critic_buf)
# plt.plot(state_buf_buf.squeeze(axis=2).T)
# time = torch.arange(0, args.trajectory_len, 1).unsqueeze(dim=0)
# plt.plot(critic_buf.T)
# plt.boxplot(critic_buf.T)

#
# tsne = manifold.TSNE(n_components=2, perplexity=30.0, early_exaggeration=12.0, learning_rate=200.0,
#                      n_iter=1000, n_iter_without_progress=300, min_grad_norm=1e-07, metric='euclidean',
#                      init='random', verbose=0, random_state=None, method='barnes_hut', angle=0.5)
# tsne = manifold.TSNE(n_components=2, init='pca') # , random_state=0)

# critic_embedding = tsne.fit_transform(critic_cell)

# outlier_critic = critic_embedding[mdps.outlier_index]
# inlier_critic = critic_embedding[mdps.inlier_index]

# mdp_trajectory = state_action_time_series.reshape(100, -1).detach().numpy()
#
# cross_distance_trajectory_space = np.zeros((args.mdp_num, args.mdp_num))
# for idx1 in range(args.mdp_num):
#     for idx2 in range(idx1):
#         cross_distance_trajectory_space[idx1, idx2] = ((state_action_time_series[idx1] - state_action_time_series[idx2])**2).sum()
#
# for idx1 in range(args.mdp_num):
#     for idx2 in range(idx1+1, args.mdp_num):
#         cross_distance_trajectory_space[idx1, idx2] = cross_distance_trajectory_space[idx2, idx1]
#
# # print(cross_distance)
#
# plt.figure()
# plt.imshow(cross_distance_trajectory_space)
# plt.colorbar()
# plt.savefig(load_dir + '/' + args.exp + '-type' + str(args.type) + '-trajectory-heat.png')
#

# cross_distance = np.zeros((args.mdp_num, args.mdp_num))
# for idx1 in range(args.mdp_num):
#     for idx2 in range(idx1):
#         cross_distance[idx1, idx2] = ((critic_cell[idx1] - critic_cell[idx2])**2).sum()
#
# for idx1 in range(args.mdp_num):
#     for idx2 in range(idx1+1, args.mdp_num):
#         cross_distance[idx1, idx2] = cross_distance[idx2, idx1]

# print(cross_distance)

# plt.figure()
# plt.imshow(np.tanh(0.5*np.log2(cross_distance)))
# plt.colorbar()

# plt.figure()
# plt.imshow(cross_distance, cmap='gray')
# cbar = plt.colorbar()
# cbar.set_label('Distance', fontsize=18)
# cbar.ax.tick_params(labelsize=15)
# plt.scatter(mdps.inlier_index, mdps.inlier_index, color='blue')
# plt.scatter(mdps.outlier_index, mdps.outlier_index, color='red')
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.xlabel('MDP index', fontsize=18)
# plt.ylabel('MDP index', fontsize=18)
# plt.savefig(os.path.join('figure', 'PAD' + args.exp + '-type' + str(args.type) + '-trajectory-heat.png'))
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


#
#
# if __name__ == '__main__':
#     random_seed(10)
#     main()
