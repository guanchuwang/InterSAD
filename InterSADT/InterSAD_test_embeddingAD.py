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
from hyperparameter_test import *

from sklearn.ensemble import IsolationForest
# from sklearn.svm import OneClassSVM

# from pybullet_envs.gym_locomotion_envs import HalfCheetah_Type2Anomaly_BulletEnv

# def main():

# random_seed(10)

# halfcheetah

load_dir = 'halfcheetah-type1/halfcheetah-type1-20210527-151143' # 0. dim=16 no STU

if load_dir[0:3] == 'hal':
    args = copy.deepcopy(args_half_cheetah_type1)
elif load_dir[0:3] == 'hop':
    args = copy.deepcopy(args_hopper_type1)
elif load_dir[0:3] == 'wal':
    args = copy.deepcopy(args_walker_type1)
elif load_dir[0:3] == 'ant':
    args = copy.deepcopy(args_ant_type1)

noise_std_buf = list(np.arange(0, 0.2, 0.01))
print(noise_std_buf)


mdp_num = 100
outlier_num = 10
mdp_inds = np.array([mdp_index for mdp_index in range(mdp_num)])
outlier_inds = np.random.choice(mdp_inds, outlier_num, replace=False)
inlier_inds = np.array(list(set(mdp_inds) - set(outlier_inds)))
mdp_label = [1 if mdp_index in outlier_inds else 0 for mdp_index in mdp_inds]

state_dim = gym.make(args.inlier_mdps[0]).observation_space.shape[0]
action_dim = gym.make(args.inlier_mdps[0]).action_space.shape[0]

roc_auc_buf = torch.zeros((len(noise_std_buf), ))
test_learn_step = 4500

actor = torch.load(load_dir + '/actor/actor_' + str(test_learn_step) + '.pkl')
critic = torch.load(load_dir + '/critic/critic_' + str(test_learn_step) + '.pkl')

outlier_score = torch.tensor([0. for mdp_index in mdp_inds])

critic_cell_buf = np.zeros((args.repeat_time, mdp_num, args.encode_dim*args.trajectory_len))
state_action_time_series_buf = torch.zeros((args.repeat_time, mdp_num, args.trajectory_len, state_dim + action_dim))

for noise_index in range(len(noise_std_buf)):
    noise_std = noise_std_buf[noise_index]
    state_action_time_series = torch.zeros((mdp_num, args.trajectory_len, state_dim + action_dim))
    state_time_series = torch.zeros((mdp_num, args.trajectory_len, state_dim))

    for mdp_index in range(mdp_num):
        if mdp_index in outlier_inds:
            outlier_mdp_sample = np.random.choice(args.outlier_mdps)
            mdp = outlier_mdp_sample
        else:
            inlier_mdp_sample = np.random.choice(args.inlier_mdps)
            mdp = gym.make(inlier_mdp_sample)

        state_noise = np.random.normal(0, noise_std, (state_dim, ))
        state = mdp.reset() + state_noise # state_normalizer(mdp.reset() + state_noise)
        # print(state)
        for step in range(args.trajectory_len):
            state_tensor = torch.tensor(state, dtype=torch.float)
            action_tensor = actor(state_tensor)
            action_array = action_tensor.detach().numpy()
            nxt_state, reward, done, _ = mdp.step(action_array)

            state_noise = np.random.normal(0, noise_std, (state_dim,))

            # print(state_noise)
            nxt_state += state_noise # state_normalizer(nxt_state + np.random.normal(0, 0.05, (state_dim, )))
            state_action_time_series[mdp_index, step] = torch.cat((state_tensor, action_tensor), dim=0)
            state_time_series[mdp_index, step] = state_tensor
            state = nxt_state
            # print(action_tensor)

        # print(torch.cat((state_tensor, action_tensor), dim=0))

            # print(state_action_time_series.shape)
    # print(state_action_time_series.shape)
    # print(critic.hidden(state_action_time_series).shape)

    critic_encoding, critic_hidden, critic_cell = critic.hidden(state_action_time_series) # shape: (mdp_num, hidden_size)

    # feature = critic_encoding.reshape(mdp_num, -1).detach().numpy()
    feature = state_time_series.reshape(mdp_num, -1).detach().numpy()

    clf = IsolationForest()
    # clf = OneClassSVM()
    # clf = LocalOutlierFactor(algorithm='auto', contamination=0.01)

    clf.fit(feature)
    outlier_score = -clf.decision_function(feature)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(mdp_label, outlier_score)
    roc_auc = auc(false_positive_rate, true_positive_rate)

    roc_auc_buf[noise_index] = roc_auc
    print(noise_std, roc_auc)


np.savetxt(load_dir + '/' + 'noise_roc_auc.txt', roc_auc_buf)

plt.plot(noise_std_buf, roc_auc_buf)
plt.xlabel('Noise std')
plt.ylabel('ROC-AUC')
# plt.legend(loc='lower right')
plt.savefig(load_dir + '/' + args.exp + '-auc_vs_noise' + '.png')
plt.show()

# # outlier_critic = critic_buf[mdps.outlier_index]
# # inlier_critic = critic_buf[mdps.inlier_index]
#
#
# # tsne = manifold.TSNE(n_components=2, perplexity=30.0, early_exaggeration=12.0, learning_rate=200.0,
# #                      n_iter=1000, n_iter_without_progress=300, min_grad_norm=1e-07, metric='euclidean',
# #                      init='random', verbose=0, random_state=0, method='barnes_hut', angle=0.5)
# tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
#
# critic_embedding = tsne.fit_transform(critic_buf)
#
# outlier_critic = critic_embedding[mdps.outlier_index]
# inlier_critic = critic_embedding[mdps.inlier_index]
#
# # print(critic_buf)
# # plt.plot(state_buf_buf.squeeze(axis=2).T)
# # time = torch.arange(0, args.trajectory_len, 1).unsqueeze(dim=0)
# # plt.plot(critic_buf.T)
# # plt.boxplot(critic_buf.T)
# # plt.scatter(x=time.repeat((inlier_critic.shape[0], 1)), y=inlier_critic,
# #             c='black', label='Inlier MDP')
# # plt.scatter(x=time.repeat((outlier_critic.shape[0], 1)), y=outlier_critic,
# #             c='red', label='Outlier MDP')
#
# plt.scatter(inlier_critic[:, 0], inlier_critic[:, 1], label='Inlier MDP')
# plt.scatter(outlier_critic[:, 0], outlier_critic[:, 1], c='red', label='Outlier MDP')
#
# # plt.xlabel('t')
# # plt.ylabel(r'$Q(s_t, a_t)$')
#
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend(loc='lower right')
#
# plt.savefig(load_dir + '/' + args.exp + '-type' + str(args.type) \
#             + '-step-' + str(learn_step) + '.png')
# plt.show()



#
#
# if __name__ == '__main__':
#     random_seed(10)
#     main()
