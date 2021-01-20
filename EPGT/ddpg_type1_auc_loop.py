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

from LSTM_PAD_type1 import MDPset
from DDPG_agent import Agent, Actor, Critic
from hyperparameter_test import *

from sklearn.ensemble import IsolationForest
# from sklearn.svm import OneClassSVM

# from pybullet_envs.gym_locomotion_envs import HalfCheetah_Type2Anomaly_BulletEnv

# random_seed(10)

# halfcheetah
# load_dir = 'DDPG-halfcheetah-type1/halfcheetah-type1-20201022-152429' #

# hopper
# load_dir = 'DDPG-hopper-type1/hopper-type1-20201022-153958' #

# walker
# load_dir = 'DDPG-walker-type1/walker-type1-20201022-154509' #

# ant
load_dir = 'DDPG-ant-type1/ant-type1-20201022-160140'

if load_dir[5:8] == 'hal':
    args = copy.deepcopy(args_half_cheetah_type1)
elif load_dir[5:8] == 'hop':
    args = copy.deepcopy(args_hopper_type1)
elif load_dir[5:8] == 'wal':
    args = copy.deepcopy(args_walker_type1)
elif load_dir[5:8] == 'ant':
    args = copy.deepcopy(args_ant_type1)


# learn_step_buf = [0, 1000, 2000, 3000, 4000, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000]
learn_step_buf = list(range(0, 5100, 100))
# learn_step_buf = list(range(5000, 10100, 100))
# learn_step_buf = list(range(0, 1100, 100))

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

# critic_cell_buf = np.zeros((args.repeat_time, args.mdp_num, 32))
roc_auc_buf = np.array([0. for learn_step in learn_step_buf])

args.trajectory_len = 10

for idx, learn_step in enumerate(learn_step_buf):

    actor = torch.load(load_dir + '/actor/actor_' + str(learn_step) + '.pkl')
    critic = torch.load(load_dir + '/critic/critic_' + str(learn_step) + '.pkl')

    outlier_score = torch.tensor([0. for mdp in mdps.mdp_buf])
    state_action_time_series = torch.zeros((args.mdp_num, args.trajectory_len, state_dim))
    critic_time_series = torch.zeros((args.mdp_num, args.trajectory_len, state_dim+action_dim))

    for mdp_index in range(args.mdp_num):
        mdp = mdps.mdp_buf[mdp_index]
        state_noise = np.random.normal(0, args.state_noise_std, (state_dim, ))
        state = state_normalizer(mdp.reset() + state_noise) # state_normalizer(mdp.reset() + state_noise)
        # print(state)
        for step in range(args.trajectory_len):
            state_tensor = torch.tensor(state, dtype=torch.float)
            action_tensor = actor(state_tensor)
            action_array = action_tensor.detach().numpy()
            nxt_state, reward, done, _ = mdp.step(action_array)

            state_noise = np.random.normal(0, args.state_noise_std, (state_dim,))
            nxt_state += state_noise # state_normalizer(nxt_state + np.random.normal(0, 0.05, (state_dim, )))
            nxt_state = state_normalizer(nxt_state)

            critic_time_series[mdp_index, step] = critic(state_tensor.unsqueeze(dim=0), action_tensor.unsqueeze(dim=0)).detach()
            # critic_time_series[mdp_index, step] = reward
            # state_action_time_series[mdp_index, step] = torch.cat((state_tensor, action_tensor), dim=0)
            # state_action_time_series[mdp_index, step] = torch.from_numpy(nxt_state)
            state = nxt_state
            # print(action_tensor)

    # mdp_time_series = state_action_time_series.reshape(args.mdp_num, -1).detach()
    mdp_time_series = critic_time_series.reshape(args.mdp_num, -1).detach()

    clf = IsolationForest()
    # clf = OneClassSVM()

    clf.fit(mdp_time_series)
    outlier_score = -clf.decision_function(mdp_time_series)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(mdps.mdp_label, outlier_score)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    # print(outlier_score)
    print(learn_step, roc_auc)

    roc_auc_buf[idx] = roc_auc

np.savetxt(load_dir + '/' + 'roc_auc.txt', roc_auc_buf)

time = np.array(learn_step_buf)
plt.plot(time, roc_auc_buf)
plt.xlabel('step')
plt.ylabel('ROC/AUC')
# plt.legend(loc='lower right')
plt.savefig(load_dir + '/' + args.exp + '-type' + str(args.type) + '.png')
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
