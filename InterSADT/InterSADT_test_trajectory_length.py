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

from InterSADT_noSTU import Encoder, Decoder, Actor, Critic, MDPset
from hyperparameter_test import *

from sklearn.ensemble import IsolationForest
# from sklearn.svm import OneClassSVM

# from pybullet_envs.gym_locomotion_envs import HalfCheetah_Type2Anomaly_BulletEnv

# def main():

# random_seed(10)

# halfcheetah
# load_dir = 'halfcheetah-type1/halfcheetah-type1-20201012-175741-success' # state_noise_std = 0.05 dim=16
# load_dir = 'halfcheetah-type1/halfcheetah-type1-20201013-123905-success' # state_noise_std = 0.05 dim=8
# load_dir = 'halfcheetah-type1/halfcheetah-type1-20201013-132427-success' # state_noise_std = 0.05 dim=32
# load_dir = 'halfcheetah-type1/halfcheetah-type1-20201102-202101' # state_noise_std = 0.05 dim=16 bad

load_dir = 'halfcheetah-type1/halfcheetah-type1-20201103-111722' # state_noise_std = 0.05 dim=16 复现 good
# load_dir = 'halfcheetah-type1/halfcheetah-type1-20201103-115530' # state_noise_std = 0.05 dim=16 复现
# load_dir = 'halfcheetah-type1/halfcheetah-type1-20201103-124532' # state_noise_std = 0.05 dim=16 复现

# hopper
# load_dir = 'hopper-type1/hopper-type1-20201020-003934'
# load_dir = 'hopper-type1/hopper-type1-20201020-082123'
# load_dir = 'hopper-type1/hopper-type1-20201021-164354'

# walker
# load_dir = 'walker-type1/walker-type1-20201020-224319' # state_noise_std = 0. dim=14
# load_dir = 'walker-type1/walker-type1-20201021-013018' # state_noise_std = 0. dim=7
# load_dir = 'walker-type1/walker-type1-20201021-150642' # state_noise_std = 0.05 dim=14

# ant
# load_dir = 'ant-type1/ant-type1-20201021-154917' # state_noise_std = 0.05 dim=18

if load_dir[0:3] == 'hal':
    args = copy.deepcopy(args_half_cheetah_type1)
elif load_dir[0:3] == 'hop':
    args = copy.deepcopy(args_hopper_type1)
elif load_dir[0:3] == 'wal':
    args = copy.deepcopy(args_walker_type1)
elif load_dir[0:3] == 'ant':
    args = copy.deepcopy(args_ant_type1)


mdps = MDPset(mdp_num=100, # args.mdp_num,
              template_mdp=args.template_mdp,
              inlier_mdps=args.inlier_mdps,  # 'HalfCheetahBulletmdp-v0',
              outlier_mdps=args.outlier_mdps,  # ['HalfCheetahBrother_0_06_BulletEnv-v0'],
              outlier_num=args.outlier_num)

state_dim = mdps.template_mdp.observation_space.shape[0]
action_dim = mdps.template_mdp.action_space.shape[0]

state_normalizer = RescaleNormalizer()
reward_normalizer = RescaleNormalizer()

sample_time = 10 # 100
max_trajectory_lengh = 10
quiry_time = 1
roc_auc_vs_trajectory_lengh = np.zeros((max_trajectory_lengh,))

learn_step = 2000

actor = torch.load(load_dir + '/actor/actor_' + str(learn_step) + '.pkl')
critic = torch.load(load_dir + '/critic/critic_' + str(learn_step) + '.pkl')


for trajectory_len in range(1, max_trajectory_lengh+1, 1):
    roc_auc_buf = torch.zeros((sample_time,))
    state_action_time_series = np.zeros((quiry_time, args.mdp_num, args.trajectory_len, state_dim + action_dim))

    for sample_index in range(sample_time):

        feature_buf_list = []
        for repeat in range(quiry_time):

            state_action_time_series = torch.zeros((args.mdp_num, args.trajectory_len, state_dim + action_dim))

            for mdp_index in range(args.mdp_num):
                mdp = mdps.mdp_buf[mdp_index]
                state_noise = np.random.normal(0, args.state_noise_std, (state_dim, ))
                state = mdp.reset() + state_noise # state_normalizer(mdp.reset() + state_noise)
                # print(state)
                for step in range(trajectory_len):
                    state_tensor = torch.tensor(state, dtype=torch.float)
                    action_tensor = actor(state_tensor)
                    action_array = action_tensor.detach().numpy()
                    nxt_state, reward, done, _ = mdp.step(action_array)

                    state_noise = np.random.normal(0, args.state_noise_std, (state_dim,))
                    nxt_state += state_noise # state_normalizer(nxt_state + np.random.normal(0, 0.05, (state_dim, )))
                    state_action_time_series[mdp_index, step] = torch.cat((state_tensor, action_tensor), dim=0)
                    state = nxt_state
                    # print(action_tensor)

            critic_encoding, critic_hidden, critic_cell = critic.hidden(state_action_time_series)  # shape: (mdp_num, hidden_size)

            # feature = critic_encoding[:, -1, :].detach().numpy()
            # feature = critic_encoding.min(dim=1).values.detach().numpy()
            # feature = critic_encoding.mean(dim=1).detach().numpy()
            feature = critic_encoding.reshape(args.mdp_num, -1).detach()

            feature_buf_list.append(feature)

        feature_buf_mean = torch.stack(feature_buf_list).mean(dim=0)

        clf = IsolationForest()
        clf.fit(feature_buf_mean)
        outlier_score = -clf.decision_function(feature_buf_mean)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(mdps.mdp_label, outlier_score)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        # print(outlier_score)

        roc_auc_buf[sample_index] = roc_auc
        print(sample_index, roc_auc)

    roc_auc_vs_trajectory_lengh[trajectory_len - 1] = roc_auc_buf.mean()
    print(trajectory_len, roc_auc_buf.mean())

np.savetxt('./data_reserve/' + args.exp + '_PAD_trajectory_length_vs_roc_auc.txt', roc_auc_vs_trajectory_lengh)


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
