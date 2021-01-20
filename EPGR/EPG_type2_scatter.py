import gym
import numpy as np
import torch
import random
import pybullet_envs
import time
from sklearn.metrics import roc_curve, auc
from sklearn import datasets, manifold

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import copy

from deep_rl import *

from LSTM_PAD_type2 import Encoder, Actor, Critic, Virtual_Userset, Virtual_User
from hyperparameter_test import *

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM


# load_dir = 'virtual_taobao-type2/virtual_taobao-type2-20201025-224609'
load_dir = 'virtual_taobao-type2/virtual_taobao-type2-20201028-202226'
# load_dir = 'virtual_taobao-type2/virtual_taobao-type2-20201029-104917'

args = copy.deepcopy(args_virtual_taobao_type2)

anomaly_prob = [0.1, -0.1]
with open('./user_characteristic_buf_100.pkl', 'rb') as r:
    user_characteristic_buf = pickle.load(r)


outlier_embedding_buf = []
inlier_embedding_buf = []
sample_time = 1

for sample_index in range(sample_time):
    userset = Virtual_Userset(template_mdp=args.template_mdp,
                                user_num=100, # args.user_num,
                                user_characteristic_buf=user_characteristic_buf,
                                anomaly_num=10,
                                anomaly_prob=anomaly_prob,)

    state_dim = userset.template_mdp.observation_space.shape[0]
    action_dim = userset.template_mdp.action_space.shape[0]

    learn_step = 1000

    args.trajectory_len = 10

    actor = torch.load(load_dir + '/actor/actor_' + str(learn_step) + '.pkl')
    critic = torch.load(load_dir + '/critic/critic_' + str(learn_step) + '.pkl')

    quiry_time = 1

    # action_reward_time_series = torch.zeros((userset.user_num, args.trajectory_len, action_dim+1)) # state_dim +
    reward_time_series = np.zeros((sample_time, quiry_time, userset.user_num, args.trajectory_len))
    reward_predict_time_series = torch.zeros((sample_time, quiry_time, userset.user_num, args.trajectory_len))
    trajectory_time_series = torch.zeros((sample_time, quiry_time, userset.user_num, args.trajectory_len, state_dim + action_dim))
    action_time_series = torch.zeros((sample_time, quiry_time, userset.user_num, args.trajectory_len, action_dim))
    roc_auc_buf = torch.zeros((sample_time, quiry_time))

    tsne = manifold.TSNE(n_components=2, init='pca') # , random_state=0)

    for quiry_index in range(quiry_time): # args.quiry_index_time):

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
                reward_time_series[sample_index, quiry_index, user_index, step] = reward
                action_time_series[sample_index, quiry_index, user_index, step] = action_tensor
                trajectory_time_series[sample_index, quiry_index, user_index, step] = torch.cat((state_tensor, action_tensor), dim=0)

                state = nxt_state

        feature = reward_time_series[sample_index, quiry_index]

        tsne.fit(feature)
        reward_embedding = tsne.fit_transform(feature + np.random.randn(userset.user_num, args.trajectory_len)*1e-8)

        # distance_feature = torch.from_numpy(feature - feature.mean(axis=0))
        # reward_embedding = torch.stack((torch.norm(distance_feature[:, 0:5], dim=1), torch.norm(distance_feature[:, 5:10], dim=1))).T

        print(reward_embedding.shape)
        # print(reward_embedding.shape)
        # print(userset.anomaly_index, userset.nomral_index)
        outlier_embedding = reward_embedding[userset.anomaly_index]
        inlier_embedding = reward_embedding[userset.nomral_index]

        for index in userset.nomral_index:
            print(feature[index], reward_embedding[index])

        outlier_embedding_buf.append(torch.tensor(outlier_embedding))
        inlier_embedding_buf.append(torch.tensor(inlier_embedding))

outlier_embedding_buf = torch.cat(outlier_embedding_buf, dim=0)
inlier_embedding_buf = torch.cat(inlier_embedding_buf, dim=0)

print(outlier_embedding_buf.shape)
print(inlier_embedding_buf.shape)

print(inlier_embedding_buf)

# Generate heatmap

clf = OneClassSVM()
clf.fit(reward_embedding)

x_min = reward_embedding[:, 0].min() - 0.1
x_max = reward_embedding[:, 0].max() + 0.1
y_min = reward_embedding[:, 1].min() - 0.1
y_max = reward_embedding[:, 1].max() + 0.1

print(x_min, x_max, y_min, y_max)
x_mesh, y_mesh = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))

data_mesh = np.concatenate((x_mesh.reshape(x_mesh.size, -1), y_mesh.reshape(y_mesh.size, -1)), axis=1)
heatvalue = -clf.decision_function(data_mesh)
heatvalue_mesh = heatvalue.reshape(x_mesh.shape)

print(heatvalue_mesh.shape)

plt.figure()

plt.contourf(x_mesh, y_mesh, heatvalue_mesh, alpha=0.5)
#画等高线
plt.contour(x_mesh, y_mesh, heatvalue_mesh, alpha=0.5)


plt.scatter(x=inlier_embedding_buf[:, 0], y=inlier_embedding_buf[:, 1],
            c='black', label='NU')
plt.scatter(x=outlier_embedding_buf[:, 0], y=outlier_embedding_buf[:, 1],
            c='red', label='AU')

plt.grid()
# plt.ylim([-4, 3])
plt.gca().legend(loc='lower right', fontsize=25) #, frameon=False)
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(1))
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.xlabel('Dimension 1', fontsize=25)
plt.ylabel('Dimension 2', fontsize=25)

plt.tight_layout()
plt.savefig('figure/' + args.exp + '-type' + str(args.type) \
            + '-PAD-heatmap' + '.pdf')
plt.show()

# roc_auc_ave = roc_auc_buf.mean()
# print(roc_auc_ave)

# critic_cell, _, _ = critic.hidden(state_action_time_series_buf.mean(dim=0))
# critic_cell = critic_cell.reshape(100, -1).detach().numpy()

# print(critic_cell[mdps.inlier_index])
# print(critic_cell[mdps.outlier_index])

# plt.plot(critic_cell[mdps.inlier_index].T, color='blue')
# plt.plot(critic_cell[mdps.outlier_index].T, color='red')
# plt.show()

# print(mse_error[mdps.inlier_index])
# print(mse_error[mdps.outlier_index])

# outlier_critic = critic_buf[mdps.outlier_index]
# inlier_critic = critic_buf[mdps.inlier_index]
# time = torch.arange(0, args.trajectory_len, 1).unsqueeze(dim=0)
# plt.scatter(x=time.quiry_index((inlier_critic.shape[0], 1)), y=inlier_critic,
#             c='black', label='Inlier MDP')
# plt.scatter(x=time.quiry_index((outlier_critic.shape[0], 1)), y=outlier_critic,
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




# mdp_trajectory = trajectory_time_series.reshape(100, -1) # .detach().numpy()
#
# cross_distance_trajectory_space = np.zeros((userset.user_num, userset.user_num))
# for idx1 in range(userset.user_num):
#     for idx2 in range(idx1):
#         cross_distance_trajectory_space[idx1, idx2] = ((trajectory_time_series[idx1] - trajectory_time_series[idx2])**2).sum()
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
