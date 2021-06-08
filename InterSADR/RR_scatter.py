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

from InterSADR_train import Encoder, Actor, Critic, Virtual_Userset, Virtual_User
from hyperparameter_test import *

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

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

    args.trajectory_len = 10

    quiry_time = 1

    # action_reward_time_series = torch.zeros((userset.user_num, args.trajectory_len, action_dim+1)) # state_dim +
    reward_time_series = np.zeros((sample_time, quiry_time, userset.user_num, args.trajectory_len))
    action_time_series = np.random.uniform(-1., 1., (sample_time, quiry_time, args.trajectory_len, action_dim))
    mdp_anomaly_label = np.zeros((sample_time, quiry_time, userset.user_num)).astype(dtype=np.int)
    outlier_score_buf = np.zeros((sample_time, quiry_time, userset.user_num))
    tsne = manifold.TSNE(n_components=2, init='pca')  # , random_state=0)

    for quiry_index in range(quiry_time): # args.quiry_index_time):

        for user_index in range(userset.user_num):

            user = userset.user_select(user_index)
            state = user.reset()
            # print(state)
            for step in range(args.trajectory_len):
                action_array = action_time_series[sample_index, quiry_index, step]
                nxt_state, reward, done, user_anomaly = user.step(action_array)
        #
                reward_time_series[sample_index, quiry_index, user_index, step] = reward
                state = nxt_state

        feature = reward_time_series[sample_index, quiry_index]
        # print(reward_time_series[sample_index, quiry_index])
        # print(reward_predict_time_series.squeeze(dim=2))
        # print(reward_time_series)

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

# Generate heatmap

clf = OneClassSVM()
clf.fit(reward_embedding)

x_min = reward_embedding[:, 0].min() - 1
x_max = reward_embedding[:, 0].max() + 1
y_min = reward_embedding[:, 1].min() - 1
y_max = reward_embedding[:, 1].max() + 1

print(x_min, x_max, y_min, y_max)
x_mesh, y_mesh = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))

data_mesh = np.concatenate((x_mesh.reshape(x_mesh.size, -1), y_mesh.reshape(y_mesh.size, -1)), axis=1)
heatvalue = -clf.decision_function(data_mesh)
heatvalue_mesh = heatvalue.reshape(x_mesh.shape)

print(heatvalue_mesh.shape)

plt.figure()

plt.contourf(x_mesh, y_mesh, heatvalue_mesh, alpha=0.5)
plt.contour(x_mesh, y_mesh, heatvalue_mesh, alpha=0.5)

plt.scatter(x=inlier_embedding_buf[:, 0], y=inlier_embedding_buf[:, 1],
            c='black', label='NU')
plt.scatter(x=outlier_embedding_buf[:, 0], y=outlier_embedding_buf[:, 1],
            c='red', label='AU')
plt.grid()
plt.gca().legend(loc='lower right', fontsize=20) #, frameon=False)
# plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(2))
# plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(2))
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Dimension 1', fontsize=20)
plt.ylabel('Dimension 2', fontsize=20)

plt.tight_layout()
plt.savefig('figure/' + args.exp + '-type' + str(args.type) + '-random_action-heatmap' + '.pdf')
plt.show()








