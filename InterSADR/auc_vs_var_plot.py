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

from hyperparameter_test import *

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import matplotlib.ticker as ticker

# load_dir = 'virtual_taobao-type2/virtual_taobao-type2-20201025-224609'
load_dir = 'virtual_taobao-type2/virtual_taobao-type2-20201028-202226'

var_vs_roc_buf_array = np.load(os.path.join(load_dir, "roc_var50.npy"))

print(var_vs_roc_buf_array.shape)

var_vs_roc_mean = var_vs_roc_buf_array.mean(axis=0).T
re_index = var_vs_roc_mean[0].argsort()
var_vs_roc_mean = var_vs_roc_mean[:, re_index]

print(var_vs_roc_buf_array[0])

var_vs_roc_buf_array[:, :, 1] = var_vs_roc_buf_array[:, re_index, 1]

print(var_vs_roc_buf_array[0])

var_vs_roc_buf = np.concatenate(list(var_vs_roc_buf_array), axis=0).T


print(var_vs_roc_buf.shape)

# print(var_vs_roc_buf_array)
learn_step_buf = var_vs_roc_buf[2].astype(np.int)

for learn_step in var_vs_roc_buf_array[0, :, 2]:

    mask = (learn_step_buf == int(learn_step))
    plt.scatter(var_vs_roc_buf[0, mask], var_vs_roc_buf[1, mask], label=(str(int(learn_step)) + ' iterations'))

# plt.plot(var_vs_roc_mean[0], var_vs_roc_mean[1], linewidth=3, linestyle='--')
for index in range(var_vs_roc_mean.shape[1]-1):
    print(var_vs_roc_mean[2,index])
    plt.gca().arrow(var_vs_roc_mean[0, index],
                    var_vs_roc_mean[1, index],
                    var_vs_roc_mean[0, index+1] - var_vs_roc_mean[0, index],
                    var_vs_roc_mean[1, index+1] - var_vs_roc_mean[1, index],
                    width=0.005,
                    length_includes_head=True, # 增加的长度包含箭头部分
                    head_width=0.02,
                    head_length=0.04,
                    fc='b',
                    ec='b',
                    label="Train direction")

# # plt.legend(loc='lower right')
# plt.savefig(load_dir + '/' + args.exp + '-type' + str(args.type) + '.png')

# plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1000))
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('ROC-AUC', fontsize=20)
plt.ylabel('$\sum_{i=1}^N(u_i - u_c)^2$', fontsize=20)
plt.grid()
plt.legend(loc='upper right', fontsize=15, frameon=False)
# plt.ylim([0.4, 1.01])
# plt.xlim([0.3, 1.01])

# exp_type = load_dir_buf[0].split('/')[0]

plt.tight_layout()
plt.savefig(os.path.join('./figure', "var_vs_roc.pdf"))
plt.show()
