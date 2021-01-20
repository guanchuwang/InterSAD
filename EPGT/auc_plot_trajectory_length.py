import matplotlib.pyplot as plt
import torch
import seaborn as sns
import numpy as np
import os
from hyperparameter_test import *
import copy


# EPAD

# halfcheetah

# load_dir_buf = ['data_reserve/halfcheetah_iForest_quiry_time_vs_',
#                 'data_reserve/halfcheetah_PAD_quiry_time_vs_',]
# tag = 'type1_quiry_time'
# legend_buf = ['Random Action', 'Policy']

load_dir_buf = ['./data_reserve/halfcheetah_iForest_trajectory_length_vs_',
                './data_reserve/halfcheetah_ddpg_trajectory_length_vs_',
                './data_reserve/halfcheetah_PAD_trajectory_length_vs_',]
tag = 'type1_trajectory_length'
legend_buf = ['RS/iForest', 'DDPG', 'EPG']

color_buf = ['blue', 'green', 'red', 'black']

# load_dir = 'halfcheetah-type1/halfcheetah-type1-20201013-123905-success'

for dir_index in range(len(load_dir_buf)):

    load_dir = load_dir_buf[dir_index]
    auc_list = []
    for idx in range(10):
        try:
            auc = np.loadtxt(load_dir + 'roc_auc_' + str(idx) + '.txt')
            auc_list.append(auc)

        except:
            pass


    # auc0 = np.loadtxt(load_dir + '/roc_auc_0.txt')
    # auc1 = np.loadtxt(load_dir + '/roc_auc_1.txt')
    # auc2 = np.loadtxt(load_dir + '/roc_auc_2.txt')
    # auc3 = np.loadtxt(load_dir + '/roc_auc_3.txt')
    # auc4 = np.loadtxt(load_dir + '/roc_auc_4.txt')
    # auc5 = np.loadtxt(load_dir + '/roc_auc_5.txt')

    print(auc_list)
    auc_buf = np.stack(auc_list).T
    auc_mean = auc_buf.mean(axis=1)
    print(auc_buf.shape)

    # auc0 = auc0[0] + auc0[6:]
    # print(type(auc0))
    # auc0 = auc0.reshape((10, 5)).mean(axis=1)
    learn_step_buf = np.arange(0, len(auc_buf)) + 1 # *100

    neighbour_len = 2
    auc_ucb = np.array([auc_buf[max(idx-neighbour_len, 0):idx+1].max()
                                for idx in range(len(auc_buf))])

    auc_lcb = np.array([auc_buf[max(idx-neighbour_len, 0):idx+1].min()
                                for idx in range(len(auc_buf))])

    # auc_mean = np.array([auc_mean[max(idx-neighbour_len, 0):idx+1].min()
    #                             for idx in range(len(auc_buf))])

    auc = [auc_mean, auc_ucb, auc_lcb]#, auc1] # , auc2, auc3, auc4, auc5]

    print(auc_ucb)

    # plt.figure()
    h_pad = sns.tsplot(time=learn_step_buf, data=auc, color=color_buf[dir_index], condition=legend_buf[dir_index])

plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
# plt.xlabel('Interaction Time', fontsize=15)
plt.xlabel('Trajecotry Length', fontsize=25)
plt.ylabel('ROC/AUC', fontsize=25)
plt.grid()
plt.legend(loc='lower right', fontsize=25, frameon=False) #
# plt.ylim([0.69, 1.01])
plt.ylim([0.4, 1.01])
plt.xlim([1, 10])
plt.tight_layout()
plt.savefig(os.path.join('./figure', 'halfcheetah_' + tag + '_auc.pdf'))
plt.show()