import matplotlib.pyplot as plt
import torch
import seaborn as sns
import numpy as np
import os
from hyperparameter_test import *
import copy
import matplotlib.ticker as ticker

# Halfcheetah.
load_dir_buf = ['halfcheetah-type1/halfcheetah-type1-20210527-151143/embedding_ad/noise_',
                'halfcheetah-type1/halfcheetah-type1-20210527-151143/trajectory_ad/noise_',
                ]

tag = 'AUC_Noise'
legend_buf = ['$\phi(u_i, \{ u_j \}_{1\leq j\leq N})$', '$\phi(τ_i, \{ τ_j \}_{1\leq j\leq N})$']
x_range = [0, 0.18]

color_buf = ['blue', 'red', 'green', 'black']

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

    noise_std_buf = list(np.arange(0, 0.2, 0.01))

    neighbour_len = 2

    # auc_mean = np.array([auc_buf[max(idx - neighbour_len, 0):idx + 1].mean()
    #                     for idx in range(len(auc_buf))])

    auc_ucb = np.array([auc_buf[max(idx-neighbour_len, 0):idx+1].max()
                                for idx in range(len(auc_buf))])

    auc_lcb = np.array([auc_buf[max(idx-neighbour_len, 0):idx+1].min()
                                for idx in range(len(auc_buf))])

    # auc_mean = np.array([auc_mean[max(idx-5, 0):idx+1].min()
    #                             for idx in range(len(auc_buf))])

    auc = [auc_mean, auc_ucb, auc_lcb]#, auc1] # , auc2, auc3, auc4, auc5]

    # index = np.where(learn_step_buf == 2000)[0]
    # print(index, auc_mean[index])

    # plt.figure()
    h_pad = sns.tsplot(time=noise_std_buf, data=auc, color=color_buf[dir_index], condition=legend_buf[dir_index])

# fig, ax = plt.subplots(1,1)
# plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1000))
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Noise STD', fontsize=20)
plt.ylabel('ROC-AUC', fontsize=20)
plt.grid()
plt.legend(loc='lower left', fontsize=20, frameon=False)
plt.xticks(list(np.arange(0, 0.2, 0.03)))
plt.ylim([0.65, 1.01])
plt.xlim([0, 0.17])

exp_type = load_dir_buf[0].split('/')[0]

plt.tight_layout()
plt.savefig(os.path.join('./figure', exp_type + tag + '-noise_std_auc_time.pdf'))
plt.show()