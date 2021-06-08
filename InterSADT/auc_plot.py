import matplotlib.pyplot as plt
import torch
import seaborn as sns
import numpy as np
import os
from hyperparameter_test import *
import copy
import matplotlib.ticker as ticker

## Embedding dim
load_dir_buf = [
                'halfcheetah-type1/halfcheetah-type1-20201102-214433/',
                'halfcheetah-type1/halfcheetah-type1-20201102-202101/',
                'halfcheetah-type1/halfcheetah-type1-20201103-082949/',
                ]
tag = 'Encode_dim'
legend_buf = ['$R^{240}$', '$R^{160}$', '$R^{80}$']
x_range = [0, 4000]

## Ablation Exp.

# ## Ablation Halfcheetah.
# load_dir_buf = ['halfcheetah-type1/halfcheetah-type1-20201102-202101/',
#                 'halfcheetah-type1/halfcheetah-type1-20210521-115315/',
#                 'halfcheetah-type1/halfcheetah-type1-20201102-202101/encoder-',
#                 ]
# tag = 'Ablation-HC'
# legend_buf = ['InterSAD', 'No ERM', 'No SEN']
# x_range = [1, 4000]

# ## Ablation ERM Walker2D.
# load_dir_buf = ['walker-type1/walker-type1-20210527-122827/',
#                 'walker-type1/walker-type1-20210527-171229/',
#                 'walker-type1/walker-type1-20210527-122827/encoder-',
#                 ]
# tag = 'Ablation-WK'
# legend_buf = ['InterSAD', 'No ERM', 'No policy']
# x_range = [1500, 4800]


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

    # auc0 = auc0[0] + auc0[6:]
    # print(type(auc0))
    # auc0 = auc0.reshape((10, 5)).mean(axis=1)
    learn_step_buf = np.arange(0, len(auc_buf))*100

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
    h_pad = sns.tsplot(time=learn_step_buf, data=auc, color=color_buf[dir_index], condition=legend_buf[dir_index])

# fig, ax = plt.subplots(1,1)
# plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1000))
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Iteration', fontsize=20)
plt.ylabel('ROC-AUC', fontsize=20)
plt.grid()
plt.xticks(list(learn_step_buf[::8]))
plt.legend(loc='lower right', fontsize=20, frameon=False)
plt.ylim([0.4, 1.01])
plt.xlim(x_range)

exp_type = load_dir_buf[0].split('/')[0]

plt.tight_layout()
plt.savefig(os.path.join('./figure', exp_type + tag + '-auc_time.pdf'))
plt.show()