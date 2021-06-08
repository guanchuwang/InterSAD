import logging, argparse, time, glob, sys, os
import copy

import gym
import virtualTB

parser = argparse.ArgumentParser("Hyperparameter")
parser.add_argument('--exp', type=str, default='virtual_taobao', help='experiment name')
parser.add_argument('--encoder_hidden_dim', type=int, default=256, help='output dimension of encoder')
parser.add_argument('--encode_dim', type=int, default=64, help='output dimension of encoder')

# parser.add_argument('--type', type=int, default=1, help='Anomaly type')
# parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
# parser.add_argument('--cpu', type=int, default=0, help='cpu device id')

#-----------------Common Hyperparameter------------------#
args_common = parser.parse_args()
args_common.critic_lr = 1e-4
args_common.actor_lr = 1e-4 # 1e-3
args_common.replay_mem_capacity = 1000000
args_common.batch_size = 32
args_common.warm_up = 1000
args_common.target_update = 5e-3
args_common.save_interval = 100
args_common.seed = 7
args_common.report_interval = 100

#-----------------Virtual Taobao------------------#
args_virtual_taobao_type2 = copy.deepcopy(args_common)
args_virtual_taobao_type2.mdp_num = 100
args_virtual_taobao_type2.template_mdp = 'VirtualTB-v0'
args_virtual_taobao_type2.outlier_rate = 0. # 5

# args_virtual_taobao_type2.encoder_hidden_dim = 128
args_virtual_taobao_type2.trajectory_len = 10
args_virtual_taobao_type2.learn_step = 10000
args_virtual_taobao_type2.reward_noise_std = 0. # 0.05
args_virtual_taobao_type2.state_noise_std = 0. # 0.05
args_virtual_taobao_type2.type = 2

args_virtual_taobao_type2.save_dir = os.path.join(args_virtual_taobao_type2.exp + '-type' + str(args_virtual_taobao_type2.type),
                                                args_virtual_taobao_type2.exp + '-type' + str(args_virtual_taobao_type2.type)
                                                + '-{}'.format(time.strftime("%Y%m%d-%H%M%S")))



args_virtual_taobao_type2.inlier_mdps = ['VirtualTB-v0'] # ['HalfCheetahBulletEnv-v0']
args_virtual_taobao_type2.outlier_mdps = []













