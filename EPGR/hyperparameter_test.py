import logging, argparse, time, glob, sys, os
import copy


import gym
import virtualTB

parser = argparse.ArgumentParser("PAD")

#-----------------Common Hyperparameter------------------#
args_common = parser.parse_args()
args_common.discount = 0.99 # 1 # cannot be 1
args_common.OU_std = 0.2
args_common.critic_lr = 1e-3
args_common.actor_lr = 1e-3
args_common.replay_mem_capacity = 1000000
args_common.batch_size = 256
args_common.warm_up = 10000
args_common.target_update = 5e-3
args_common.save_interval = 100
args_common.seed = 7
args_common.report_interval = 100

#-----------------HalfCheetah------------------#
args_virtual_taobao_type2 = copy.deepcopy(args_common)
args_virtual_taobao_type2.mdp_num = 100
args_virtual_taobao_type2.template_mdp = 'VirtualTB-v0'
args_virtual_taobao_type2.outlier_rate = 0.1 # 0.2 # 5

args_virtual_taobao_type2.trajectory_len = 10
args_virtual_taobao_type2.learn_step = 10000
args_virtual_taobao_type2.repeat_time = 1
args_virtual_taobao_type2.reward_noise_std = 0. # 0.05
args_virtual_taobao_type2.state_noise_std = 0.
args_virtual_taobao_type2.exp = 'virtual_taobao'
args_virtual_taobao_type2.type = 2










