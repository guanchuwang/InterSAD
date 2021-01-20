import logging, argparse, time, glob, sys, os
import copy

# from pybullet_anomaly.gym_locomotion_envs import HalfCheetahBulletEnv, HopperBulletEnv, Walker2DBulletEnv, AntBulletEnv

from pybullet_anomaly.gym_locomotion_envs import HalfCheetahBrother_0_025_BulletEnv
from pybullet_anomaly.gym_locomotion_envs import HalfCheetahBrother_0_03_BulletEnv
from pybullet_anomaly.gym_locomotion_envs import HalfCheetahBrother_0_035_BulletEnv
from pybullet_anomaly.gym_locomotion_envs import HalfCheetahBrother_0_04_BulletEnv
from pybullet_anomaly.gym_locomotion_envs import HalfCheetahBrother_0_05_BulletEnv
from pybullet_anomaly.gym_locomotion_envs import HalfCheetahBrother_0_055_BulletEnv
from pybullet_anomaly.gym_locomotion_envs import HalfCheetahBrother_0_06_BulletEnv

from pybullet_anomaly.gym_locomotion_envs import HopperBrotherBulletEnv
from pybullet_anomaly.gym_locomotion_envs import Walker2DBrotherBulletEnv, Walker2D_Type1_Anomaly_0_005_BulletEnv

from pybullet_anomaly.gym_locomotion_envs import Hopper_Type1_Anomaly_0_005_BulletEnv

from pybullet_anomaly.gym_locomotion_envs import Ant_Type1_Anomaly_0_008_BulletEnv

parser = argparse.ArgumentParser("EPG")

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
args_half_cheetah_type1 = copy.deepcopy(args_common)
args_half_cheetah_type1.mdp_num = 100
args_half_cheetah_type1.template_mdp = 'HalfCheetahBulletEnv-v0'
args_half_cheetah_type1.inlier_mdps = ['HalfCheetahBulletEnv-v0'] # ['HalfCheetahBulletEnv-v0']
args_half_cheetah_type1.outlier_mdps = [
                                        # HalfCheetahBrother_0_03_BulletEnv(),
                                        # HalfCheetahBrother_0_035_BulletEnv(),
                                        # HalfCheetahBrother_0_04_BulletEnv(),
                                        HalfCheetahBrother_0_05_BulletEnv(),
                                        # HalfCheetahBrother_0_055_BulletEnv(),
                                        # HalfCheetahBrother_0_06_BulletEnv(),
                                        ]
                                 # 'HalfCheetahBrother_0_03_BulletEnv-v0',
                                 # 'HalfCheetahBrother_0_035_BulletEnv-v0',
                                 # 'HalfCheetahBrother_0_04_BulletEnv-v0',
                                 # 'HalfCheetahBrother_0_05_BulletEnv-v0',
                                 # 'HalfCheetahBrother_0_055_BulletEnv-v0',
                                 # 'HalfCheetahBrother_0_06_BulletEnv-v0'

args_half_cheetah_type1.outlier_num = 10 # 5

args_half_cheetah_type1.encode_dim = 16
args_half_cheetah_type1.encoder_hidden_dim = 32 # 8 # 16
args_half_cheetah_type1.trajectory_len = 10 # 10
args_half_cheetah_type1.learn_step = 100000
args_half_cheetah_type1.repeat_time = 1
args_half_cheetah_type1.state_noise_std = 0.05

args_half_cheetah_type1.exp = 'halfcheetah'
args_half_cheetah_type1.type = 1

#-----------------Hopper------------------#
args_hopper_type1 = copy.deepcopy(args_common)
args_hopper_type1.mdp_num = 100
args_hopper_type1.template_mdp = 'HopperBulletEnv-v0'
args_hopper_type1.inlier_mdps = ['HopperBulletEnv-v0'] # [HopperBulletEnv()]
args_hopper_type1.outlier_mdps = [Hopper_Type1_Anomaly_0_005_BulletEnv()]
args_hopper_type1.outlier_rate = 0.1

args_hopper_type1.encode_dim = 5
args_hopper_type1.encoder_hidden_dim = 32 # 8 # 16
args_hopper_type1.trajectory_len = 10 # 10
args_hopper_type1.learn_step = 100000
args_hopper_type1.repeat_time = 1
args_hopper_type1.state_noise_std = 0.05 # 0.1 #

args_hopper_type1.exp = 'hopper'
args_hopper_type1.type = 1

#-----------------Walker------------------#
args_walker_type1 = copy.deepcopy(args_common)
args_walker_type1.mdp_num = 100
args_walker_type1.template_mdp = 'Walker2DBulletEnv-v0'
args_walker_type1.inlier_mdps = ['Walker2DBulletEnv-v0'] # [Walker2DBulletEnv()]
args_walker_type1.outlier_mdps = [Walker2D_Type1_Anomaly_0_005_BulletEnv()]

args_walker_type1.outlier_rate = 0.1
args_walker_type1.encode_dim = 5
args_walker_type1.encoder_hidden_dim = 32 # 8 # 16
args_walker_type1.trajectory_len = 10 # 10
args_walker_type1.learn_step = 100000
args_walker_type1.repeat_time = 1
args_walker_type1.state_noise_std = 0.05

args_walker_type1.exp = 'walker'
args_walker_type1.type = 1

# print(args_half_hopper)

#-----------------Ant------------------#
args_ant_type1 = copy.deepcopy(args_common)
args_ant_type1.mdp_num = 100
args_ant_type1.template_mdp = 'AntBulletEnv-v0'
args_ant_type1.inlier_mdps = ['AntBulletEnv-v0'] # [AntBulletEnv()]
args_ant_type1.outlier_mdps = [Ant_Type1_Anomaly_0_008_BulletEnv()]
args_ant_type1.outlier_rate = 0.1

args_ant_type1.encode_dim = 9
args_ant_type1.encoder_hidden_dim = 32 # 8 # 16
args_ant_type1.trajectory_len = 10 # 10
args_ant_type1.learn_step = 100000
args_ant_type1.repeat_time = 1
args_ant_type1.state_noise_std = 0.05

args_ant_type1.exp = 'ant'
args_ant_type1.type = 1

#-----------------HalfCheetah------------------#
args_half_cheetah_type2 = copy.deepcopy(args_common)
args_half_cheetah_type2.mdp_num = 100
args_half_cheetah_type2.template_mdp = 'HalfCheetahBulletEnv-v0'
args_half_cheetah_type2.outlier_rate = 0.1 # 0.2 # 5

args_half_cheetah_type2.trajectory_len = 10
args_half_cheetah_type2.learn_step = 10000
args_half_cheetah_type2.repeat_time = 1
args_half_cheetah_type2.reward_noise_std = 0.005 # 0.05
args_half_cheetah_type2.state_noise_std = 0.05
# args_half_cheetah_type2.reward_bias = 0. # 0.05










