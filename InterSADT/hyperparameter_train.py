import logging, argparse, time, glob, sys, os
import copy

# from pybullet_anomaly.gym_locomotion_envs import HalfCheetahBulletEnv, HopperBulletEnv, Walker2DBulletEnv, AntBulletEnv

# from pybullet_anomaly.gym_locomotion_envs import HalfCheetahBrother_0_025_BulletEnv
# from pybullet_anomaly.gym_locomotion_envs import HalfCheetahBrother_0_03_BulletEnv
# from pybullet_anomaly.gym_locomotion_envs import HalfCheetahBrother_0_035_BulletEnv
# from pybullet_anomaly.gym_locomotion_envs import HalfCheetahBrother_0_04_BulletEnv
# from pybullet_anomaly.gym_locomotion_envs import HalfCheetahBrother_0_05_BulletEnv
# from pybullet_anomaly.gym_locomotion_envs import HalfCheetahBrother_0_055_BulletEnv
# from pybullet_anomaly.gym_locomotion_envs import HalfCheetahBrother_0_06_BulletEnv
# from pybullet_anomaly.gym_locomotion_envs import HopperBrotherBulletEnv
# from pybullet_anomaly.gym_locomotion_envs import Walker2DBrotherBulletEnv
# from pybullet_anomaly.gym_locomotion_envs import HalfCheetah_Type2Anomaly_BulletEnv

parser = argparse.ArgumentParser("Hyperparameter")
parser.add_argument('--exp', type=str, default='halfcheetah', help='experiment name')
parser.add_argument('--encode_dim', type=int, default=16, help='output dimension of encoder')
# parser.add_argument('--type', type=int, default=1, help='Anomaly type')
# parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
# parser.add_argument('--cpu', type=int, default=0, help='cpu device id')

#-----------------Common Hyperparameter------------------#
args_common = parser.parse_args()
args_common.critic_lr = 1e-3 # 1e-4 #
args_common.actor_lr = 1e-3 # 1e-4 #
args_common.replay_mem_capacity = 1000000
args_common.batch_size = 32
args_common.warm_up = 1000
args_common.target_update = 5e-3
args_common.save_interval = 100
args_common.seed = 7
args_common.report_interval = 100


#-----------------HalfCheetah------------------#
args_half_cheetah_type1 = copy.deepcopy(args_common)
args_half_cheetah_type1.mdp_num = 100
args_half_cheetah_type1.template_mdp = 'HalfCheetahBulletEnv-v0'
args_half_cheetah_type1.inlier_mdps = ['HalfCheetahBulletEnv-v0'] # [HalfCheetahBulletEnv()] # ['HalfCheetahBulletEnv-v0']
args_half_cheetah_type1.outlier_mdps = []

args_half_cheetah_type1.outlier_rate = 0.0 # 5
args_half_cheetah_type1.outlier_num = 0 # 5
# args_half_cheetah_type1.encode_dim = 16 # 8 24 32
args_half_cheetah_type1.encoder_hidden_dim = 32 # 8 # 16
args_half_cheetah_type1.trajectory_len = 10
args_half_cheetah_type1.learn_step = 6000 # 10000
args_half_cheetah_type1.state_noise_std = 0.
args_half_cheetah_type1.type = 1

args_half_cheetah_type1.save_dir = os.path.join(args_half_cheetah_type1.exp + '-type' + str(args_half_cheetah_type1.type),
                                                args_half_cheetah_type1.exp + '-type' + str(args_half_cheetah_type1.type)
                                                + '-{}'.format(time.strftime("%Y%m%d-%H%M%S")))

#-----------------Walker------------------#
args_walker_type1 = copy.deepcopy(args_common)
args_walker_type1.mdp_num = 100
args_walker_type1.template_mdp = 'Walker2DBulletEnv-v0'
args_walker_type1.inlier_mdps = ['Walker2DBulletEnv-v0'] # [Walker2DBulletEnv()]
args_walker_type1.outlier_mdps = []

args_walker_type1.outlier_rate = 0.
args_walker_type1.outlier_num = 0 # 5
# args_walker_type1.encode_dim = 14 # 7 21 28
args_walker_type1.encoder_hidden_dim = 32 # 8 # 16
args_walker_type1.trajectory_len = 10
args_walker_type1.learn_step = 6000 # 10000
args_walker_type1.state_noise_std = 0. # 0.05 #
args_walker_type1.type = 1

args_walker_type1.save_dir = os.path.join(args_walker_type1.exp + '-type' + str(args_walker_type1.type),
                                          args_walker_type1.exp + '-type' + str(args_walker_type1.type)
                                          + '-{}'.format(time.strftime("%Y%m%d-%H%M%S")))
# print(args_half_hopper)

#-----------------Ant------------------#
args_ant_type1 = copy.deepcopy(args_common)
args_ant_type1.mdp_num = 100
args_ant_type1.template_mdp = 'AntBulletEnv-v0'
args_ant_type1.inlier_mdps = ['AntBulletEnv-v0'] # [AntBulletEnv()]
args_ant_type1.outlier_mdps = []

args_ant_type1.outlier_rate = 0.
args_ant_type1.outlier_num = 0 # 5
# args_ant_type1.encode_dim = 18 # 9 27 36
args_ant_type1.encoder_hidden_dim = 64 # 8 # 16
args_ant_type1.trajectory_len = 10
args_ant_type1.learn_step = 6000
args_ant_type1.state_noise_std = 0.
args_ant_type1.type = 1

args_ant_type1.save_dir = os.path.join(args_ant_type1.exp + '-type' + str(args_ant_type1.type),
                                       args_ant_type1.exp + '-type' + str(args_ant_type1.type)
                                          + '-{}'.format(time.strftime("%Y%m%d-%H%M%S")))
# print(args_half_hopper)


# #-----------------Hopper------------------#
# args_hopper_type1 = copy.deepcopy(args_common)
# args_hopper_type1.mdp_num = 100
# args_hopper_type1.template_mdp = 'HopperBulletEnv-v0'
# args_hopper_type1.inlier_mdps = ['HopperBulletEnv-v0'] # [HopperBulletEnv()]
# args_hopper_type1.outlier_mdps = []
#
# args_hopper_type1.outlier_rate = 0.0 # 5
# args_half_cheetah_type1.outlier_num = 0 # 5
# # args_hopper_type1.encode_dim = 9 # 5
# args_hopper_type1.encoder_hidden_dim = 32 #
# args_hopper_type1.trajectory_len = 10
# args_hopper_type1.learn_step = 6000 # 10000
# args_hopper_type1.state_noise_std = 0. # 0.05
# args_hopper_type1.type = 1
#
# args_hopper_type1.save_dir = os.path.join(args_hopper_type1.exp + '-type' + str(args_hopper_type1.type),
#                                           args_hopper_type1.exp + '-type' + str(args_hopper_type1.type)
#                                           + '-{}'.format(time.strftime("%Y%m%d-%H%M%S")))
#

# elif parser.parse_args().exp == 'walker' and parser.parse_args().type == 1:
#     args = copy.deepcopy(args_walker_type1)
# elif parser.parse_args().exp == 'halfcheetah' and parser.parse_args().type == 2:
#     args = copy.deepcopy(args_half_cheetah_type2)
# elif parser.parse_args().exp == 'hopper' and type == 2:
#     args = copy.deepcopy(args_hopper_type2)
# elif parser.parse_args().exp == 'walker' and type == 2:
#     args = copy.deepcopy(args_walker_type2)













