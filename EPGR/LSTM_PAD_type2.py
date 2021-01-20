
import gym
import torch
import math
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import copy
import log_utils

from deep_rl import *
import logging, argparse, time, glob, sys, os
from hyperparameter_train import *


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, *batch):
        if len(self.buffer) == self.capacity:
            self.buffer.pop(0)
        self.buffer.append(batch)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        # state, action, reward, nxt_state, done = map(np.stack, zip(*batch))
        # print(batch)
        return zip(*batch)

    def __len__(self):
        return len(self.buffer)


class Encoder(nn.Module):
  def __init__(self, input_dim, embed_dim, hid_dim, output_dim, n_layers=1, dropout=0):
    super().__init__()

    self.input_layer = layer_init(nn.Linear(input_dim, embed_dim), 1e-3) # nn.Linear(input_dim, embed_dim)
    self.rnn = nn.LSTM(embed_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
    self.output_layer = layer_init(nn.Linear(hid_dim, output_dim), 1e-3) # nn.Linear(hid_dim, output_dim)
    self.dropout = nn.Dropout(dropout)

    # self.state0_layer1 = layer_init(nn.Linear(state_dim, hid_dim), 1e-3)
    # self.state0_layer2 = layer_init(nn.Linear(state_dim, hid_dim), 1e-3)

  def forward(self, action):

    # hidden_init = torch.zeros((1, 32, 32)) # self.state0_layer1(state0).unsqueeze(dim=0)
    # cell_init = self.state0_layer2(state0).unsqueeze(dim=0)

    # hidden_init = self.state0_layer1(state0).unsqueeze(dim=0)
    # cell_init = torch.zeros((1, 32, 32))

    outputs, (hidden, cell) = self.rnn(self.input_layer(action)) # , (hidden_init, cell_init))
    # outputs, (hidden, cell) = self.rnn(src)
    y = self.output_layer(outputs)
    # y = torch.relu(self.output_layer(outputs))
    return y, hidden, cell


class Actor(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=(128, 64)):
        super().__init__()

        self.body = FCBody(input_size, hidden_units=hidden_size, gate=F.relu)
        self.head = layer_init(nn.Linear(hidden_size[-1], output_size), 1e-3)

    def forward(self, state):
        return torch.tanh(self.head(self.body(state)))


class Critic(nn.Module):
    def __init__(self, input_dim, embed_dim, hid_dim, output_dim):
        super().__init__()
        self.encoder = Encoder(input_dim=input_dim, embed_dim=embed_dim, hid_dim=hid_dim, output_dim=output_dim)

    def forward(self, action_time_series):

        encoding, hidden, cell = self.encoder(action_time_series)
        return encoding, hidden, cell


class Agent(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        state_dim = self.mdps.template_mdp.observation_space.shape[0]
        action_dim = self.mdps.template_mdp.action_space.shape[0]
        self.state_dim = state_dim
        self.action_dim = action_dim
        print(state_dim, action_dim)
        self.actor = Actor(state_dim, action_dim, (400, 300))
        self.actor_target = Actor(state_dim, action_dim, (400, 300))
        self.critic = Critic(input_dim=state_dim+action_dim, embed_dim=self.encoder_hidden_dim, hid_dim=self.encoder_hidden_dim, output_dim=1) # (input_dim=32, embed_dim=64, hid_dim=16, output_dim=128)
        self.critic_target = Critic(input_dim=state_dim+action_dim, embed_dim=self.encoder_hidden_dim, hid_dim=self.encoder_hidden_dim, output_dim=1) # (256, 256)) # (512, 256, 128))
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.critic_loss_func = nn.MSELoss()

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.replay_mem = ReplayBuffer(capacity=self.replay_mem_capacity)
        self.explore_noise = OrnsteinUhlenbeckProcess(size=(action_dim,), std=LinearSchedule(0.1))

        # self.state_normalizer = RescaleNormalizer()
        # self.reward_normalizer = RescaleNormalizer()

        self.step = 0

    def explore(self):

        action_time_series = torch.zeros((self.trajectory_len, self.state_dim + self.action_dim)) #
        reward_time_series = torch.zeros((self.trajectory_len, 1))

        mdp_index = np.random.randint(0, self.mdps.mdp_num)
        mdp = self.mdps.mdp_buf[mdp_index]
        state = mdp.reset()
        self.explore_noise.reset_states()
        for step in range(self.trajectory_len):
            state_tensor = torch.tensor(state, dtype=torch.float)
            action_tensor = self.actor_target(state_tensor).detach() + torch.from_numpy(self.explore_noise.sample())
            action_array = action_tensor.numpy()
            nxt_state, reward, done, _ = mdp.step(action_array)

            reward_tensor = torch.tensor(reward, dtype=torch.float).unsqueeze(dim=0)

            action_time_series[step] = torch.cat((state_tensor, action_tensor), dim=0) # action_tensor #
            reward_time_series[step] = reward_tensor

            state = nxt_state

        self.replay_mem.push(action_time_series, reward_time_series)

    def critic_learn(self, action_time_series, reward_time_series):

        # print(state_action_time_series.shape)

        reward_predict, _, _ = self.critic(action_time_series)
        self.critic_loss = self.critic_loss_func(reward_predict, reward_time_series)

        # print(self.step, self.critic_loss)

        self.critic_optim.zero_grad()
        self.critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.1, norm_type=2) # necessary!!!
        self.critic_optim.step()

    def actor_learn(self):

        # print(state_action_time_series.shape)
        action_time_series = torch.zeros((self.batch_size, self.trajectory_len, self.state_dim + self.action_dim)) # self.state_dim + self.action_dim))
        # reward_time_series = torch.zeros((self.batch_size, self.trajectory_len, 1)) # self.state_dim +

        for batch_index in range(self.batch_size):
            mdp_index = np.random.randint(0, self.mdps.mdp_num)
            mdp = self.mdps.mdp_buf[mdp_index]
            state = mdp.reset()
            for step in range(self.trajectory_len):
                state_tensor = torch.tensor(state, dtype=torch.float)
                action_tensor = self.actor(state_tensor)
                action_array = action_tensor.detach().numpy()
                nxt_state, reward, done, _ = mdp.step(action_array)
                # reward_tensor = torch.tensor(reward, dtype=torch.float).unsqueeze(dim=0)

                action_time_series[batch_index, step] = torch.cat((state_tensor, action_tensor), dim=0) # action_tensor #
                # reward_time_series[batch_index, step] = reward_tensor # state_tensor,
                state = nxt_state

        reward_predict, _, _ = self.critic(action_time_series)

        self.actor_loss = reward_predict.var(unbiased=False, dim=0).mean() # shape: (num_layers*num_directions, batch_size, hidden_size)
        self.actor_optim.zero_grad()
        self.actor_loss.backward()

        # for name, parms in self.actor.named_parameters():
        #     print('name:', name, ' grad_requirs:', parms.requires_grad, ' grad_value:', parms.grad)

        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.1, norm_type=2) # necessary!!!
        self.actor_optim.step()

    def learn(self):

        action_time_series_batch, reward_time_series_batch = self.replay_mem.sample(self.batch_size)

        action_time_series_batch_tensor = torch.stack(action_time_series_batch)
        reward_time_series_batch_tensor = torch.stack(reward_time_series_batch)
        # print(torch.stack(state_action_time_series_batch).shape)

        self.critic_learn(action_time_series_batch_tensor, reward_time_series_batch_tensor)
        self.actor_learn()

        soft_update(self.critic_target, self.critic, self.target_update)
        soft_update(self.actor_target, self.actor, self.target_update)

        self.step += 1

    def save_actor(self, folder, step):
        torch.save(self.actor, folder + '/actor_' + str(step) + '.pkl')

    def save_critic(self, folder, step):
        torch.save(self.critic, folder + '/critic_' + str(step) + '.pkl')

    def load_actor(self, fullname):
        self.actor = torch.load(fullname)

    def load_critic(self, fullname):
        self.critic = torch.load(fullname)


class Virtual_User:

    def __init__(self, template_mdp, user_characteristic, anomaly_prob=0., total_time=10):
        self.template_mdp = gym.make(template_mdp)
        self.user_characteristic = user_characteristic
        self.anomaly_prob = anomaly_prob
        self.anomaly_time_number = int(np.floor(np.abs(self.anomaly_prob)*total_time))
        self.time_anomaly = None
        self.total_time = 10

    def reset(self):
        state = self.template_mdp.setuser(self.user_characteristic)
        self.time_anomaly = np.random.choice([time for time in range(self.total_time)], size=self.anomaly_time_number)
        self.time_step = 0
        return state

    def step(self, recommendations):

        nxt_state, reward, done, info = self.template_mdp.step(recommendations)
        if self.time_step in self.time_anomaly:
            if self.anomaly_prob > 0: # and reward > 0:
                reward += 1.
            elif self.anomaly_prob < 0 and reward > 0:
                reward -= 1.

        self.time_step += 1
        return nxt_state, reward, done, info

        # np_rand = np.random.rand(1)
        # anomaly_flag = 0
        # if self.anomaly_prob > 0 and np_rand < self.anomaly_prob:
        #     reward += 1.
        #     anomaly_flag = 1
        # elif self.anomaly_prob < 0 and np_rand < -self.anomaly_prob:
        #     anomaly_flag = 1
        #     reward = reward-1 if reward > 0 else reward
        # self.time_series_anomaly.append(anomaly_flag)
        # return nxt_state, reward, done, self.time_series_anomaly


class Virtual_Userset:
    def __init__(self, template_mdp, user_num, user_characteristic_buf, anomaly_num=0, anomaly_prob=0.):
        self.template_mdp = gym.make(template_mdp)
        self.user_num = user_num
        self.user_characteristic_buf = random.sample(user_characteristic_buf, user_num)
        self.user_index = [user_index for user_index in range(user_num)]
        self.anomaly_index = np.random.choice(self.user_index, size=anomaly_num)
        self.nomral_index = list(set(self.user_index) - set(self.anomaly_index))

        self.anomaly_prob = np.array([np.random.choice(anomaly_prob) if user_index in self.anomaly_index else 0. for user_index in range(user_num)])
        self.user_label = [1 if user_index in self.anomaly_index else 0 for user_index in range(user_num)]
        self.user_mdp = [Virtual_User(template_mdp=template_mdp,
                                      user_characteristic=self.user_characteristic_buf[user_index],
                                      anomaly_prob=self.anomaly_prob[user_index]) for user_index in range(user_num)]


    def random_user_select(self):
        return np.random.choice(self.user_mdp, size=1)[0]

    def user_select(self, user_index):
        return self.user_mdp[user_index]


class MDPset:
    def __init__(self, mdp_num, template_mdp):

        self.mdp_num = mdp_num
        self.template_mdp = gym.make(template_mdp)
        self.mdp_buf = [gym.make(template_mdp) for mdp_index in range(self.mdp_num)]


def soft_update(target_net, eval_net, target_update):
    for target_params, eval_params in zip(target_net.parameters(), eval_net.parameters()):
        target_params.data.copy_(target_params.data * (1.0 - target_update) + eval_params * target_update)


def logger_init(args):

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save_dir, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    logger.info("args = %s", args)


def main():

    # mdp = gym.make('VirtualTB-v0')
    mdps = MDPset(mdp_num=args.mdp_num,
                  template_mdp=args.template_mdp)

    # constant_reward = torch.rand((1, args.critic_dim)) * 20 - 10
    agent = Agent(mdps=mdps,
                  batch_size=args.batch_size,
                  actor_lr=args.actor_lr,
                  critic_lr=args.critic_lr,
                  target_update=args.target_update,
                  replay_mem_capacity=args.replay_mem_capacity,
                  trajectory_len=args.trajectory_len,
                  encoder_hidden_dim=args.encoder_hidden_dim,
                  trajectory_length=args.trajectory_len,
                  )

    # torch.save(constant_reward, args.save_dir + '/constant_reward.pkl')

    for step in range(args.learn_step):
        agent.explore()

        if agent.replay_mem.__len__() >= args.warm_up:

            if step % args.save_interval == args.save_interval - 1:
                learn_step = step - args.warm_up + 1
                agent.save_actor(args.save_dir + '/actor', learn_step)
                agent.save_critic(args.save_dir + '/critic', learn_step)

            agent.learn()

            if step % args.report_interval == args.report_interval - 1:
                learn_step = step - args.warm_up + 1
                logger.info('step = %d, critic loss = %f, actor loss = %f', learn_step, agent.critic_loss, agent.actor_loss)



    return

args = parser.parse_args()

if args.exp == 'virtual_taobao':
    args = copy.deepcopy(args_virtual_taobao_type2)

logger = get_logger(tag=(args.exp+'-type'+str(args.type)), log_level=logging.INFO)

if __name__ == '__main__':

    log_utils.create_exp_dir(args.exp + '-type' + str(args.type))
    log_utils.create_exp_dir(args.save_dir, scripts_to_save=glob.glob('*.py'))
    log_utils.create_exp_dir(args.save_dir + '/actor')
    log_utils.create_exp_dir(args.save_dir + '/critic')

    logger_init(args)

    random_seed(args.seed)
    main()
