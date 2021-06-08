
import gym
import torch
import math
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import matplotlib.pyplot as plt
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

  def forward(self, src):
    outputs, (hidden, cell) = self.rnn(self.input_layer(src))
    # outputs, (hidden, cell) = self.rnn(src)
    y = self.output_layer(outputs)
    return y, hidden, cell


class Decoder(nn.Module):
  def __init__(self, input_dim, embed_dim, output_dim, hid_dim, n_layers=1, dropout=0):
    super().__init__()

    self.input_layer = layer_init(nn.Linear(input_dim, embed_dim), 1e-3) # nn.Linear(input_dim, embed_dim)
    self.rnn = nn.LSTM(embed_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
    self.output_layer = layer_init(nn.Linear(hid_dim, output_dim), 1e-3) # nn.Linear(hid_dim, output_dim)
    self.dropout = nn.Dropout(dropout)

  def forward(self, src): # , hidden, cell):
    # outputs, (hidden, cell) = self.rnn(src, (hidden, cell))
    # outputs, (hidden, cell) = self.rnn(self.input_layer(src), (hidden, cell))
    outputs, (hidden, cell) = self.rnn(self.input_layer(src)) # , (hidden, cell))
    y = self.output_layer(outputs)
    return y


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
        self.decoder = Decoder(input_dim=output_dim, embed_dim=embed_dim, hid_dim=hid_dim, output_dim=input_dim)

    def forward(self, state_action_time_series):
        encoding, hidden, cell = self.encoder(state_action_time_series)

        # decoding = self.decoder(encoding[:, -1, :].unsqueeze(dim=1)) # , hidden, cell)
        # return decoding[range(decoding.shape[0]-1, -1, -1)]

        # decoding = self.decoder(encoding[range(encoding.shape[0]-1, -1, -1)]) # , hidden, cell)
        # return decoding[range(decoding.shape[0]-1, -1, -1)]

        decoding = self.decoder(encoding) # , hidden, cell)
        return decoding

    def hidden(self, state_action_time_series):
        encoding, hidden, cell = self.encoder(state_action_time_series)
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
        self.critic = Critic(input_dim=state_dim+action_dim, embed_dim=self.encoder_hidden_dim, hid_dim=self.encoder_hidden_dim, output_dim=self.encode_dim) # (input_dim=32, embed_dim=64, hid_dim=16, output_dim=128)
        self.critic_target = Critic(input_dim=state_dim+action_dim, embed_dim=self.encoder_hidden_dim, hid_dim=self.encoder_hidden_dim, output_dim=self.encode_dim) # (256, 256)) # (512, 256, 128))
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.critic_loss_func = nn.MSELoss()

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.replay_mem = ReplayBuffer(capacity=self.replay_mem_capacity)

        # self.state_normalizer = RescaleNormalizer()
        # self.reward_normalizer = RescaleNormalizer()

        self.step = 0

    def explore(self):

        state_action_time_series = torch.zeros((self.trajectory_len, self.state_dim + self.action_dim))
        mdp_index = np.random.randint(0, self.mdps.mdp_num)
        mdp = self.mdps.mdp_buf[mdp_index]
        state_noise = np.random.normal(0, self.state_noise_std, (self.state_dim, ))
        state = mdp.reset() + state_noise # self.state_normalizer(mdp.reset())

        for step in range(self.trajectory_len):
            state_tensor = torch.tensor(state, dtype=torch.float)
            action_tensor = self.actor_target(state_tensor).detach()
            action_array = action_tensor.numpy()
            nxt_state, reward, done, _ = mdp.step(action_array)

            state_noise = np.random.normal(0, self.state_noise_std, (self.state_dim,))
            nxt_state += state_noise # self.state_normalizer(nxt_state)
            state_action_time_series[step] = torch.cat((state_tensor, action_tensor), dim=0)
            state = nxt_state

        self.replay_mem.push(state_action_time_series)

    def critic_learn(self, state_action_time_series):

        # print(state_action_time_series.shape)

        decoding = self.critic(state_action_time_series)
        self.critic_loss = self.critic_loss_func(decoding, state_action_time_series)

        # print(self.step, self.critic_loss)

        self.critic_optim.zero_grad()
        self.critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.1, norm_type=2) # necessary!!!
        self.critic_optim.step()

    def actor_learn(self):

        # print(state_action_time_series.shape)
        state_action_time_series = torch.zeros((self.batch_size, self.trajectory_len, self.state_dim + self.action_dim))

        for batch_index in range(self.batch_size):
            mdp_index = np.random.randint(0, self.mdps.mdp_num)
            mdp = self.mdps.mdp_buf[mdp_index]
            state_noise = np.random.normal(0, self.state_noise_std, (self.state_dim,))
            state = mdp.reset() + state_noise # self.state_normalizer(mdp.reset())
            for step in range(self.trajectory_len):
                state_tensor = torch.tensor(state, dtype=torch.float)
                action_tensor = self.actor(state_tensor)
                action_array = action_tensor.detach().numpy()
                nxt_state, reward, done, _ = mdp.step(action_array)

                state_noise = np.random.normal(0, self.state_noise_std, (self.state_dim,))
                nxt_state += state_noise  # self.state_normalizer(nxt_state)
                state_action_time_series[batch_index, step] = torch.cat((state_tensor, action_tensor), dim=0)
                state = nxt_state

        critic_encoding, critic_hidden, critic_cell = self.critic.hidden(state_action_time_series)
        # self.actor_loss = critic_cell.squeeze(dim=0).var(unbiased=False, dim=0).mean() # shape: (num_layers*num_directions, batch_size, hidden_size)
        self.actor_loss = critic_encoding.var(unbiased=False, dim=0).mean() # shape: (num_layers*num_directions, batch_size, hidden_size)
        # self.actor_loss = critic_encoding[:, -1, :].var(unbiased=False, dim=0).mean() # shape: (num_layers*num_directions, batch_size, hidden_size)
        # self.actor_loss = (critic_encoding[:, -1, :] - critic_encoding[:, -1, :].mean(dim=0)).abs().mean()

        self.actor_optim.zero_grad()
        self.actor_loss.backward()

        # for name, parms in self.actor.named_parameters():
        #     print('name:', name, ' grad_requirs:', parms.requires_grad, ' grad_value:', parms.grad)

        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.1, norm_type=2) # necessary!!!
        self.actor_optim.step()

    def learn(self):

        state_action_time_series_batch, = self.replay_mem.sample(self.batch_size)
        state_action_time_series_batch_tensor = torch.stack(state_action_time_series_batch)
        # print(torch.stack(state_action_time_series_batch).shape)

        self.critic_learn(state_action_time_series_batch_tensor)
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


def soft_update(target_net, eval_net, target_update):
    for target_params, eval_params in zip(target_net.parameters(), eval_net.parameters()):
        target_params.data.copy_(target_params.data * (1.0 - target_update) + eval_params * target_update)


class MDPset:
    def __init__(self, mdp_num, template_mdp, inlier_mdps, outlier_mdps, outlier_rate=None, outlier_num=None):

        self.mdp_num = mdp_num
        self.inlier_mdps = inlier_mdps
        self.outlier_mdps = outlier_mdps
        self.outlier_rate = outlier_rate
        self.outlier_num = outlier_num

        self.template_mdp, self.mdp_buf, self.mdp_label, self.outlier_index, self.inlier_index = self._generate_mdp(template_mdp)
        # self.state_normalizer = RescaleNormalizer()
        # self.mdp_reset()


    def step(self, mdp_index, action):
        mdp = self.mdp_buf[mdp_index]
        # print(action)
        nxt_state, reward, done, info = mdp.step(action)
        nxt_state = self.state_normalizer(nxt_state)
        self.cstate_buf[mdp_index] = nxt_state
        self.mdp_step[mdp_index] += 1

        return nxt_state, reward, done, info

    def mdp_reset(self, mdp_index=None):

        if mdp_index is None:
            self.cstate_buf = []
            self.mdp_step = np.zeros((self.mdp_num,))
            for idx in range(self.mdp_num):
                mdp = self.mdp_buf[idx]
                self.cstate_buf.append(mdp.reset())
            return

        mdp = self.mdp_buf[mdp_index]
        self.cstate_buf[mdp_index] = self.state_normalizer(mdp.reset())
        self.mdp_step[mdp_index] = 0

    def _generate_mdp(self, template_mdp):

        # print(outlier_buf)
        mdp_buf = []
        mdp_label = []
        outlier_index = []
        inlier_index = []

        if self.outlier_num is not None:
            outlier_buf = np.random.choice([mdp_index for mdp_index in range(self.mdp_num)], self.outlier_num, replace=False)

        for mdp_index in range(self.mdp_num):
            if (self.outlier_num is None and np.random.rand() < self.outlier_rate) \
                    or (self.outlier_num is not None and mdp_index in outlier_buf):

                outlier_mdp_sample = np.random.choice(self.outlier_mdps)

                if isinstance(outlier_mdp_sample, str):
                    mdp = gym.make(outlier_mdp_sample)
                else:
                    mdp = outlier_mdp_sample

                mdp_buf.append(mdp)
                mdp_label.append(1)
                outlier_index.append(mdp_index)

            else:
                inlier_mdp_sample = np.random.choice(self.inlier_mdps)

                if isinstance(inlier_mdp_sample, str):
                    mdp = gym.make(inlier_mdp_sample)
                else:
                    mdp = inlier_mdp_sample

                mdp_buf.append(mdp)
                mdp_label.append(0)
                inlier_index.append(mdp_index)

        # for idx in range(self.mdp_num):
        #     mdp_buf.append(gym.make(self.inlier_mdps))
        #     mdp_label.append(0)
        #     inlier_index.append(idx)

        print('Outlier number:', len(outlier_index), 'Outlier index:', outlier_index)
        return gym.make(template_mdp), mdp_buf, mdp_label, outlier_index, inlier_index


def logger_init(args):

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save_dir, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    logger.info("args = %s", args)


def main():

    mdps = MDPset(mdp_num=args.mdp_num,
                  template_mdp=args.template_mdp,
                  inlier_mdps=args.inlier_mdps, # 'HalfCheetahBulletmdp-v0',
                  outlier_mdps=args.outlier_mdps, # 'HalfCheetahBrother_0_075_Bulletmdp-v0',
                  outlier_num=args.outlier_num)
                # outlier_rate=args.outlier_rate)

    # constant_reward = torch.rand((1, args.critic_dim)) * 20 - 10
    agent = Agent(mdps=mdps,
                  batch_size=args.batch_size,
                  actor_lr=args.actor_lr,
                  critic_lr=args.critic_lr,
                  target_update=args.target_update,
                  replay_mem_capacity=args.replay_mem_capacity,
                  trajectory_len=args.trajectory_len,
                  encode_dim=args.encode_dim,
                  encoder_hidden_dim=args.encoder_hidden_dim,
                  state_noise_std = args.state_noise_std
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

if args.exp == 'halfcheetah':
    args = copy.deepcopy(args_half_cheetah_type1)
elif args.exp == 'hopper':
    args = copy.deepcopy(args_hopper_type1)
elif args.exp == 'walker':
    args = copy.deepcopy(args_walker_type1)
elif args.exp == 'ant':
    args = copy.deepcopy(args_ant_type1)

logger = get_logger(tag=(args.exp+'-type'+str(args.type)), log_level=logging.INFO)

if __name__ == '__main__':

    log_utils.create_exp_dir(args.exp + '-type' + str(args.type))
    log_utils.create_exp_dir(args.save_dir, scripts_to_save=glob.glob('*.py'))
    log_utils.create_exp_dir(args.save_dir + '/actor')
    log_utils.create_exp_dir(args.save_dir + '/critic')

    logger_init(args)

    random_seed(args.seed)
    main()
