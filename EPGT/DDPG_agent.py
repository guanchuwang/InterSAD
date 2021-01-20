
import gym
import torch
import math
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from deep_rl import *
from hyperparameter_train import *

import logging, argparse, time, glob, sys, os
import log_utils

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

class Actor(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=(128, 64)):
        super().__init__()
        self.body = FCBody(input_size, hidden_units=hidden_size, gate=F.relu)
        self.head = layer_init(nn.Linear(hidden_size[-1], output_size), 1e-3)

    def forward(self, state):
        return torch.tanh(self.head(self.body(state)))

class Critic(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=(128, 64)):
        super().__init__()
        self.body = FCBody(input_size, hidden_units=hidden_size, gate=F.relu)
        self.head = layer_init(nn.Linear(hidden_size[-1], output_size), 1e-3)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        return self.head(self.body(x))

class Agent(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor = Actor(state_dim, action_dim, (400, 300))
        self.actor_target = Actor(state_dim, action_dim, (400, 300))
        self.critic = Critic(state_dim + action_dim, 1, (400, 300))
        self.critic_target = Critic(state_dim + action_dim, 1, (400, 300))
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.explore_noise = OrnsteinUhlenbeckProcess(size=(action_dim,), std=LinearSchedule(0.2))
        self.state_normalizer = RescaleNormalizer()
        self.reward_normalizer = RescaleNormalizer()

        # state_noise = np.random.normal(0, self.state_noise_std, (self.state_dim, ))
        self.state = self.state_normalizer(self.env.reset())
        self.creward = 0.
        self.explore_noise.reset_states()
        self.step = 0

        self.interact_step = 0

    def explore(self, replay_mem):

        action = self.act(self.state) + self.explore_noise.sample()

        nxt_state, reward, done, _ = self.env.step(action)
        # state_noise = np.random.normal(0, self.state_noise_std, (self.state_dim, ))
        nxt_state = self.state_normalizer(nxt_state)
        reward = self.reward_normalizer(reward)

        state_tensor = torch.tensor(self.state, dtype=torch.float)
        action_tensor = torch.tensor(action, dtype=torch.float)
        reward_tensor = torch.tensor(reward, dtype=torch.float)
        nxt_state_tensor = torch.tensor(nxt_state, dtype=torch.float)
        done_tensor = torch.tensor(done, dtype=torch.int)

        replay_mem.push(state_tensor, action_tensor, reward_tensor, nxt_state_tensor, done_tensor)

        self.step += 1
        self.interact_step += 1

        if self.interact_step >= self.trajectory_len:
            self.state = self.state_normalizer(self.env.reset())
            self.explore_noise.reset_states()
            self.interact_step = 0
            self.creward = 0.
            logger.info('Step=%d, reward=%f', self.step, self.creward)

        else:
            self.state = nxt_state
            self.creward += reward

    def act(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        action = self.actor(state_tensor).squeeze(0).detach().numpy()
        return action

    def critic_learn(self, state, action, reward, nxt_state, mask):
        nxt_action = self.actor_target(nxt_state).detach()
        Q_hat = reward + self.discount * mask * self.critic_target(nxt_state, nxt_action).detach()
        Q_predict = self.critic(state, action)
        # print(Q_hat, Q_predict)
        loss_func = nn.MSELoss()
        Q_loss = loss_func(Q_predict, Q_hat)
        self.critic_optim.zero_grad()
        # nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.1, norm_type=2)
        Q_loss.backward()
        self.critic_optim.step()

    def actor_learn(self, state):
        action = self.actor(state) ### necessary!
        pi_loss = -self.critic_target(state, action).mean()
        self.actor_optim.zero_grad()
        # nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.1, norm_type=2)
        pi_loss.backward()
        self.actor_optim.step()

    def soft_update(self, target_net, eval_net):
        for target_params, eval_params in zip(target_net.parameters(), eval_net.parameters()):
            target_params.data.copy_(target_params.data * (1.0 - self.target_update) + eval_params * self.target_update)

    def learn(self, replay_mem):

        state, action, reward, nxt_state, terminals = replay_mem.sample(self.batch_size)
        state_tensor = torch.stack(state)
        action_tensor = torch.stack(action)
        reward_tensor = torch.stack(reward).unsqueeze(1)
        nxt_state_tensor = torch.stack(nxt_state)
        mask_tensor = (1 - torch.stack(terminals)).unsqueeze(-1)

        self.critic_learn(state_tensor, action_tensor, reward_tensor, nxt_state_tensor, mask_tensor)
        self.actor_learn(state_tensor)
        self.soft_update(self.critic_target, self.critic)
        self.soft_update(self.actor_target, self.actor)

    def save_actor(self, folder, step):
        torch.save(self.actor, folder + '/actor_' + str(step) + '.pkl')

    def save_critic(self, folder, step):
        torch.save(self.critic, folder + '/critic_' + str(step) + '.pkl')


def main():

    env = args.inlier_mdps[0]
    args.env = env

    agent = Agent(**vars(args))
    replay_mem = ReplayBuffer(capacity=1000000)

    # creward = 0.
    for step in range(args.max_step):
        agent.explore(replay_mem)
        if replay_mem.__len__() >= args.warm_up:
            agent.learn(replay_mem)
        # creward += reward

            if step % args.save_interval == args.save_interval - 1:
                agent.save_actor(args.save_dir + '/actor', step - args.warm_up + 1)
                agent.save_critic(args.save_dir + '/critic', step - args.warm_up + 1)

    return


def logger_init(args):

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save_dir, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    logger.info("args = %s", args)


args = parser.parse_args()

if args.exp == 'halfcheetah':
    args = copy.deepcopy(args_half_cheetah_type1)
elif args.exp == 'hopper':
    args = copy.deepcopy(args_hopper_type1)
elif args.exp == 'walker':
    args = copy.deepcopy(args_walker_type1)
elif args.exp == 'ant':
    args = copy.deepcopy(args_ant_type1)


args.critic_lr = 1e-5
args.actor_lr = 1e-5
args.discount = 0.9
args.batch_size = 32
args.max_step = args.warm_up + 5000

logger = get_logger(tag=('DDPG-' + args.exp+'-type'+str(args.type)), log_level=logging.INFO)

if __name__ == '__main__':

    args.save_dir = 'DDPG-' + args.save_dir

    log_utils.create_exp_dir('DDPG-' + args.exp + '-type' + str(args.type))
    log_utils.create_exp_dir(args.save_dir, scripts_to_save=glob.glob('*.py'))
    log_utils.create_exp_dir(args.save_dir + '/actor')
    log_utils.create_exp_dir(args.save_dir + '/critic')

    logger_init(args)

    random_seed(args.seed)
    main()
