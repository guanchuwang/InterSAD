import gym
import numpy as np
import torch
import random
import pybullet_envs
import time
from sklearn.metrics import roc_curve, auc
from sklearn import datasets, manifold

import matplotlib.pyplot as plt
import copy
import pickle
import os

import virtualTB

from deep_rl import *

from LSTM_PAD_type2 import MDPset
from hyperparameter_test import *

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM


mdp = gym.make('VirtualTB-v0')
state_dim = mdp.observation_space.shape[0]
action_dim = mdp.action_space.shape[0]
mdp.seed(0)

user_num = 10000

user_characteristic_buf = []
for user_index in range(user_num):
    mdp.reset()
    user_characteristic = mdp.getuser()
    user_characteristic_buf.append(user_characteristic)

with open('./user_characteristic_buf_10000_seed0.pkl', 'wb') as output:
    pickle.dump(user_characteristic_buf, output)

