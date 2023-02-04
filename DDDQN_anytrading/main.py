import tensorflow as tf 
import numpy as np 
import gym
import gym_anytrading
from gym_anytrading.envs import TradingEnv, ForexEnv, StocksEnv, Actions, Positions 
from gym_anytrading.datasets import FOREX_EURUSD_1H_ASK, STOCKS_GOOGL
from agent import Agent

from tensorflow.keras.models import load_model

import json
import datetime as dt
import pandas as pd

#from StockTradingEnv import StockTradingEnv

tf.config.list_physical_devices('GPU') 

#df = pd.read_csv('./data/AAPL.csv')
#df = df.sort_values('Date')

env = gym.make('stocks-v0')
low = env.observation_space.low
high = env.observation_space.high

#asize = 3
agentoo7 = Agent(gamma=0.99, epsilon=1, lr=0.0001,
                  input_dims=(env.observation_space.shape),
                  n_actions=env.action_space.n, mem_size=50000, eps_min=0.1,
                  batch_size=64, replace=1000, eps_dec=1e-5,
                  chkpt_dir='models/', algo='DQNAgent',
                  env_name='StockTradingEnv')
steps = 400
for s in range(steps):
  done = False
  state = env.reset()
  total_reward = 0
  while not done:
    #env.render()
    action = agentoo7.choose_action(state)
    next_state, reward, done, _ = env.step(action)
    agentoo7.store_transition(state, action, reward, next_state, done)
    agentoo7.learn()
    state = next_state
    total_reward += reward
    
    if done:
      print("total reward after {} episode is {} and epsilon is {}".format(s, total_reward, agentoo7.epsilon))