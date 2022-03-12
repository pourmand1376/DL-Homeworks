#!/usr/bin/env python
# coding: utf-8

# # CE-40719: Deep Learning
# ## HW6 - Deep Reinforcement Learning
# 
# (40 points)
# 
# #### Name: 
# #### Student No.: 
# 

# In this assignment, we are going to design an agent to play Atari game. 
# we will use a wrapper to change MDP problem to POMDP, as a result, we are able to investigate the efficiency of using memory to solve problems in a partial observability setting.
# For this reseaon, we use a Deep Q-Network model as memory less architecture and a [DRQN](https://arxiv.org/abs/1507.06527) as a memoryful agent to play game Pong(`PongNoFrameskip-v4` environment of [gym](https://gym.openai.com/) library).
# In this environment, the observation is an RGB image of the screen, which is an array of shape (210, 160, 3). And action space is 6.
# We will train the model for 200,000 steps and should take approximately 2 hour.
# 
# At the end, you should be able to conclude effectiveness of recurrent memory to cancel out noisy observation.
# 

# ![Pong](https://cdn-images-1.medium.com/max/800/1*UHYJE7lF8IDZS_U5SsAFUQ.gif)

#  ### 1. Setup

# if you use google colab to train your network, mount to google drive would be necessary 

# In[ ]:


from google.colab import drive
drive._mount('/content/drive')


# First, we need to install `stable-baselines`. This library  is a set of improved implementations of Reinforcement Learning (RL) algorithms based on OpenAI Baselines. We will use some of **wrappers** of this library. Wrappers will allow us to add functionality to environments, such as modifying observations and rewards to be fed to our agent. It is common in reinforcement learning to preprocess observations in order to make them more easy to learn from. 
# 
# - For linux based Operating Systems or google colab run cell below:

# In[ ]:


get_ipython().run_cell_magic('shell', '', '\nsudo apt-get update && sudo apt-get install cmake libopenmpi-dev zlib1g-dev\n\npip install stable-baselines[mpi]==2.8.0')


# - For Windows: 
#     - First install [MPI for Windows](https://www.microsoft.com/en-us/download/details.aspx?id=57467) (you need to download and install `msmpisetup.exe`)
#     - Then run this command in Prompt: `pip install stable-baselines[mpi]==2.8.0`

# install ROMs which needed for creating atari env

# In[ ]:


import urllib.request
urllib.request.urlretrieve('http://www.atarimania.com/roms/Roms.rar','Roms.rar')
get_ipython().system('pip install unrar')
get_ipython().system('unrar x Roms.rar')
get_ipython().system('mkdir rars')
get_ipython().system('mv HC\\ ROMS.zip   rars')
get_ipython().system('mv ROMS.zip  rars')
get_ipython().system('python -m atari_py.import_roms rars')


# ### 2. Import Libraries:

# In[ ]:


import random, os.path, math, glob, csv, os
import numpy as np

from timeit import default_timer as timer
from datetime import timedelta

import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.display import clear_output
from plot import plot_all_data 

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

# ######################## #
# USE ONLY IN GOOGLE COLAB #
get_ipython().run_line_magic('tensorflow_version', '1.x')
# ######################## #

import gym
from gym.spaces.box import Box
from stable_baselines import bench
from stable_baselines.common.atari_wrappers import make_atari, wrap_deepmind


# ### 2. Hyperparameters

# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Epsilon variables for epsilon-greedy:
epsilon_start    = 1.0
epsilon_final    = 0.01
epsilon_decay    = 30000
epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay) 

# Misc agent variables
GAMMA = 0.99 
LR    = 1e-4 

# Memory
TARGET_NET_UPDATE_FREQ = 1000 
EXP_REPLAY_SIZE        = 100000 
BATCH_SIZE             = 32 

# Learning control variables
LEARN_START = 10000
MAX_FRAMES  = 200000 # Probably takes about an hour training. You can increase it if you have time!
UPDATE_FREQ = 1 

# Data logging parameters
ACTION_SELECTION_COUNT_FREQUENCY = 1000 

#DRQN Parameters
SEQUENCE_LENGTH = 8


# ### 3. Wrapper

# In[ ]:


class WrapPOMDP(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(WrapPOMDP, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        # this method change MDP into POMDP
        pomdp = np.random.uniform()
        if pomdp >= 0.5:
          return observation.transpose(2, 0, 1)
        else:
          return observation.transpose(2, 0, 1) * 0.0


# ### 4. Abstract Agent Class (3 Points)

# In[ ]:


class BaseAgent():
    def __init__(self, model, target_model, log_dir, env):
        self.device = device
        self.gamma = GAMMA
        self.lr = LR
        self.target_net_update_freq = TARGET_NET_UPDATE_FREQ
        self.experience_replay_size = EXP_REPLAY_SIZE
        self.batch_size = BATCH_SIZE
        self.learn_start = LEARN_START
        self.update_freq = UPDATE_FREQ
        self.update_count = 0
        self.nstep_buffer = []
        self.rewards = []
        self.model = model
        self.target_model = target_model
        #################################################################################
        #                          COMPLETE THE FOLLOWING SECTION                       #
        #################################################################################
        # compelete this method
        # * save log_dir
        # * save env
        # * make a list of action selections
        # * using load_state_dict, share learnable parameters (i.e. weights and biases) of
        #   self.model with self.target_model 
        # * move both model to correct device
        # * use Adam optimizer
        # * set both model to train mode
        #################################################################################
        self.log_dir = None
        self.env = None
        self.action_selections = None
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = None
        #################################################################################
        #                                   THE END                                     #
        #################################################################################

    def update_target_model(self):
        # update target model:
        self.update_count += 1
        self.update_count = self.update_count % self.target_net_update_freq
        if self.update_count == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def save_sigma_param_magnitudes(self, tstep):
        with torch.no_grad():
            sum_, count = 0.0, 0.0
            for name, param in self.model.named_parameters():
                if param.requires_grad and 'sigma' in name:
                    sum_ += torch.sum(param.abs()).item()
                    count += np.prod(param.shape)
            if count > 0:
                with open(os.path.join(self.log_dir, 'sig_param_mag.csv'), 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow((tstep, sum_ / count))

    def save_td(self, td, tstep):
        with open(os.path.join(self.log_dir, 'td.csv'), 'a') as f:
            writer = csv.writer(f)
            writer.writerow((tstep, td))

    def save_reward(self, reward):
        self.rewards.append(reward)

    def save_action(self, action, tstep):
        self.action_selections[int(action)] += 1.0 / self.action_log_frequency
        if (tstep + 1) % self.action_log_frequency == 0:
            with open(os.path.join(self.log_dir, 'action_log.csv'), 'a') as f:
                writer = csv.writer(f)
                writer.writerow(list([tstep] + self.action_selections))
            self.action_selections = [0 for _ in range(len(self.action_selections))]

    def get_max_next_state_action(self, next_states):
        return self.target_model(next_states).max(dim=1)[1].view(-1, 1)


# # 1. Simple DQN

# We will implement a DQN model with experience replay. We implement a class for experience replay  `ExperienceReplayMemory`, for extarct features of observed picture of game we implement a class `DQN` that uses a CNN and inheritance from `nn.Module`. We implemented a wrapper class `WrapPyTorch` that you will use it in training loop.

# ## 1.1. Experience Replay (3 Points)

# In[ ]:


#################################################################################
#                          COMPLETE THE FOLLOWING SECTION                       #
#################################################################################
# complete push and sample methods.
#################################################################################
class ExperienceReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        pass

    def sample(self, batch_size):
        return None

    def __len__(self):
        return len(self.memory)
#################################################################################
#                                   THE END                                     #
#################################################################################  


# ## 1.2. Network Declaration (4 Points)

# In[ ]:


class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions
        #################################################################################
        #                          COMPLETE THE FOLLOWING SECTION                       #
        #################################################################################
        # Initialize CNN Model :
        # conv1: out_channels:32, kernel_size=8, stride=4
        # conv2: out_channels:64, kernel_size=4, stride=2
        # conv3: out_channels:64, kernel_size=3, stride=1
        # fc1(512)
        # fc2(512)
        #################################################################################
        pass
        #################################################################################
        #                                   THE END                                     #
        #################################################################################

    def forward(self, x):
        #################################################################################
        #                          COMPLETE THE FOLLOWING SECTION                       #
        #################################################################################
        # complete forward pass using initialized CNN Model. use Relu activation function 
        # for conv1, conv2, conv3, and fc1.  
        #################################################################################
        pass

        return x
        #################################################################################
        #                                   THE END                                     #
        #################################################################################   


#  ## 1.3. Agent (6 Points)

# In[ ]:


class Model():
    def __init__(self, env=None, log_dir):
        self.device = device

        self.gamma = GAMMA 
        self.lr = LR
        self.target_net_update_freq = TARGET_NET_UPDATE_FREQ
        self.experience_replay_size = EXP_REPLAY_SIZE
        self.batch_size = BATCH_SIZE
        self.learn_start = LEARN_START
        self.update_freq = UPDATE_FREQ
        self.log_dir = log_dir
        self.rewards = []
        self.action_log_frequency = ACTION_SELECTION_COUNT_FREQUENCY
        #################################################################################
        #                          COMPLETE THE FOLLOWING SECTION                       #
        #################################################################################
        # do followings line by line:
        # * make a list of action selections
        # * use shape of observation space to save number of features
        # * save naumber of actions
        # * use DQN class to declare model and target model (the game is 2-player game, so
        #   we declare 2 model)
        # * using load_state_dict, share learnable parameters (i.e. weights and biases) of
        #   self.model with self.target_model 
        # * use Adam optimizer
        #################################################################################
        self.num_feats = None
        self.num_actions = None
        self.model = None
        self.target_model = None
        super(Model, self).__init__()
        #################################################################################
        #                                   THE END                                     #
        #################################################################################
        
        self.memory = ExperienceReplayMemory(self.experience_replay_size)
        

    def prep_minibatch(self):
      
        #################################################################################
        #                          COMPLETE THE FOLLOWING SECTION                       #
        #################################################################################
        # random transition batch is taken from experience replay memory
        # do followings line by line:
        # * sample from self.memory with batch size, and save result in transitions
        # * use transitions to save batch_state, batch_action, batch_reward, 
        #   batch_next_state as tensors
        # * save non_final_mask,  non_final_next_states as tensors, note that sometimes 
        #   all next states are false
        #################################################################################
        transitions = None
        
        batch_state, batch_action, batch_reward, batch_next_state = None

  

        batch_state = None
        batch_action = None
        batch_reward = None
        
        non_final_mask = None

        try: 
            non_final_next_states = None
            empty_next_state_values = False
        except:
            non_final_next_states = None
            empty_next_state_values = True
        #################################################################################
        #                                   THE END                                     #
        #################################################################################
        return batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values

    def compute_loss(self, batch_vars):
        batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values = batch_vars

        #################################################################################
        #                          COMPLETE THE FOLLOWING SECTION                       #
        #################################################################################
        # implement calculation of loss (you should use "with torch.no_grad():" for target_model)
        #################################################################################

        loss = None
        #################################################################################
        #                                   THE END                                     #
        #################################################################################
        return loss

    def update(self, s, a, r, s_, frame=0):
        #################################################################################
        #                          COMPLETE THE FOLLOWING SECTION                       #
        #################################################################################
        # implement update method to optimize model 
        # * push state, action, reward and new state to memoty
        # * note that if frame is lower than self.learn_start or frame % self.update_freq != 0
        #   return None
        # * take a random transition batch and compute loss
        # * optimize the model
        #################################################################################
        pass
        #################################################################################
        #                                   THE END                                     #
        #################################################################################

        self.update_target_model()       
        self.save_td(loss.item(), frame)
        self.save_sigma_param_magnitudes(frame)

    def get_action(self, s, eps=0.1):

        #################################################################################
        #                          COMPLETE THE FOLLOWING SECTION                       #
        #################################################################################  
         # implement get_action method (epsilon greedy)
         # you should use "with torch.no_grad():"
         ################################################################################# 
        pass

        #################################################################################
        #                                   THE END                                     #
        #################################################################################

    def update_target_model(self):
      # update target model:
        self.update_count+=1
        self.update_count = self.update_count % self.target_net_update_freq
        if self.update_count == 0:
            self.target_model.load_state_dict(self.model.state_dict())


#  ## 1.3. Training Loop (4 Points)

# In[ ]:


start=timer()

log_dir = "/tmp/gym/dqn"
try:
    os.makedirs(log_dir)
except OSError:
    files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))         + glob.glob(os.path.join(log_dir, '*td.csv'))         + glob.glob(os.path.join(log_dir, '*sig_param_mag.csv'))         + glob.glob(os.path.join(log_dir, '*action_log.csv'))
    for f in files:
        os.remove(f)

#################################################################################
#                          COMPLETE THE FOLLOWING SECTION                       #
#################################################################################
# do followings line by line:
# * use "make_atari" wrapper and save "PongNoFrameskip-v4" game to env
# * use "bench.Monitor" wrapper to know the episode reward, length, time and other data.
# * use "wrap_deepmind" wrapper to configure environment for DeepMind-style Atari.
# * use *WrapPyTorch*
# * save model
# * implement training loop
#################################################################################
env_id = "PongNoFrameskip-v4"
env =  None

model  = None

episode_reward = 0
observation = env.reset()
for frame_idx in range(1, MAX_FRAMES + 1):
    epsilon = None

    action = None

    model.save_action(action, frame_idx) #log action selection

    prev_observation = None
    observation, reward, done, _ = None

    ....

    if done:
        pass
        
#################################################################################
#                                   THE END                                     #
################################################################################# 
    if frame_idx % 1000 == 0:
        try:
            clear_output(True)
            plot_all_data(log_dir, env_id, 'DQN', MAX_FRAMES, bin_size=(10, 100, 100, 1), smooth=1, time=timedelta(seconds=int(timer()-start)), ipynb=True)
        except IOError:
            pass
 
env.close()
plot_all_data(log_dir, env_id, 'DQN', MAX_FRAMES, bin_size=(10, 100, 100, 1), smooth=1, time=timedelta(seconds=int(timer()-start)), ipynb=True)


# # 2. DRQN

# ## 2.1. Experience Replay (4 Points)

# In[ ]:


class RecurrentExperienceReplayMemory:
    def __init__(self, capacity, sequence_length=10):
        self.capacity = capacity
        self.memory = []
        self.seq_length = sequence_length

    def push(self, transition):
        #################################################################################
        #                          COMPLETE THE FOLLOWING SECTION                       #
        #################################################################################
        # complete push method.
        #################################################################################
        pass
        #################################################################################
        #                                   THE END                                     #
        #################################################################################  

    def sample(self, batch_size):
        #################################################################################
        #                          COMPLETE THE FOLLOWING SECTION                       #
        #################################################################################
        # complete sample method.
        # notice that you should take these tips into consideration
        # * sample here will be trajectory not transition
        # * should use padding if trajectories aren't in same len
        #################################################################################
        samples = []
        pass
        #################################################################################
        #                                   THE END                                     #
        #################################################################################  
        return samples

    def __len__(self):
        return len(self.memory)


# ## 2.2. Network Declaration (4 Points)

# In[ ]:


class RecurrentDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(RecurrentDQN,self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions
        #################################################################################
        #                          COMPLETE THE FOLLOWING SECTION                       #
        #################################################################################
        # Initialize CNN Model :
        # conv1: out_channels:32, kernel_size=8, stride=4
        # conv2: out_channels:64, kernel_size=4, stride=2
        # conv3: out_channels:64, kernel_size=3, stride=1
        # fc2(256)
        # GRU:  input_size: 256  hidden_size=256
        # fc2(256)
        #################################################################################
        pass
        #################################################################################
        #                                   THE END                                     #
        #################################################################################

    def forward(self, x, bsize, time_step, hidden_state, cell_state):
        #################################################################################
        #                          COMPLETE THE FOLLOWING SECTION                       #
        #################################################################################
        # complete forward pass using initialized CNN and GRU Model. use Relu activation
        # function for conv1, conv2, conv3 and fc1 . 
        #################################################################################
        pass  
        return x, hidden
      
      def init_hidden_states(self,bsize):
        h = None
        pass
        return h
        #################################################################################
        #                                   THE END                                     #
        #################################################################################   


#  ## 2.3. Agent (6 Points)

# In[ ]:


class RecurrentModel():
    def __init__(self, env=None, log_dir):
        self.sequence_length = SEQUENCE_LENGTH
        #################################################################################
        #                          COMPLETE THE FOLLOWING SECTION                       #
        #################################################################################
        # do followings line by line:
        # * use shape of observation space to save number of features
        # * save naumber of actions
        # * use DRQN class to declare model and target model 
        # * call parent class constructor
        # * declare memory
        # * reset hidden state
        #################################################################################
        self.num_feats = None
        self.num_actions = None
        model = None
        target_model = None
        super(RecurrentModel, self).__init__()
        self.memory = None
        #################################################################################
        #                                   THE END                                     #
        #################################################################################
        

    def prep_minibatch(self):
      
        #################################################################################
        #                          COMPLETE THE FOLLOWING SECTION                       #
        #################################################################################
        # random transition batch is taken from experience replay memory
        # do followings line by line:
        # * sample from self.memory with batch size, and save result in transitions
        # * use transitions to save batch_state, batch_action, batch_reward, 
        #   batch_next_state as tensors
        # * reshape batch_state, batch_action, batch_reward, batch_next_state into 
        #   (batch_size, sequence_length, feat_size)
        # * get set of next states for end of each sequence
        # * save non_final_mask,  non_final_next_states as tensors, note that sometimes 
        #   all next states are false
        #################################################################################
        try: 
            non_final_next_states = None
            empty_next_state_values = False
        except:
            non_final_next_states = None
            empty_next_state_values = True
        #################################################################################
        #                                   THE END                                     #
        #################################################################################
        return batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values

    def compute_loss(self, batch_vars):
        batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values = batch_vars

        #################################################################################
        #                          COMPLETE THE FOLLOWING SECTION                       #
        #################################################################################
        # implement calculation of loss (you should use "with torch.no_grad():" for target_model)
        #################################################################################
        pass
        #################################################################################
        #                                   THE END                                     #
        #################################################################################
        return loss

    def update(self, s, a, r, s_, frame=0):
        #################################################################################
        #                          COMPLETE THE FOLLOWING SECTION                       #
        #################################################################################
        # implement update method to optimize model 
        # * push state, action, reward and new state to memoty
        # * note that if frame is lower than self.learn_start or frame % self.update_freq != 0
        #   return None
        # * take a random transition batch and compute loss
        # * clamp grad between -1 and 1
        # * optimize the model 
        #################################################################################
        pass
        #################################################################################
        #                                   THE END                                     #
        #################################################################################

        self.update_target_model()       
        self.save_td(loss.item(), frame)
        self.save_sigma_param_magnitudes(frame)

    def get_action(self, s, eps=0.1):

        #################################################################################
        #                          COMPLETE THE FOLLOWING SECTION                       #
        #################################################################################  
         # implement get_action method (epsilon greedy)
         # you should use "with torch.no_grad():"
         ################################################################################# 
        pass

        #################################################################################
        #                                   THE END                                     #
        #################################################################################

    def reset_hx(self):
        self.seq = [np.zeros(self.num_feats) for j in range(self.sequence_length)]

    def finish_nstep(self):
        while len(self.nstep_buffer) > 0:
            R = sum([self.nstep_buffer[i][2] * (self.gamma ** i) for i in range(len(self.nstep_buffer))])
            state, action, _, _ = self.nstep_buffer.pop(0)

            self.memory.push((state, action, R, None))


#  ## 2.4. Training Loop (2 Points)

# In[ ]:


start = timer()

log_dir = "/tmp/gym/drqn"
try:
    os.makedirs(log_dir)
except OSError:
    files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))             + glob.glob(os.path.join(log_dir, '*td.csv'))             + glob.glob(os.path.join(log_dir, '*sig_param_mag.csv'))             + glob.glob(os.path.join(log_dir, '*action_log.csv'))
    for f in files:
        os.remove(f)
#################################################################################
#                          COMPLETE THE FOLLOWING SECTION                       #
#################################################################################
# do followings line by line:
# * use "make_atari" wrapper and save "PongNoFrameskip-v4" game to env
# * use "bench.Monitor" wrapper to know the episode reward, length, time and other data.
# * use "wrap_deepmind" wrapper to configure environment for DeepMind-style Atari.
# * use *WrapPyTorch*
# * save model
# * implement training loop
#################################################################################
env_id = "PongNoFrameskip-v4"
env =  None

model  = None

episode_reward = 0
observation = env.reset()

for frame_idx in range(1, MAX_FRAMES + 1):
    epsilon = None

    action = None

    model.save_action(action, frame_idx) #log action selection

    prev_observation = None
    observation, reward, done, _ = None

    ....

    if done:
        model.finish_nstep()
        model.reset_hx()
        pass
#################################################################################
#                                   THE END                                     #
################################################################################# 
    if frame_idx % 10000 == 0:
        try:
            clear_output(True)
            plot_all_data(log_dir, env_id, 'DRQN', MAX_FRAMES, bin_size=(10, 100, 100, 1), smooth=1,
                          time=timedelta(seconds=int(timer() - start)), ipynb=True)
        except IOError:
            pass

env.close()
plot_all_data(log_dir, env_id, 'DRQN', MAX_FRAMES, bin_size=(10, 100, 100, 1), smooth=1,
              time=timedelta(seconds=int(timer() - start)), ipynb=True)


#  ## 2.5. Interpret Results (4 Points)

# Explain what you have seen. Is using recurrent memory improve performance? support your answer
