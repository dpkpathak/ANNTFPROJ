# # Functions for testing trained agents and visualizing test performance.

# Tensorflow.
# %tensorflow_version 2.x
import tensorflow as tf

# OpenAI Gym.
from gym import wrappers
import gym
from gym import logger as gymlogger
from gym.wrappers import Monitor
gymlogger.set_level(40) #error only

# Visualization.
import matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline
import base64
from IPython.display import HTML
from IPython import display as ipythondisplay

# Other.
import os
import numpy as np
import math
import glob
import io
import random
from tqdm import tqdm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
tf.get_logger().setLevel(logging.ERROR)


# Epsilon greedy policy variables.
max_epsilon = 1
min_epsilon = 0.01
lambda_ = 0.0005

# DQN algorithm variables.
gamma = 0.95
batch_size = 32
tau = 0.08
max_experiences = 500000
min_experiences = 96

# Network model variables.
hidden_units = [50,50]
steps = 0

# Environment variables.
env = gym.make("CartPole-v0")
num_actions = env.action_space.n
num_states = len(env.observation_space.sample())
num_episodes = 100
total_rewards = np.empty(num_episodes)
total_loss = np.empty(num_episodes)


# Class objects and functions for implementing the DQN and DDQN algorithms.

import tensorflow as tf

class Qnetwork(tf.keras.Model):
  """
  A single Q-value approximation network. Used for implementing online and target networks in DQN/DDQN.
  """
  def __init__(self, num_states, hidden_units, num_actions):
      super(Qnetwork, self).__init__()
      self.input_layer = tf.keras.layers.InputLayer(input_shape=(num_states,))
      self.hidden_layers = []
      for units in hidden_units:
          self.hidden_layers.append(tf.keras.layers.Dense(
              units, activation='relu', kernel_initializer=tf.keras.initializers.he_normal()))
      self.output_layer = tf.keras.layers.Dense(num_actions)

  @tf.function
  def call(self, inputs):
      x = self.input_layer(inputs)
      for layer in self.hidden_layers:
          x = layer(x)
      output = self.output_layer(x)
      return output


class ExperienceBuffer:
  """
  Stores experiences for training approximation network.
  """
  def __init__(self, max_experiences, min_experiences):
    self.max_experiences = max_experiences
    self.min_experiences = min_experiences
    self.experiences= []

  def add_experiences(self, exp):
    self.experiences.append(exp)
    if len(self.experiences) > self.max_experiences:
      self.experiences.pop(0)

# Reinforcement learning agent class.

class Agent:
  """
  Reinforcement learning agent to implement DQN and DDQN algorithms.
  """
  def __init__(self, train_net, gamma, batch_size, num_actions, algorithm, experience, target_net=None):
    # Network for approximation function.
    self.train_net = train_net
    # Target network to regulate training.
    self.target_net = target_net
    # Discount factor for next states.
    self.gamma = gamma
    # Batch size for training network.
    self.batch_size = batch_size
    # Network optimizer.
    self.optimizer = tf.optimizers.Adam()
    # Loss function.
    self.mse = tf.keras.losses.MeanSquaredError()
    # Actions space.
    self.num_actions = num_actions
    # Algorithm: DQN or DDQN
    self.algorithm = algorithm
    # Experience buffer
    self.experience_buffer = experience


  def get_action(self, state, eps):
    """
    Select action based on epsilon-greedy policy values. As value decays,
    the agent will tend to exploit the environment. Before decay, the agent
    will explore the environment to gather experiences.
    """
    if random.random() < eps:
      # Explore.
      return random.randint(0, num_actions - 1)
    else:
      # Exploit.
      return np.argmax(self.train_net(state.reshape(1, -1)))


  def train(self,exp_buffer):
    """
    Update approximation function (network) based on prior experience and
    either DQN or DDQN update algorithm.
    """
    # Return zero loss if not enough experiences for training.
    if len(exp_buffer.experiences) < exp_buffer.min_experiences:
        return 0

    # Gather experiences for training.
    ids = np.random.randint(low=0, high=len(exp_buffer.experiences), size=self.batch_size)
    states = np.array([exp_buffer.experiences[id_][0] for id_ in ids])
    actions = np.array([exp_buffer.experiences[id_][1] for id_ in ids])
    rewards = np.array([exp_buffer.experiences[id_][2] for id_ in ids])
    next_states = np.array([(np.zeros(4)if exp_buffer.experiences[id_][3] is None else exp_buffer.experiences[id_][3]) for id_ in ids])

    # Obtaining Q from states
    Q_train = self.train_net(states)
    # Obtaining Q prime from next states
    Q_train_prime = self.train_net(next_states)
    # Compute target Q.
    Q_train_target = Q_train.numpy()

    updates = rewards

    # Select only valid ids for training, ie those without next state == 0.
    valid_idxs = np.array(next_states).sum(axis=1) != 0
    batch_idxs = np.arange(self.batch_size)

    if self.algorithm == 'dqn':
      updates[valid_idxs] += self.gamma * np.amax(Q_train_prime.numpy()[valid_idxs, :], axis=1)

    elif self.algorithm == 'ddqn':
      A_prime = np.argmax(Q_train_prime.numpy(), axis=1)
      Q_target = self.target_net(next_states)
      updates[valid_idxs] += self.gamma * Q_target.numpy()[batch_idxs[valid_idxs], A_prime[valid_idxs]]

    Q_train_target[batch_idxs, actions] = updates
    with tf.GradientTape() as tape:
      output = self.train_net(states)
      loss = self.mse(Q_train_target, output)
      gradients = tape.gradient(loss, self.train_net.trainable_variables)

    # Apply gradients
    self.optimizer.apply_gradients(zip(gradients, self.train_net.trainable_variables))

    if self.algorithm == 'ddqn':
      #copying train network into target network partially
      for t, e in zip(self.target_net.trainable_variables, self.train_net.trainable_variables):
        t.assign(t * (1 - tau) + e * tau)

    return loss


  def learn(self):
    """
    Reinforcement learning within openAI gym environment.
    """
    eps = max_epsilon
    train_steps = []
    train_losses = []
    train_rewards = []
    steps = 0
    for i in range(num_episodes):
      rewards = 0
      state = env.reset()
      iteration = 0
      avg_loss = 0
      while True:
        action = self.get_action(state, eps)
        next_state, reward, done, info = env.step(action)
        reward = np.random.normal(1, 1)
        if done:
          next_state = None
        self.experience_buffer.add_experiences((state, action, reward, next_state))
        loss = self.train(self.experience_buffer)
        avg_loss += loss
        state = next_state
        steps += 1
        eps = min_epsilon + (max_epsilon - min_epsilon) * math.exp(-lambda_ * steps)
        if done:
          avg_loss /= iteration
          rewards = iteration
          break
        iteration += 1
      train_steps.append(i)
      train_rewards.append(rewards)
      train_losses.append(float(avg_loss))
      print(f"Episode: {i}\t Reward: {rewards}\t Loss: {avg_loss: .3f}\t Epsilon: {eps: .3f}")
    self.train_steps = train_steps
    self.train_losses = train_losses
    self.train_rewards = train_rewards


  def visualize_training(self):
    """
    Visualize rewards and loss during training episodes.
    """
    fig = plt.figure(figsize=(16,5))
    loss_ax = fig.add_subplot(1,2,1)
    plt.plot(self.train_steps, self.train_losses)
    loss_ax.title.set_text('Loss plot')

    loss_ax.set_xlabel('Episodes')
    loss_ax.set_ylabel('Loss')

    rewards_ax = fig.add_subplot(1,2,2)
    plt.plot(self.train_steps, self.train_rewards)

    rewards_ax.title.set_text('Reward plot')

    rewards_ax.set_xlabel('Episode')
    rewards_ax.set_ylabel('Rewards')
    plt.show()
