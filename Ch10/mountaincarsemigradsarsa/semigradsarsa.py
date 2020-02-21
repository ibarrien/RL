import numpy as np
import random
import random

class SemiGradientSarsa(object):
    """Semi Gradient Sarsa"""
    def __init__(self, action_space, observation_space, regularization, features=lambda x: x):
        self.action_space = action_space
        self.actions = list(range(self.action_space.n))
        self.observation_space = observation_space
        self.features = features
        no_features = features(observation_space.sample()).size
        # self.weights = { action: np.zeros(shape=(self.observation_space.low.size + 1,)) for action in self.actions}
        self.weights = { action: np.zeros(shape=(no_features,)) for action in self.actions}
        self.regularization = regularization

    def q_tilde(self, observation, action):
      # return np.dot(np.append([1], observation), self.weights[action])
      return np.dot(self.features(observation), self.weights[action])

    def dq_tilde(self, observation, action):
      # return np.append([1], observation) # Compute derivative
      return self.features(observation) # Compute derivative

    def act(self, observation, epsilon):
      if np.random.uniform() < epsilon:
        # print("### 002")
        return random.choice(self.actions)
      else:
        m = max([self.q_tilde(observation, action) for action in self.actions])
        a = random.choice([action for action in self.actions if self.q_tilde(observation, action) == m])
        return a
        # return max(self.actions, key=lambda action: self.q_tilde(observation, action))

    def update(self, alpha, gamma, action, observation, reward, observation_prime, action_prime, episode_ix):
      try:
        self.weights[action] += alpha*(reward + gamma*self.q_tilde(observation_prime, action_prime) - self.q_tilde(observation, action))*self.dq_tilde(observation, action) - alpha*self.regularization*self.weights[action]
      except RuntimeWarning as e:
        print("### 401", episode_ix)

    def terminal_update(self, alpha, action, observation, reward):
      self.weights[action] += alpha*(reward - self.q_tilde(observation, action))*self.dq_tilde(observation, action)- self.regularization*self.weights[action]



