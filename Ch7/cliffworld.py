from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import time
from gym.envs.toy_text.cliffwalking import CliffWalkingEnv


LOGGING_STEP = 100

st.title('Cliffworld Q(sigma)-Learning')


class QAgent:
    def __init__(self, env):
        self.env = env
        self._Q = [[0] * env.nA for state in range(env.nS)]

    def greedy_policy(self, state, action=None):
        best_action = np.argmax([self._Q[state][a] for a in range(self.env.nA)])
        if action is None:
            return best_action
        else:
            return 1. if action == best_action else 0.

    def epsilon_greedy_policy(self, state, action=None):
        p = [self.epsilon / (self.env.nA - 1.) for _ in range(self.env.nA)]
        p[self.greedy_policy(state)] = 1. - self.epsilon
        if action is None:
            return np.random.choice(range(self.env.nA), p=p)
        else:
            return p[action]

    def train(self, alpha, discount, episodes, T):
        raise NotImplementedError


class ExpectedSarsaAgent(QAgent):
    def __init__(self, env, epsilon):
        self.env = env
        self.epsilon = epsilon
        super(ExpectedSarsaAgent, self).__init__(env)
        # st.write(f'Created expected Sarsa agent with {epsilon}-greedy policy')

    def train(self, alpha, discount, episodes=1000, T=1000):
        """
        st.write(
            f'Training {self.__class__.__name__} for {episodes} episodes '
            + f'of max length {T} steps '
            + f'with learning rate {alpha} and discount factor {discount}'
        )
        """
        for episode in range(episodes):
            if episode % LOGGING_STEP == 0:
                st.text(f'Episode {episode}')
            states = [self.env.reset()]
            actions = [self.epsilon_greedy_policy(states[-1])]
            rewards = []
            for t in range(T):
                state, reward, done, info = self.env.step(actions[-1])
                states.append(state)
                rewards.append(reward)
                if done:
                    break
                else:
                    actions.append(self.epsilon_greedy_policy(state))
                s_t, s_t1, a_t = states[-2], states[-1], actions[-2]
                V = sum([
                    self.epsilon_greedy_policy(s_t1, a) * self._Q[s_t1][a]
                    for a in range(self.env.nA)
                ])
                self._Q[s_t][a_t] += (
                    alpha * (rewards[-1] + discount * V - self._Q[s_t][a_t])
                )


class QSigmaAgent(QAgent):
    """
    Args:
        n: n-step Q(sigma) algorithm
        sigma: function returning 0 <= sigma(t) <= 1
        env: OpenAI gym environment
        alpha: learning rate
        epsilon: for epsilon-greedy policy
        discount: gamma for calculating discounted return
    """

    def __init__(self, n, sigma, env, epsilon):
        self.n = n
        self.sigma = sigma
        self.epsilon = epsilon
        super(QSigmaAgent, self).__init__(env)
        # sigma_str = ','.join([str(sigma(t)) for t in range(n)])
        # st.write(f'Created {n}-step Q({sigma_str}) agent')

    def train(self, alpha, discount, episodes=4000, T=1000):
        def update(steps, states, actions, rewards):
            """steps is [t + 1, t, t - 1, ..., t + 2 - n]
            """
            G = self._Q[states[steps[0]]][actions[steps[0]]]
            for k in steps:
                if done:
                    G = rewards[k - 1]
                else:
                    s_k, a_k = states[k], actions[k]
                    V = sum([
                        self.epsilon_greedy_policy(s_k, a) * self._Q[s_k][a]
                        for a in range(self.env.nA)
                    ])
                    G = (
                        rewards[k - 1]
                        + discount
                        * (self.sigma(k) + (1 - self.sigma(k))
                            * self.epsilon_greedy_policy(s_k, a_k))
                        * (G - self._Q[s_k][a_k])
                        + discount * V
                    )
            tau = steps[-1] - 1
            self._Q[states[tau]][actions[tau]] += (
                alpha * (G - self._Q[states[tau]][actions[tau]])
            )

        """
        st.write(
            f'Training {self.__class__.__name__} for {episodes} episodes '
            + f'of max length {T} steps '
            + f'with learning rate {alpha} and discount factor {discount}'
        )
        """
        steps = deque()
        for episode in range(episodes):
            if episode % LOGGING_STEP == 0:
                # st.text(f'Episode {episode}')
                pass
            states = [self.env.reset()]
            actions = [self.epsilon_greedy_policy(states[-1])]
            rewards = []
            steps.clear()
            steps.appendleft(0)
            for t in range(T):
                state, reward, done, info = self.env.step(actions[-1])
                states.append(state)
                rewards.append(reward)
                actions.append(self.epsilon_greedy_policy(state))
                steps.appendleft(t + 1)
                if len(steps) > self.n:
                    steps.pop()
                if len(steps) == self.n:
                    update(steps, states, actions, rewards)
                if done:
                    break
            steps.pop()
            while steps:
                update(steps, states, actions, rewards)
                steps.pop()


class CliffWalkingGraphicEnv(CliffWalkingEnv):
    def __init__(self):
        super(CliffWalkingGraphicEnv, self).__init__()

    def render(self, mode='graphic'):
        if mode == 'graphic':
            heatmap = [[50] * self.shape[1] for _ in range(self.shape[0])]
            for s in range(self.nS):
                i, j = np.unravel_index(s, self.shape)
                if s == self.s:
                    heatmap[i][j] = 0
                elif (i, j) == (self.shape[0] - 1, self.shape[1] - 1):
                    heatmap[i][j] = 100
                elif self._cliff[i, j]:
                    heatmap[i][j] = 75
            return np.array(heatmap, dtype=np.float32)
        elif mode == 'human':
            return super(CliffWalkingGraphicEnv, self).render(mode)


def sigmaSarsa(t):
    return 1.


def sigmaTreeBackup(t):
    return 0.


def sigmaOneZero(t):
    return (t + 1) % 2


def sigmaZeroOne(t):
    return t % 2


def sigmaBalanced(t):
    return 0.5


def get_q_sigma(env, n, sigma):
    epsilon = 0.1
    agent = QSigmaAgent(n, sigma, env, epsilon)
    return agent


def get_expected_sarsa(env):
    epsilon = 0.1
    agent = ExpectedSarsaAgent(env, epsilon)
    return agent


def main():
    max_steps = 25
    env = CliffWalkingGraphicEnv()
    alg = st.radio(
        'Training algorithm',
        (
            'Expected Sarsa',
            '1-step Q(1) Learning',
            '2-step Q(1, 1) Learning',
            '1-step Q(0) Learning',
            '2-step Q(0, 0) Learning',
            '2-step Q(1, 0) Learning',
            '2-step Q(0, 1) Learning',
            '1-step Q(0.5) Learning',
            '2-step Q(0.5, 0.5) Learning',
        )
    )
    if alg == 'Expected Sarsa':
        agent = get_expected_sarsa(env)
    elif alg == '1-step Q(1) Learning':
        agent = get_q_sigma(env, 1, sigmaSarsa)
    elif alg == '2-step Q(1, 1) Learning':
        agent = get_q_sigma(env, 2, sigmaSarsa)
    elif alg == '1-step Q(0) Learning':
        agent = get_q_sigma(env, 1, sigmaTreeBackup)
    elif alg == '2-step Q(0, 0) Learning':
        agent = get_q_sigma(env, 2, sigmaTreeBackup)
    elif alg == '2-step Q(1, 0) Learning':
        agent = get_q_sigma(env, 2, sigmaOneZero)
    elif alg == '2-step Q(0, 1) Learning':
        agent = get_q_sigma(env, 2, sigmaZeroOne)
    elif alg == '1-step Q(0.5) Learning':
        agent = get_q_sigma(env, 1, sigmaBalanced)
    elif alg == '2-step Q(0.5, 0.5) Learning':
        agent = get_q_sigma(env, 2, sigmaBalanced)
    alpha = st.slider('alpha', 0.1, 1.2, 0.5, 0.1)
    discount = st.slider('discount', 0., 1.0, 0.95, 0.05)
    episodes = st.slider('episodes', 1, 201, 201, 20)
    agent.train(alpha=alpha, discount=discount, episodes=episodes)
    state = env.reset()
    frames = [env.render(mode='graphic')]
    for _ in range(max_steps):
        action = agent.greedy_policy(state)
        state, _, done, _ = env.step(action)
        frames.append(env.render(mode='graphic'))
        if done:
            break
    # st.plotly_chart(figure(env, frames))
    fig, ax = plt.subplots()
    the_plot = st.pyplot(plt)
    for frame in frames:
        plt.imshow(frame, cmap='hot')
        the_plot.pyplot(plt)
        time.sleep(0.2)


if __name__ == '__main__':
    main()
