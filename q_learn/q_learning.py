import numpy as np
import random
from IPython.display import clear_output
import gym
import matplotlib.pyplot as plt

# Globals
ALPHA = 0.1
GAMMA = 0.6
EPSILON = 0.05

# As the first step of this experiment we implement simple environment:


class MDP():
    def __init__(self, action_tree=9):
        # actions:
        self.down, self.up = 0, 1

        # state and possible actions:
        self.state_actions = {
            'X': [self.down, self.up],
            'Y': [i for i in range(action_tree)],
            'W': [self.down],
            'Z': [self.up]
        }

        # Transitions
        self.transitions = {
            'X': {self.down: 'Z',
                  self.up: 'Y'},
            'Y': {a: 'W' for a in range(action_tree)},
            'W': {self.down: 'Done'},
            'Z': {self.up: 'Done'}
        }

        self.state_space = 4
        self.action_space = action_tree
        self.state = 'X'

    def _get_reward(self):
        return np.random.normal(-0.5, 1) if self.state == 'W' else 0

    def _is_terminated_state(self):
        return True if self.state == 'W' or self.state == 'Z' else False

    def reset(self):
        self.state = 'X'
        return self.state

    def step(self, action):
        self.state = self.transitions[self.state][action]
        return self.state, self._get_reward(), self._is_terminated_state()

    def available_actions(self, state):
        return self.state_actions[state]

    def random_action(self):
        return np.random.choice(self.available_actions(self.state))


def mdp_q_learning(environment, num_of_tests=2000, num_of_episodes=300):
    num_of_ups = np.zeros(num_of_episodes)

    for _ in range(num_of_tests):

        if _ % 100 == 0:
            clear_output(wait=True)
            print(f'#test : {_}')

        # initialize Q-table
        q_table = {state: np.zeros(9) for state in environment.state_actions.keys()}
        rewards = np.zeros(num_of_episodes)

        for episode in range(0, num_of_episodes):
            #             clear_output(wait=True)
            #             print(f'#test : {_}')
            #             print(f'#episode: {episode}')

            # rest the env
            state = environment.reset()

            # initialize variables
            terminated = False

            while not terminated:

                # pick action a...
                if np.random.rand() < EPSILON:
                    action = environment.random_action()
                else:
                    available_actions = environment.available_actions(environment.state)
                    state_actions = q_table[state][available_actions]
                    max_q = np.where(np.max(state_actions) == state_actions)[0]
                    action = np.random.choice(max_q)

                # ...and get r and s'
                next_state, reward, terminated = environment.step(action)

                # 'up's from state 'X'
                if state == 'X' and action == 1:
                    num_of_ups[episode] += 1

                # update q-table
                max_value = np.max(q_table[next_state])
                q_table[state][action] += ALPHA * (reward + GAMMA * max_value - q_table[state][action])
                state = next_state
                rewards[episode] += reward

    return rewards, q_table, num_of_ups


if __name__ == '__main__':
    mdp_environment = MDP()
    q_reward, q_table, num_of_ups = mdp_q_learning(mdp_environment)

    plt.figure(figsize=(15, 8))
    plt.plot(num_of_ups / 10000 * 100, label='UPs in X', color='red')
    plt.plot(q_reward, color='blue', label='Reward')
    plt.legend()
    plt.ylabel('Percentage of ups in state X')
    plt.xlabel('Episodes')
    plt.title(r'Q-Learning')
    plt.show()
