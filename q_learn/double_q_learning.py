import numpy as np
import random
from IPython.display import clear_output
import gym
import matplotlib.pyplot as plt

from q_learning import MDP

# Globals
ALPHA = 0.1
GAMMA = 0.6
EPSILON = 0.05


def mdp_double_q_learning(environment, num_of_tests=2000, num_of_episodes=300):
    num_of_ups = np.zeros(num_of_episodes)

    for _ in range(num_of_tests):

        if _ % 100 == 0:
            clear_output(wait=True)
            print(f'#test : {_}')

        # initialize the Q-tables
        q_a_table = {state: np.zeros(9) for state in environment.state_actions.keys()}
        q_b_table = {state: np.zeros(9) for state in environment.state_actions.keys()}
        rewards = np.zeros(num_of_episodes)

        for episode in range(num_of_episodes):
            # reset the environment
            state = environment.reset()

            # initialize the variables
            terminated = False

            while not terminated:
                # pick an action a...
                if np.random.rand() < EPSILON:
                    action = environment.random_action()
                else:
                    q_table = q_a_table[state][environment.available_actions(environment.state)] + \
                              q_b_table[state][environment.available_actions(environment.state)]
                    max_q = np.where(np.max(q_table) == q_table)[0]
                    action = np.random.choice(max_q)

                # ... and get r and s'
                next_state, reward, terminated = environment.step(action)

                # ups from X
                if state == 'X' and action == 1:
                    num_of_ups[episode] += 1

                # update A or B
                if np.random.rand() < 0.5:
                    # If Update(A)
                    q_a_table[state][action] += ALPHA * (
                                reward + GAMMA * q_b_table[next_state][np.argmax(q_a_table[next_state])] -
                                q_a_table[state][action])

                else:
                    # If Update(B)
                    q_b_table[state][action] = ALPHA * (
                                reward + GAMMA * q_a_table[next_state][np.argmax(q_b_table[next_state])] -
                                q_b_table[state][action])

                state = next_state
                rewards[episode] += reward

    return rewards, q_a_table, q_b_table, num_of_ups


if __name__ == '__main__':
    mdp_environment = MDP()
    dq_reward, _, _, dq_num_of_ups = mdp_double_q_learning(mdp_environment)

    plt.figure(figsize=(15, 8))
    plt.plot(dq_num_of_ups / 10000 * 100, label='UPs in X', color='#FF171A')
    plt.plot(dq_reward, color='#6C5F66', label='Reward')
    plt.legend()
    plt.ylabel('Percentage of UPs in state X')
    plt.xlabel('Episodes')
    plt.title(r'Double Q-Learning')
    plt.show()
