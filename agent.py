# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 12:34:36 2020
Original code have been taken from https://github.com/kumararduino/Reinforcement-learning/blob/master/Q-Learning/git_simple_game.py
@author: AntonD
"""
import math

import numpy as np
from random import randint as r
import random
from gameboard import GameBoard


def init_q(env):
    q = np.random.random((env.n ** 2, 4))
    for i in range(len(q)):
        for j in [0, 1, 2, 3]:
            actions = [k[0] for k in env.get_actions(i)]
            if j not in actions:
                q[i, j] = -np.power(10, 4)

    return q


class Agent:
    def __init__(self, env: GameBoard, alpha=0.01, lamda=0.9, gamma=0.9, current_state=0, epsilon=0.1):
        """Agent that can use different tactic and algorithms playing on static board/environment
            The goal is to find the the end point as fast as possible at unknown environments.
            The environment can be characterized by Markov Decision Process
            This class have to get environment class or GameBoard"""
        # policy = None
        self.alpha = alpha
        self.lamda = lamda
        self.gamma = gamma
        self.current_state = current_state
        self.epsilon = epsilon
        self.env = env
        self.policy = None

        # self.Q = np.zeros((env.n ** 2, 4))  # Initializing Q-Table
        self.Q = init_q(env)  # Initializing Q-Table
        # self.Q = np.random.random((env.n ** 2, 4)) * 0.05 + np.ones((env.n ** 2, 4)) * 0.475  # Initializing Q-Table

    # -----------  Task 2  -----------
    def bellman_eq(self):
        """policy (~U) is unified probability that depends on the number of available actions
        Transition probability equal to 1 for any available action
        At any situation with every episode agent location will change"""
        n = self.env.n
        A = np.zeros((n ** 2, n ** 2))
        B = np.zeros((n ** 2))
        for eq in range(n ** 2):  # equation number for each state
            actions = self.env.get_actions(eq)
            pai = 1 / len(actions)
            tr = 1
            A[eq, eq] = 1

            # if current state is terminal we should to begin the game from state 0 again
            # There for all actions will take us to state 0
            if eq in self.env.terminals:
                pai = 1
                A[eq, 0] = -1 * pai * tr * self.gamma
                B[eq] = pai * tr * self.env.get_reward(0)
                continue

            for action in actions:
                A[eq, action[1]] = -1 * pai * tr * self.gamma
                B[eq] += pai * tr * self.env.get_reward(action[1])
        V = np.linalg.solve(A, B)
        return A, B, V

    def value_iteration(self):
        n = self.env.n
        V = np.zeros(n ** 2)  # initialize V with zeros
        v = np.zeros(n ** 2)  # initialize V with zeros
        delta = 1  # the initial number not really matter, it just should allow to enter to the loop
        steps = 0
        tr = 1
        while delta > 0.001:  # theta = 0.001
            steps += 1
            for state in range(self.env.n ** 2):  # for each state
                v[state] = V[state]
                # is state = terminal its always start game from the beginning at state 0
                if state in self.env.terminals:
                    pai = 1
                    V[state] = pai * tr * (self.env.get_reward(0) + self.gamma * V[0])
                    continue

                actions = self.env.get_actions(state)
                pai = 1 / len(actions)
                V[state] = 0
                for action in actions:
                    V[state] += pai * tr * (
                            self.env.get_reward(action[1]) + self.gamma * V[action[1]])
            delta = max(abs(v - V))
        return V, steps

    # -----------  Task 3  -----------
    def policy_eval(self, policy, V):
        n = self.env.n
        delta = 1  # the initial number not really matter, it just should allow to enter to the loop
        steps = 0
        tr = 1
        v = np.zeros(n ** 2)  # initialize V with zeros
        while delta > 0.001:  # theta = 0.001
            steps += 1
            for state in range(self.env.n ** 2):  # for each state
                v[state] = V[state]
                # is state = terminal its always start game from the beginning at state 0
                if state in self.env.terminals:
                    pai = 1  # its actually a sum of all possible actions because they all will lead us to start
                    V[state] = pai * tr * (self.env.get_reward(0) + self.gamma * V[0])
                    continue

                actions = self.env.get_actions(state)
                V[state] = 0
                for action in actions:
                    pai = policy[state, action[0]]
                    V[state] += pai * tr * (
                            self.env.get_reward(action[1]) + self.gamma * V[action[1]])
            delta = max(abs(v - V))
        return V, steps

    def policy_iteration(self):
        n = self.env.n
        tr = 1
        policy = np.ones((n ** 2, 4))  # random policy
        policy = policy / policy.sum(axis=1, keepdims=True)  # convert numbers into probabilities
        V = np.zeros(n ** 2)  # initialize V with zeros
        steps1 = 0
        steps2 = 0
        while True:
            V, steps = self.policy_eval(policy, V)
            steps1 += steps
            steps2 += 1
            policy_stable = True
            for state in range(self.env.n ** 2):
                b = np.argmax(policy[state])
                action_value = np.ones(4) * (-99999999)
                if state in self.env.terminals:
                    action_value = np.ones(4) * (tr * (self.env.get_reward(0) + self.gamma * V[0]))
                    continue

                actions = self.env.get_actions(state)
                for a in actions:
                    action_value[a[0]] = tr * (self.env.get_reward(a[1]) + self.gamma * V[a[1]])

                best_action = np.argmax(action_value)

                # Greedy Policy
                if b != best_action:
                    policy_stable = False
                policy[state] = np.eye(4)[best_action]
            print(steps2)
            if policy_stable:
                return policy, V, steps1 + steps2

    # -----------  Task 4  -----------
    def tmp_difference(self, num_episodes):
        v = np.zeros(self.env.n ** 2)  # Initialize the value of all states to 0
        # loop over all episodes
        action = [0, 0]
        for episode in range(num_episodes):
            self.current_state = 0
            # keep updating the episode until we reach a terminal state
            while not self.env.terminals.count(self.current_state):
                # we take the action that supposed to be taken in the previous loop, we are doing this
                # like that to let the terminal positions update their states as well and not exiting the loop immediately
                self.current_state = action[1]
                # choose a random action from the current state
                action = random.choice(self.env.get_actions(self.current_state))
                # update the Value of the state
                v[self.current_state] = v[self.current_state] + self.alpha * (
                        self.env.get_reward(self.current_state) + self.gamma * v[action[1]] - v[self.current_state])
        return v

    # -----------  Task 5  -----------
    def sarsa(self, num_episodes):
        initial_q = np.copy(self.Q)
        self.epsilon = 0.1
        for episode in range(num_episodes):
            self.current_state = next_state = 0
            while self.current_state not in self.env.terminals:
                # change to next state only after we enter the while for
                self.current_state = next_state
                action, next_state = self.epsilon_greedy_action()  # (action, next state)
                next_action, next_next_state = (self.epsilon_greedy_action(next_state))  # (next state action, next next state)
                reward = self.env.get_reward(self.current_state)
                self.Q[self.current_state, action] = self.Q[self.current_state, action] + self.alpha * (
                        reward + self.gamma * self.Q[next_state][next_action] - self.Q[self.current_state, action])

        return self.Q, initial_q

    def get_greedy_policy(self, q_table):
        actions_dict = {0: -7, 1: 7, 2: -1, 3: 1}
        policy_array = []
        state = 0
        for i in range(17):
            policy_array.append(state)
            lst_temp = [action[0] for action in self.env.get_actions(state)]
            chosen_action = random.choice(np.where(q_table[state] == max(q_table[state][lst_temp]))[0])
            state += actions_dict[int(chosen_action)]
        return policy_array

        # -----------  Task 6  -----------

    def q_learning(self, num_episodes):
        episodes = np.array([])
        count = 0
        cum_reward = 0
        for episode in range(num_episodes):

            self.current_state = next_state = 0
            while self.current_state not in self.env.terminals:
                count = count + 1
                action, next_state = self.epsilon_greedy_action()  # take next action from epsilon-greedy policy (action, next state)
                next_action, next_next_state = self.max_q_value(next_state)  # act greedly
                # get reward from taking an action at current_sate
                reward = self.env.get_reward(self.current_state)
                cum_reward = cum_reward + reward

                # update q value by TD rule
                self.Q[self.current_state, action] = self.Q[self.current_state, action] + self.alpha * (
                        reward + self.gamma * self.Q[next_state][next_action] - self.Q[self.current_state, action])
                # change to next state only after we update the q values
                self.current_state = next_state

            count = count + 1
            # when current_State is terminal - update for the final time then start a new episode
            action, next_state = self.epsilon_greedy_action()
            next_action, next_next_state = self.max_q_value(next_state)
            reward = self.env.get_reward(self.current_state)
            cum_reward = cum_reward + reward

            self.Q[self.current_state, action] = self.Q[self.current_state, action] + self.alpha * (
                    reward + self.gamma * self.Q[next_state][next_action] - self.Q[self.current_state, action])

            episodes = np.append(episodes, self.Q[1][1])
        #            if self.current_state==48:
        #                episodes=np.append(episodes, cum_reward/count)
        #                episodes=np.append(episodes, cum_reward)
        #                episodes=np.append(episodes, count)
        #                count=0
        #                cum_reward=0

        return episodes

    def max_q_value(self, state=None):
        state = state
        """choose action based on greedy policy"""
        if state == None:
            state = self.current_state

        if state in self.env.terminals:
            # in case we are on terminal, just take a valid action (all lead to same reward) and set next state as 0
            return random.choice([action[0] for action in self.env.get_actions(state)]), 0
        else:
            # list of all possible actions
            lst_temp = [action[0] for action in self.env.get_actions(state)]

            # make a list of possible action to pick from
            choose_max_from = [self.Q[state][i] for i in lst_temp]

            # pick the max q value of possible actions
            chosen_action = random.choice(np.where(self.Q[state] == max(choose_max_from))[0])
            # return tuple: (action index , next state)
            return chosen_action, self.env.get_actions(state)[lst_temp.index(chosen_action)][1]

    def epsilon_greedy_action(self, state=None):
        state = state
        """choose action based on epsilon-greedy policy"""
        if state == None:
            state = self.current_state

        if state in self.env.terminals:
            # in case we are on terminal, just took a valid action (all lead to same reward) and set next state as 0
            return random.choice([action[0] for action in self.env.get_actions(state)]), 0
        else:
            if random.random() > self.epsilon:
                return self.max_q_value(state)
            else:
                return random.choice(self.env.get_actions(state))

    def extract_greedy_policy(self):
        p = []
        state = 0
        while state not in self.env.terminals:
            p.append(state)
            state = self.max_q_value(state)[1]
        p.append(state)
        return p

    def q_lamda(self, num_episodes):
        episodes = np.array([])
        count = 0
        cum_reward = 0
        for episode in range(num_episodes):
            self.current_state = next_state = 0
            self.traces = np.zeros((self.env.n ** 2, 4))

            while self.current_state not in self.env.terminals:
                count = count + 1

                action, next_state = self.epsilon_greedy_action()  # take next action from epsilon-greedy policy (action, next state)
                next_possible_action, next_next_state1 = self.epsilon_greedy_action(next_state)
                next_action, next_next_state2 = self.max_q_value(next_state)  # act greedly
                # get reward from taking an action at current_sate
                reward = self.env.get_reward(self.current_state)

                cum_reward = cum_reward + reward
                # update the TD error (matrix -wise)
                TD_error = reward + self.gamma * self.Q[next_state][next_action] - self.Q[self.current_state, action]

                # update the current state's trace by 1
                self.traces[self.current_state][action] = self.traces[self.current_state][action] + 1
                # update q value by TD rule
                self.Q = self.Q + self.alpha * (TD_error * self.traces)

                if next_action == next_possible_action:
                    self.traces = self.traces * self.gamma * self.lamda

                else:
                    self.traces = np.zeros((self.env.n ** 2, 4))

                self.current_state = next_state

            count = count + 1
            # when current_State is terminal - update for the final time then start a new episode
            action, next_state = self.epsilon_greedy_action()
            next_possible_action, next_next_state1 = self.epsilon_greedy_action(next_state)
            next_action, next_next_state2 = self.max_q_value(next_state)
            reward = self.env.get_reward(self.current_state)

            cum_reward = cum_reward + reward
            TD_error = reward + self.gamma * self.Q[next_state][next_action] - self.Q[self.current_state, action]
            self.traces[self.current_state][action] = self.traces[self.current_state][action] + 1
            self.Q = self.Q + self.alpha * (TD_error * self.traces)

            episodes = np.append(episodes, self.Q[1][1])
        #            if self.current_state==48:

        #                episodes=np.append(episodes, cum_reward/count)
        #                episodes=np.append(episodes, cum_reward)
        #                episodes=np.append(episodes, count)
        #                count=0
        #                cum_reward=0

        #            self.epsilon =1/(episode+1)

        return episodes
