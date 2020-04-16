# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 12:34:36 2020
Original code have been taken from https://github.com/kumararduino/Reinforcement-learning/blob/master/Q-Learning/git_simple_game.py
@author: AntonD
"""

import numpy as np
from random import randint as r
import random
from gameboard import GameBoard


class Agent:
    def __init__(self, env: GameBoard, alpha=0.01, gamma=0.9, current_state=0, epsilon=0.25):
        """Agent that can use different tactic and algorithms playing on static board/environment
            The goal is to find the the end point as fast as possible at unknown environments.
            The environment can be characterized by Markov Decision Process
            This class have to get environment class or GameBoard"""
        # policy = None
        self.alpha = alpha
        self.gamma = gamma
        self.current_state = current_state or 0  # current pos set to 0 in case its it's None
        self.epsilon = epsilon
        self.env = env
        self.policy = None

        # initialize Q
        # self.Q = np.zeros((env.n ** 2, 4))  # Initializing Q-Table
        self.Q = np.random.random((env.n ** 2, 4))  # Initializing Q-Table

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
        V = np.zeros(self.env.n ** 2)  # Initialize the value of all states to 0
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
                V[self.current_state] = V[self.current_state] + self.alpha * (
                        self.env.get_reward(self.current_state) + self.gamma * V[action[1]] - V[self.current_state])
        return V

    # -----------  Task 5  -----------
    def sarsa(self, num_episodes):
        self.policy, V, steps = self.policy_iteration()
        initial_q = np.copy(self.Q)
        self.epsilon = 0.1
        for episode in range(num_episodes):
            self.current_state = next_state = 0
            while self.current_state not in self.env.terminals:
                # change to next state only after we enter the while for
                self.current_state = next_state
                action, next_state = self.greedy_action()  # (action, next state)
                next_action, next_next_state = (self.greedy_action(next_state))  # (next state action, next next state)
                reward = self.env.get_reward(self.current_state)
                self.Q[self.current_state, action] = self.Q[self.current_state, action] + self.alpha * (
                        reward + self.gamma * self.Q[next_state][next_action] - self.Q[self.current_state, action])

        return self.Q, initial_q

    def greedy_action(self, state=None):
        """choose action based on epsilon-greedy policy"""
        state = state or self.current_state
        if state in self.env.terminals:
            # in case we are on terminal, just took a valid action (all lead to same reward) and set next state as 0
            return random.choice([action[0] for action in self.env.get_actions(state)]), 0
        else:
            if random.random() > self.epsilon:
                # list of all possible actions
                lst_temp = [action[0] for action in self.env.get_actions(state)]
                # in case of same max-q-value between actions, pick randomly between them
                chosen_action = random.choice(np.where(self.policy[state] == max(self.policy[state]))[0])
                # return tuple: (action index , next state)
                return chosen_action, self.env.get_actions(state)[lst_temp.index(chosen_action)][1]
            else:
                return random.choice(self.env.get_actions(state))

    # -----------  Task 6  -----------
    def q_learning(self):
        pass

    def eligibility_traces(self):
        pass
