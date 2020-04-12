# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 12:34:36 2020
Originsl code have been taken from https://github.com/kumararduino/Reinforcement-learning/blob/master/Q-Learning/git_simple_game.py
@author: AntonD
"""

import numpy as np
from random import randint as r
import random
from gameboard import GameBoard


class Agent:
    def __init__(self, env: GameBoard, alpha=0.01, gamma=0.9, current_pos=[0, 0], epsilon=0.1):
        '''Agent that can use different tactic and algorithms playing on static board/environment
            The goal is to find the the end poit as fast as possible at unnown environments.
            The environment can be characterized by Markov Deciesion Process           
            This class have to get environment class or GameBoard'''
        # policy = None
        self.alpha = alpha
        self.gamma = gamma
        self.current_pos = current_pos
        self.epsilon = epsilon
        self.env = env
        self.policy = None

        # initialize Q
        self.Q = np.zeros((env.n ** 2, 4))  # Initializing Q-Table
        # self.Q = rand((n**2,4)) #Initializing Q-Table

    def postostate(self, i, j):
        return (self.env.n * i + j)

    def statetopos(self, state):
        return int(state / self.env.n), state % self.env.n

        # -----------  Task 2  -----------

    def BelamanEq(self):
        '''policy (~U) is unified probability that depends on the number of available actions
        Transition probability equal to 1 for any available action
        At any situtation with every episode agent location will change'''
        n = self.env.n
        A = np.zeros((n ** 2, n ** 2))
        B = np.zeros((n ** 2))
        for eq in range(n ** 2):  # equation number for each state
            actions = self.env.get_actions(*self.statetopos(eq))
            pai = 1 / len(actions)
            tr = 1
            A[eq, eq] = 1

            # if current state is terminal we should to begin the game frop state 0 again
            # There for all actions will take us to state 0
            if eq in self.env.terminals:
                pai = 1
                A[eq, 0] = -1 * pai * tr * self.gamma
                B[eq] = pai * tr * self.env.getreward(0, 0)
                continue

            for action in actions:
                A[eq, action[1]] = -1 * pai * tr * self.gamma
                B[eq] += pai * tr * self.env.getreward(*self.statetopos(action[1]))
        V = np.linalg.solve(A, B)
        return A, B, V

    def value_iteration(self):
        n = self.env.n
        V = np.zeros((n ** 2))  # initialize V with zeros
        v = np.zeros((n ** 2))  # initialize V with zeros
        delta = 1  # the initial number not realy metter, it just should allow to enter to the loop
        steps = 0
        tr = 1
        while delta > 0.001:  # teta = 0.001
            steps += 1
            for state in range(self.env.n ** 2):  # for each state
                v[state] = V[state]
                # is state = terminal its always start game from the begining at state 0
                if state in self.env.terminals:
                    pai = 1
                    V[state] = pai * tr * (self.env.getreward(0, 0) + self.gamma * V[0])
                    continue

                actions = self.env.get_actions(*self.statetopos(state))
                pai = 1 / len(actions)
                V[state] = 0
                for action in actions:
                    V[state] += pai * tr * (self.env.getreward(*self.statetopos(action[1])) + self.gamma * V[action[1]])
            delta = max(abs(v - V))
        return V, steps

    # -----------  Task 3  -----------
    def policy_eval(self, policy, V):
        n = self.env.n
        delta = 1  # the initial number not realy metter, it just should allow to enter to the loop
        steps = 0
        tr = 1
        v = np.zeros((n ** 2))  # initialize V with zeros
        while delta > 0.001:  # teta = 0.001
            steps += 1
            for state in range(self.env.n ** 2):  # for each state
                v[state] = V[state]
                # is state = terminal its always start game from the begining at state 0
                if state in self.env.terminals:
                    pai = 1  # its actually a sum of all posible actions because they all will lead us to start
                    V[state] = pai * tr * (self.env.getreward(0, 0) + self.gamma * V[0])
                    continue

                actions = self.env.get_actions(*self.statetopos(state))
                V[state] = 0
                for action in actions:
                    pai = policy[state, action[0]]
                    V[state] += pai * tr * (self.env.getreward(*self.statetopos(action[1])) + self.gamma * V[action[1]])
            delta = max(abs(v - V))
        return V, steps

    def policy_iteration(self):
        n = self.env.n
        tr = 1
        policy = np.ones((n ** 2, 4))  # random policy
        policy = policy / policy.sum(axis=1, keepdims=True)  # convert numbers into probabilities
        V = np.zeros((n ** 2))  # initialize V with zeros
        steps1 = 0
        steps2 = 0
        while (True):
            V, steps = self.policy_eval(policy, V)
            steps1 += steps
            steps2 += 1
            policy_stable = True
            for state in range(self.env.n ** 2):
                b = np.argmax(policy[state])
                action_value = np.zeros((4))
                if state in self.env.terminals:
                    action_value = np.ones((4)) * (tr * (self.env.getreward(0, 0) + self.gamma * V[0]))
                    continue

                actions = self.env.get_actions(*self.statetopos(state))
                for a in actions:
                    action_value[a[0]] = tr * (self.env.getreward(*self.statetopos(a[1])) + self.gamma * V[a[1]])

                best_action = np.argmax(action_value)

                # Gridy Policy
                if b != best_action:
                    policy_stable = False
                policy[state] = np.eye(4)[best_action]
            print(steps2)
            if policy_stable:
                return policy, V, steps1 + steps2

    # -----------  Task 4  -----------
    def tmp_difference(self):
        pass

    # -----------  Task 5  -----------
    def TD_state_action(self):
        pass

    # -----------  Task 6  -----------
    def Q_learning(self):
        pass

    def eligibility_traces(self):
        pass

    # def select_action(current_state):
    #     action = random.choice([i for i,a in enumerate(possible_actions) if a == max(possible_actions)]) #randomly selecting one of all possible actions with maximin value
    #     return action

    # def episode():
    #     global current_pos,epsilon
    #     current_state = states[(current_pos[0],current_pos[1])]
    #     action = select_action(current_state)
    #     if action == 0: #move up
    #         current_pos[0] -= 1
    #     elif action == 1: #move down
    #         current_pos[0] += 1
    #     elif action == 2: #move left
    #         current_pos[1] -= 1
    #     elif action == 3: #move right
    #         current_pos[1] += 1
    #     new_state = states[(current_pos[0],current_pos[1])]
    #     if new_state not in terminals:
    #         Q[current_state,action] += alpha*(reward[current_pos[0],current_pos[1]] + gamma*(np.max(Q[new_state])) - Q[current_state,action])
    #     else:
    #         Q[current_state,action] += alpha*(reward[current_pos[0],current_pos[1]] - Q[current_state,action])
    #         current_pos = [0,0]
    #         if epsilon > 0.05:
    #             epsilon -= 3e-4 #reducing as time increases to satisfy Exploration & Exploitation Tradeoff
