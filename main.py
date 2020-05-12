# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 12:34:36 2020
Original code have been taken from https://github.com/kumararduino/Reinforcement-learning/blob/master/Q-Learning/git_simple_game.py
@author: AntonD
"""
import seaborn as sns
import numpy as np
import pygame
from gameboard import GameBoard
from agent import Agent
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
import random
from operator import itemgetter

alpha = 0.001
gamma = 0.9
lamda = 0.9
current_pos = 0
epsilon = 0.25
NUM_OF_EPISODES = 100000

# both are important!
np.random.seed(679)
random.seed(679)


def test_bellman():
    """Calculates and shows Bellman Eq."""
    board1 = GameBoard()
    agent = Agent(board1, alpha, gamma, current_pos, epsilon)
    A, B, V = agent.bellman_eq()

    V = V.reshape((board1.n, board1.n))
    board1.draw_board_with_num(V)
    print("V Matrix")
    for i in range(V.shape[0]):
        line = ""
        for j in range(V.shape[1]):
            line += ' %5.3f ' % (V[i, j])
        print(line)


def test_value_iteration():
    board1 = GameBoard()
    agent = Agent(board1, alpha, gamma, current_pos, epsilon)
    V, steps = agent.value_iteration()
    A, B, v = agent.bellman_eq()
    delta = V - v
    delta = delta.reshape((board1.n, board1.n))
    V = V.reshape((board1.n, board1.n))
    for i in range(V.shape[0]):
        line = ""
        for j in range(V.shape[1]):
            line += ' %5.3f ' % (V[i, j])
        print(line)
    print('step number = ', steps)
    board1.draw_board_with_num(V)


def test_policy_iteration():
    board1 = GameBoard()
    agent = Agent(board1, alpha, gamma, current_pos, epsilon)
    policy, V, steps = agent.policy_iteration()
    V = V.reshape((board1.n, board1.n))
    board1.draw_board_with_num(V)
    print(policy)
    print(steps)

    return policy


def test_tmp_difference(num_of_episodes):
    board1 = GameBoard()
    agent = Agent(board1, alpha, gamma, current_pos, epsilon)
    V, V_count = agent.tmp_difference(num_of_episodes)
    V = V.reshape((board1.n, board1.n))
    board1.draw_board_with_num(V)
    pygame.image.save(board1.screen, "temporal_differencing_results/stage_4-TD_10e6_values.jpg")
    V_count = V_count.reshape((board1.n, board1.n))
    board1.draw_board_with_num(V_count)
    pygame.image.save(board1.screen, "temporal_differencing_results/stage_4-TD_10e6_count.jpg")


def test_sarsa(num_of_episodes):
    board1 = GameBoard()
    agent = Agent(board1, alpha, gamma, current_pos, epsilon)
    q, init_q, greedy_policy, first_optimal_episode = agent.sarsa(num_of_episodes)
    print("the greedy policy: " + str(greedy_policy) + ", been discovered first at the: " + str(first_optimal_episode) + " episode")


def test_q_learning(num_of_episodes):
    board1 = GameBoard()
    agent = Agent(board1, alpha, lamda, gamma, current_pos, epsilon)
    array = np.copy(agent.q_learning(num_of_episodes))
    return array


def test_q_lamda(num_of_episodes):
    board1 = GameBoard()
    agent = Agent(board1, alpha, lamda, gamma, current_pos, epsilon)
    array = np.copy(agent.q_lamda(num_of_episodes))
    return array


def test_by_plot(array1, array2):
    a = min(len(array1), len(array2))
    array1 = array1[0:a]
    array2 = array2[0:a]
    df = pd.DataFrame(array1, columns=["Q-Learning"])
    df["Q-Lambda"] = array2
    df = df.reset_index()
    df = df.rename(columns={'index': 'episodes'})
    df = df.melt('episodes', var_name='algorithms', value_name='Q[1]-down')
    ax = sns.lineplot(x="episodes", y='Q[1]-down', hue='algorithms', data=df)
    return df


# test_bellman()
# test_value_iteration()
# test_policy_iteration()
# test_tmp_difference(NUM_OF_EPISODES)
# test_sarsa(NUM_OF_EPISODES)
# test_q_learning(NUM_OF_EPISODES)

array1 = test_q_learning(NUM_OF_EPISODES)
pygame.quit()

array2 = test_q_lamda(NUM_OF_EPISODES)
pygame.quit()

test_by_plot(array1, array2)
