# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 12:34:36 2020
Original code have been taken from https://github.com/kumararduino/Reinforcement-learning/blob/master/Q-Learning/git_simple_game.py
@author: AntonD
"""
import seaborn as sns
import numpy as np
import pygame
# import matplotlib.pylab as pl
from gameboard import GameBoard
from agent import Agent

alpha = 0.01
gamma = 0.9
current_pos = 0
epsilon = 0.25
NUM_OF_EPISODES = 1000000


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
    V = agent.tmp_difference(num_of_episodes)
    V = V.reshape((board1.n, board1.n))
    board1.draw_board_with_num(V)
    pygame.image.save(board1.screen, "stage_4-TD.jpg")
    for i in range(V.shape[0]):
        line = ""
        for j in range(V.shape[1]):
            line += ' %5.3f ' % (V[i, j])
        print(line)


def test_sarsa(num_of_episodes):
    board1 = GameBoard()
    agent = Agent(board1, alpha, gamma, current_pos, epsilon)
    q, init_q = agent.sarsa(num_of_episodes)
    print(q)
    print(init_q)


"""
    # saving the tables as sns 'heatmap' with annotations
    sns.set()
    q_plot = sns.heatmap(Q, annot=True, fmt='.2f')
    figure = q_plot.get_figure()
    figure.savefig('Final_Q_table.png', dpi=400)
    init_q_plot = sns.heatmap(init_Q, annot=True, fmt='.2f')
    figure = init_q_plot.get_figure()
    figure.savefig('Initial_Q_table.png', dpi=400)
"""

# test_bellman()
# test_value_iteration()
# test_policy_iteration()
test_tmp_difference(NUM_OF_EPISODES)
# test_sarsa(NUM_OF_EPISODES)
input()
pygame.quit()
