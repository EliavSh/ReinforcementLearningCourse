# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 12:34:36 2020
Originsl code have been taken from https://github.com/kumararduino/Reinforcement-learning/blob/master/Q-Learning/git_simple_game.py
@author: AntonD
"""

import numpy as np
import pygame
# import matplotlib.pylab as pl
from gameboard import GameBoard
from agent import Agent

alpha = 0.01
gamma = 0.9
current_pos = [0, 0]
epsilon = 0.25


def testBellman():
    'Calculates and shows Bellmans Eq.'
    board1 = GameBoard()
    agent = Agent(board1, alpha, gamma, current_pos, epsilon)
    A, B, V = agent.BelamanEq()

    V = V.reshape((board1.n, board1.n))
    board1.drawboard_withnum(V)
    print("V Matrix")
    for i in range(V.shape[0]):
        line = ""
        for j in range(V.shape[1]):
            line += ' %5.3f ' % (V[i, j])
        print(line)

        # print ("A Matrix")
    # for i in range(A.shape[0]):
    #     line = ""
    #     for j in range(A.shape[1]):
    #         line += ' %5.3f ' % (A[i,j])
    #     print (line) 
    # line = ""
    # for i in range(len(B)):
    #     line += ' %5.3f ' % (B[i])
    #     print (B[i])               
    # print ('B = ')
    # print (line)          


def testValueIteration():
    board1 = GameBoard()
    agent = Agent(board1, alpha, gamma, current_pos, epsilon)
    V, steps = agent.value_iteration()
    A, B, v = agent.BelamanEq()
    delta = V - v
    delta = delta.reshape((board1.n, board1.n))
    V = V.reshape((board1.n, board1.n))
    for i in range(V.shape[0]):
        line = ""
        for j in range(V.shape[1]):
            line += ' %5.3f ' % (V[i, j])
        print(line)
    print('step number = ', steps)
    board1.drawboard_withnum(V)


def testPolicy_iteration():
    board1 = GameBoard()
    agent = Agent(board1, alpha, gamma, current_pos, epsilon)
    policy, V, steps = agent.policy_iteration()
    V = V.reshape((board1.n, board1.n))
    board1.drawboard_withnum(V)
    print(policy)
    print(steps)

    return policy


def test4():
    pass


testBellman()
# testValueIteration()
# tmp = testPolicy_iteration()
input()
pygame.quit()

# run = True
# while run:
#     # sleep(0.3)
#     screen.fill(background)
#     layout()
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             run = False
#     pygame.display.flip()
#     episode()
# pygame.quit()
