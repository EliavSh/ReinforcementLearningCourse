# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 12:34:36 2020
Originsl code have been taken from https://github.com/kumararduino/Reinforcement-learning/blob/master/Q-Learning/git_simple_game.py
@author: AntonD
"""

import numpy as np
import pygame
from random import randint as rand


class GameBoard:
    def __init__(self, n=7):

        self.n = n  # represents no. of side squares(n*n total squares)

        self.actions = {"up": 0, "down": 1, "left": 2, "right": 3}  # possible actions

        self.terminals = []
        self.penalties = 10

        # layout variables
        pygame.init()
        self.scrx = self.n * 100  # cell sizes
        self.scry = self.n * 100  # cell sizes
        self.screen = pygame.display.set_mode((self.scrx, self.scry))  # creating a screen using Pygame
        self.colors = [(224, 224, 224) for i in range(self.n ** 2)]
        self.colors[0] = (255, 255, 0)  # start point
        self.colors[self.n ** 2 - 1] = (0, 255, 0)  # finish point

        # initialize board
        # self.randomboard()
        self.setboard1()
        self._reward = self._setreward2()

    def postostate(self, i, j):
        return (self.n * i + j)

    def statetopos(self, state):
        return (int(state / self.env.n), state % self.n)

    def randomboard(self):
        penalties = self.penalties
        while penalties != 0:
            i = rand(0, self.n - 1)
            j = rand(0, self.n - 1)
            if (((self.n * i + j) not in self.terminals) and
                    ([i, j] != [0, 0]) and
                    ([i, j] != [self.n - 1, self.n - 1])):
                penalties -= 1
                self.terminals.append(self.n * i + j)
                self.colors[self.n * i + j] = (255, 0, 0)
        self.terminals.append(self.n ** 2 - 1)
        return True

    def setboard1(self):
        'this board created specialy for 7X7 grid'
        if self.n != 7:
            self.n = 7
            self.scrx = self.n * 100  # cell sizes
            self.scry = self.n * 100  # cell sizes
            pygame.quit()
            self.screen = pygame.display.set_mode((self.scrx, self.scry))  # creating a screen using Pygame
            self.colors = [(224, 224, 224) for i in range(self.n ** 2)]
            self.colors[0] = (255, 255, 0)  # start point
            self.colors[self.n ** 2 - 1] = (0, 255, 0)  # finish point

        self.terminals = [3, 7, 15, 16, 30, 31, 32, 33, 34, 47]
        for terminal in self.terminals:
            self.colors[terminal] = (255, 0, 0)
        return True

    def loadboard(self):
        pass

    def _setreward(self):
        '''reward = 1 when access finish
           reward = -1 when the player hit one of the internal walls/terminals '''
        reward = np.zeros((self.n, self.n))
        for terminal in self.terminals:
            reward[int(terminal / self.n), terminal % self.n] = -1
        reward[self.n - 1, self.n - 1] = 1
        return reward

    def _setreward2(self):
        '''reward = 20 when find final location
           reward = -3 when the player hit one of the internal walls/terminals
           rewrd = -1 for any other step'''
        reward = np.ones((self.n, self.n)) * (-1)
        for terminal in self.terminals:
            reward[int(terminal / self.n), terminal % self.n] = -3
        reward[self.n - 1, self.n - 1] = 20
        return reward

    def getreward(self, i, j):
        '''return reward value at location [i,j]'''
        if (i >= 0 and j >= 0 and i < self.n and j < self.n):
            return self._reward[i, j]
        else:  # you can return -1 if you want to punish for hiting the wall
            return (False)

    def get_actions(self, i, j):
        possible_actions = []
        if i > 0:
            possible_actions.append((self.actions['up'], self.postostate(i - 1, j)))
        if i < self.n - 1:
            possible_actions.append((self.actions['down'], self.postostate(i + 1, j)))
        if j > 0:
            possible_actions.append((self.actions['left'], self.postostate(i, j - 1)))
        if j < self.n - 1:
            possible_actions.append((self.actions['right'], self.postostate(i, j + 1)))
        return possible_actions

    def drawlayout(self, current_pos, delay=0):
        self.screen.fill((51, 51, 51))  # background
        c = 0
        for i in range(0, self.scrx, 100):
            for j in range(0, self.scry, 100):
                pygame.draw.rect(self.screen, self.colors[c], (j + 2, i + 2, 96, 96), 0)
                c += 1
        pygame.draw.circle(self.screen, (25, 129, 230), (current_pos[1] * 100 + 50, current_pos[0] * 100 + 50), 30, 0)
        pygame.time.delay(delay)
        pygame.display.flip()

    def drawboard_withnum(self, nums):
        font = pygame.font.SysFont('Arial', 25)
        self.screen.fill((51, 51, 51))  # background
        c = 0
        for i in range(0, self.scrx, 100):
            for j in range(0, self.scry, 100):
                pygame.draw.rect(self.screen, self.colors[c], (j + 2, i + 2, 96, 96), 0)
                c += 1
                self.screen.blit(font.render('%5.3f' % (nums[int(i / 100), int(j / 100)]), True, (51, 25, 0)),
                                 (j + 15, i + 35))
        pygame.display.flip()


if __name__ == "__main__":
    board1 = GameBoard()
    board1.drawlayout([0, 0])
    board1.drawboard_withnum(np.arange(49).reshape((7, 7)))
    # board1.drawboard_withnum('a')
    input()
    pygame.quit()
