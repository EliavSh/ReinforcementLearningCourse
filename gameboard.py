# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 12:34:36 2020
Original code have been taken from https://github.com/kumararduino/Reinforcement-learning/blob/master/Q-Learning/git_simple_game.py
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
        # self.random_board()
        self.set_board1()
        self._reward = self._set_reward2()

    def random_board(self):
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

    def set_board1(self):
        """this board created specially for 7X7 grid"""
        if self.n != 7:
            self.n = 7
            self.scrx = self.n * 100  # cell sizes
            self.scry = self.n * 100  # cell sizes
            pygame.quit()
            self.screen = pygame.display.set_mode((self.scrx, self.scry))  # creating a screen using Pygame
            self.colors = [(224, 224, 224) for i in range(self.n ** 2)]

        self.terminals = [3, 7, 15, 16, 30, 31, 32, 33, 34, 47, 48]
        for terminal in self.terminals:
            self.colors[terminal] = (255, 0, 0)
        self.colors[0] = (255, 255, 0)  # start point
        return True

    def load_board(self):
        pass

    def _set_reward2(self):
        """reward = 20 when find final location
           reward = -3 when the player hit one of the internal walls/terminals
           reward = -1 for any other step"""
        reward = np.ones(self.n ** 2) * (-1)
        for terminal in self.terminals:
            reward[terminal] = -3
        reward[48] = 20
        return reward

    def get_reward(self, state):
        """return reward value at state: 'state' """
        return self._reward[state]

    def get_actions(self, state):
        possible_actions = []
        if state > 6:
            possible_actions.append((self.actions['up'], state - 7))
        if state < 42:
            possible_actions.append((self.actions['down'], state + 7))
        if state % self.n != 0:
            possible_actions.append((self.actions['left'], state - 1))
        if state % self.n != 6:
            possible_actions.append((self.actions['right'], state + 1))
        return possible_actions

    def draw_layout(self, current_pos, delay=0):
        self.screen.fill((51, 51, 51))  # background
        c = 0
        for i in range(0, self.scrx, 100):
            for j in range(0, self.scry, 100):
                pygame.draw.rect(self.screen, self.colors[c], (j + 2, i + 2, 96, 96), 0)
                c += 1
        pygame.draw.circle(self.screen, (25, 129, 230), (current_pos[1] * 100 + 50, current_pos[0] * 100 + 50), 30, 0)
        pygame.time.delay(delay)
        pygame.display.flip()

    def draw_board_with_num(self, nums):
        font = pygame.font.SysFont('Arial', 25)
        self.screen.fill((51, 51, 51))  # background
        c = 0
        for i in range(0, self.scrx, 100):
            for j in range(0, self.scry, 100):
                pygame.draw.rect(self.screen, self.colors[c], (j + 2, i + 2, 96, 96), 0)
                c += 1
                self.screen.blit(
                    font.render('%5.3f' % (nums[int(i / 100), int(j / 100) % self.n]), True, (51, 25, 0)),
                    (j + 15, i + 35))
        pygame.display.update()


if __name__ == "__main__":
    board1 = GameBoard()
    board1.draw_layout([0, 0])
    board1.draw_board_with_num(np.arange(49).reshape((7, 7)))
    # board1.draw_board_with_num('a')
    input()
    pygame.quit()
