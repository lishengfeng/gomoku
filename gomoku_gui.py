# -*- coding: utf-8 -*-
import os
import sys

import numpy as np
import pygame


class GomokuGUI:
    def __init__(self, max_steps=11, human_color=1, fps=3):

        # color, white for player 1, black for player -1
        self.white = (255, 255, 255)
        self.black = (0, 0, 0)
        self.green = (0, 255, 0)

        # screen
        self.width = 800
        self.height = 800

        self.max_steps = max_steps
        self.grid_width = self.width / (max_steps + 3)
        self.fps = fps

        # human color
        self.human_color = human_color
        self.board = np.zeros((self.max_steps, self.max_steps), dtype=int)
        self.number = np.zeros((self.max_steps, self.max_steps), dtype=int)
        self.k = 1  # step number

        self.is_human = False
        self.human_move = -1

        # set running
        self.is_running = True

        # init
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Gomoku")

        # timer
        self.clock = pygame.time.Clock()

        # background image
        base_folder = os.path.dirname(__file__)
        self.background_img = pygame.image.load(
                os.path.join(base_folder, './assets/background.png')).convert()

        # font
        self.font_size = 16
        self.font = pygame.font.SysFont('Arial', self.font_size)

    def __del__(self):
        # close window
        self.is_running = False

    # reset status
    def reset_status(self):
        self.board = np.zeros((self.max_steps, self.max_steps), dtype=int)
        self.number = np.zeros((self.max_steps, self.max_steps), dtype=int)
        self.k = 1  # step number

        self.is_human = False
        self.human_move = -1

    # human play
    def set_is_human(self, value=True):
        self.is_human = value

    def get_is_human(self):
        return self.is_human

    def get_human_move(self):
        return self.human_move

    def get_human_color(self):
        return self.human_color

    # execute move
    def execute_move(self, color, move):
        x, y = move // self.max_steps, move % self.max_steps
        assert self.board[x][y] == 0

        self.board[x][y] = color
        self.number[x][y] = self.k
        self.k += 1

    # main loop
    def loop(self):
        while self.is_running:
            # timer
            self.clock.tick(self.fps)

            # handle event
            for event in pygame.event.get():
                # close window
                if event.type == pygame.QUIT:
                    self.is_running = False
                    sys.exit()

                # human play
                if self.is_human and event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_y, mouse_x = event.pos
                    position = (int(mouse_x / self.grid_width + 0.5) - 2,
                                int(mouse_y / self.grid_width + 0.5) - 2)

                    if position[0] in range(0, self.max_steps) and position[1] in range(0, self.max_steps) \
                            and self.board[position[0]][position[1]] == 0:
                        self.human_move = position[0] * self.max_steps + position[1]
                        self.execute_move(self.human_color, self.human_move)
                        self.set_is_human(False)

            # draw
            self._draw_background()
            self._draw_chessman()

            # refresh
            pygame.display.flip()

    def _draw_background(self):
        # load background
        self.screen.blit(self.background_img, (0, 0))

        # draw lines
        rect_lines = [
            ((self.grid_width, self.grid_width),
             (self.grid_width, self.height - self.grid_width)),
            ((self.grid_width, self.grid_width), (self.width - self.grid_width,
                                                  self.grid_width)),
            ((self.grid_width, self.height - self.grid_width),
             (self.width - self.grid_width, self.height - self.grid_width)),
            ((self.width - self.grid_width, self.grid_width),
             (self.width - self.grid_width, self.height - self.grid_width)),
        ]
        for line in rect_lines:
            pygame.draw.line(self.screen, self.black, line[0], line[1], 2)

        # draw grid
        for i in range(self.max_steps):
            pygame.draw.line(
                    self.screen, self.black,
                    (self.grid_width * (2 + i), self.grid_width),
                    (self.grid_width * (2 + i), self.height - self.grid_width))
            pygame.draw.line(
                    self.screen, self.black,
                    (self.grid_width, self.grid_width * (2 + i)),
                    (self.height - self.grid_width, self.grid_width * (2 + i)))

    def _draw_chessman(self):
        # draw chessmen
        for i in range(self.max_steps):
            for j in range(self.max_steps):
                if self.board[i][j] != 0:
                    # circle
                    position = (int(self.grid_width * (j + 2)),
                                int(self.grid_width * (i + 2)))
                    color = self.white if self.board[i][j] == 1 else self.black
                    pygame.draw.circle(self.screen, color, position,
                                       int(self.grid_width / 2.3))
                    # text
                    p_h_offset = self.font_size // 2 if self.number[i][j] > 9 else self.font_size // 4
                    position = (position[0] - p_h_offset, position[1] - self.font_size // 2)
                    color = self.white if self.board[i][j] == -1 else self.black
                    text = self.font.render(str(self.number[i][j]), 3, color)
                    self.screen.blit(text, position)


