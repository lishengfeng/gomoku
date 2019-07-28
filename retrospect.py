# -*- coding: utf-8 -*-
import os
import pickle
import re
import sys
import threading
from time import sleep

from gomoku_gui import GomokuGUI

if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise ValueError("Please specify the file to be retrospected.")
    file_path = 'saved/' + sys.argv[1]
    if not os.path.isfile(file_path):
        raise ValueError("File does not exist.")
    board_info = re.search(r'gomoku-board(.*?)-.*\.states', file_path).group(1)
    if board_info is None:
        raise ValueError("File name is invalid.")
    board_info_list = board_info.split('x')
    b_width = int(board_info_list[0])
    gomoku_gui = GomokuGUI(b_width)
    t = threading.Thread(target=gomoku_gui.loop)
    t.start()
    selfplay_state_buffer = pickle.load(open(file_path, 'rb'))
    for selfplay_state in selfplay_state_buffer:
        color = 1
        # TODO change to self_state[:-1] last one is winner
        for step in selfplay_state:
            color = 1 if color == -1 else -1
            gomoku_gui.execute_move(color, step)
            sleep(0.5)
        sleep(3)
        gomoku_gui.reset_status()
