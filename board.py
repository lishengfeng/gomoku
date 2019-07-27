from collections import OrderedDict

import numpy as np

import configs


class Board:
    def __init__(self):
        self.board_config = configs.BoardConfig()
        bc = self.board_config
        width = bc.width
        height = bc.height
        n_in_row = bc.n_in_row
        first_player = bc.first_player
        his_size = bc.his_size
        if width < n_in_row or height < n_in_row:
            raise Exception(
                    "board width and height cannot be "
                    "less than {}".format(n_in_row))
        self.width = width
        self.height = height
        self.states = OrderedDict()
        self.n_in_row = n_in_row
        # Two players, first player and second player
        self.players = [1, 2]
        self.current_player = self.players[first_player]
        # Store all the available positions
        self.available_positions = list(range(width * height))
        self.his_size = his_size
        # Both current player and opponent's history
        # The 1 extra is used to store current move
        self.last_n_move = [-1] * (his_size * 2 + 1)

    def init_board(self):
        bc = configs.BoardConfig()
        first_player = bc.first_player
        self.current_player = self.players[first_player]
        width = bc.width
        height = bc.height
        self.available_positions = list(range(width * height))
        his_size = bc.his_size
        self.last_n_move = [-1] * (his_size * 2 + 1)
        self.states = OrderedDict()

    def position_to_place(self, move):
        """
        3*3 board's moves like:
        6 7 8
        3 4 5
        0 1 2
        and move 5's location is (1,2)
        """
        h = move // self.width
        w = move % self.width
        return [h, w]

    def location_to_move(self, location):
        """Convert location to move
        5 -> (1,2)
        """
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move

    def current_state(self):
        """return the board state from the perspective of the current player.
        s(t) = [X(t), Y(t), X(t-1), Y(t-1), ..., X(t-n), Y(t-n), C]
        X is the current player, Y is the opponent, C is the color
        X(t-n) is the historical move of current player as well as Y(t-n)
        """
        his_size = self.his_size
        # 1 feature map for locations possessed by play 1
        # 1 feature map for locations possessed by play 2
        # 1 feature map for whether current player is the first player or not
        # several feature maps represent the history of play 1 and player 2
        square_state = np.zeros((2 * his_size + 3, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            square_state[0][move_curr // self.width,
                            move_curr % self.height] = 1.0
            square_state[1][move_oppo // self.width,
                            move_oppo % self.height] = 1.0
            # indicate the last moves location
            # The first element is the opponent's cur move
            # (player has already been switched)
            last_n_move = self.last_n_move[1:]
            for idx, val in enumerate(last_n_move):
                if val < 0:
                    continue
                else:
                    square_state[idx + 2][val // self.width,
                                          val % self.height] = 1.0
        if len(self.states) % 2 == 0:
            square_state[-1][:, :] = 1.0  # indicate the colour to play
        # The step using -1 is that the python y-axis is top-down
        # while our y-axis is down-top
        return square_state[:, ::-1, :]

    def do_move(self, move):
        self.states[move] = self.current_player
        # save current move
        self.last_n_move.insert(0, move)
        # remove last history
        self.last_n_move = self.last_n_move[:-1]
        self.available_positions.remove(move)
        # Switch current player
        self.current_player = (
            self.players[0] if self.current_player == self.players[1]
            else self.players[1]
        )

    def has_a_winner(self):
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        moved = list(
                set(range(width * height)) - set(self.available_positions))
        # one needs to reach n to make it win
        # (loser reaches at least (n -1)).
        if len(moved) < self.n_in_row * 2 - 1:
            return False, -1

        for m in moved:
            position = self.position_to_place(m)
            h = position[0]
            w = position[1]
            player = states[m]

            # Horizontal n in a row
            # First condition: at least n available locations to place stones
            # Section condition: all possession belongs to one player
            # Otherwise, another player or -1(empty location) makes set sizes
            # bigger than one
            if (w in range(width - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player

            # Vertical n in a row
            if (h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in
                            range(m, m + n * width, width))) == 1):
                return True, player
            # Forward slash diagonal
            # - - - - O
            # - - - O -
            # - - O - -
            # - O - - -
            # O - - - - (O -> start point)
            if (w in range(width - n + 1) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in
                            range(m, m + n * (width + 1), width + 1))) == 1):
                return True, player

            # Backslash diagonal
            # O - - - -
            # - O - - -
            # - - O - -
            # - - - O -
            # - - - - O (O -> start point)
            if (w in range(n - 1, width) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in
                            range(m, m + n * (width - 1), width - 1))) == 1):
                return True, player

        return False, -1

    def game_end(self):
        """Check whether the game is ended or not"""
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.available_positions):
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player
