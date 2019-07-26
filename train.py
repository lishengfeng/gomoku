import os.path
import pickle
import random
from collections import defaultdict, deque

import numpy as np
from keras.models import clone_model

import callbacks as cbks
from board import Board
from configs import TrainConfig, BoardConfig, FilepathConfig
from game import Game
from mcts import MCTSPlayer
from model_gomoku import GomokuModel


class Train:
    def __init__(self):
        self.filepath_config = FilepathConfig()
        self.config = TrainConfig()
        self.board_config = BoardConfig()
        self.board_width = self.board_config.width
        self.board_height = self.board_config.height
        self.game_batch_num = self.config.game_batch_num
        self.selfplay_per_iter = self.config.selfplay_per_iter
        # adaptively adjust the learning rate based on KL
        self.learn_rate = self.config.learn_rate
        self.lr_multiplier = self.config.lr_multiplier
        self.epochs = self.config.epochs
        self.kl_targ = self.config.kl_targ
        # mini-batch size for training
        self.batch_size = self.config.batch_size
        # check frequency
        self.check_freq = self.config.check_freq
        # deque list-like container with fast appends and pops on either end
        self.data_buffer = deque()
        self.board = Board()
        self.game = Game(self.board)
        # number of states when one episode ends (winner appears)
        self.episode_len = 0
        self.model_gomoku = GomokuModel()
        self.mcts_player = MCTSPlayer(self.model_gomoku.policy_value_fn,
                                      is_selfplay=True)
        self.previous_model = None

        filepath = self.filepath_config.filepath

        selfplay_states_file = '{}.selfplay.states'.format(filepath)
        if os.path.isfile(selfplay_states_file):
            self.selfplay_state_buffer = pickle.load(open(selfplay_states_file, 'rb'))
        else:
            self.selfplay_state_buffer = deque()

        evalplay_states_file = '{}.evalplay.states'.format(filepath)
        if os.path.isfile(evalplay_states_file):
            self.evalplay_state_buffer = pickle.load(open(evalplay_states_file, 'rb'))
        else:
            self.evalplay_state_buffer = deque()

        history_file = '{}.history'.format(filepath)
        if os.path.isfile(history_file):
            self.history_buffer = pickle.load(open(history_file, 'rb'))
        else:
            self.history_buffer = deque()

        session_file = '{}.session'.format(filepath)
        if os.path.isfile(session_file):
            session_state = pickle.load(open(session_file, 'rb'))
            self.start_batch = session_state['batch']
        else:
            self.start_batch = 0

    def get_equi_data(self, play_data):
        """augment the data set by rotation and flipping
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        """
        extend_data = []
        for state, mcts_probs, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                        mcts_probs.reshape(self.board_height,
                                           self.board_width)), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
        return extend_data

    def collect_selfplay_data(self, selfplay_per_iter):
        """collect self-play data for training"""
        for i in range(selfplay_per_iter):
            winner, play_data, state_his = self.game.start_self_play(self.mcts_player, is_shown=False)
            play_data = list(play_data)[:]
            # self.episode_len = len(play_data)
            # augment the data
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)
            self.selfplay_state_buffer.append(state_his)

    def policy_update(self):
        """update the policy-value net"""
        kl = 0
        loss = 0
        entropy = 0
        new_v = 0
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.model_gomoku.policy_value(state_batch)
        for i in range(self.epochs):
            loss, entropy = self.model_gomoku.train_step(
                    state_batch, mcts_probs_batch, winner_batch,
                    self.learn_rate * self.lr_multiplier)
            new_probs, new_v = self.model_gomoku.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                                axis=1)
                         )
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             (np.var(np.array(winner_batch)) or 1))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             (np.var(np.array(winner_batch)) or 1))
        his_record = {'kl': kl, 'lr_multiplier': self.lr_multiplier, 'loss': loss, 'entropy': entropy,
                      'explained_var_old': explained_var_old, 'explained_var_new': explained_var_new}
        self.history_buffer.append(his_record)
        # print(("kl:{:.5f},"
        #        "lr_multiplier:{:.3f},"
        #        "loss:{},"
        #        "entropy:{},"
        #        "explained_var_old:{:.3f},"
        #        "explained_var_new:{:.3f}"
        #        ).format(kl,
        #                 self.lr_multiplier,
        #                 loss,
        #                 entropy,
        #                 explained_var_old,
        #                 explained_var_new))
        return loss, entropy

    def policy_evaluate(self):
        """
        Evaluate the trained policy by playing against the previous MCTS player
        Note: this is only for monitoring the progress of training
        """
        n_games = self.config.evaluate_match_num
        win_ratio = 1.0
        if self.previous_model is not None:
            previous_model = GomokuModel(model=self.previous_model)
            previous_player = MCTSPlayer(previous_model.policy_value_fn)
            current_mcts_player = MCTSPlayer(self.model_gomoku.policy_value_fn)
            win_cnt = defaultdict(int)
            for i in range(n_games):
                winner, state_his = self.game.start_play(current_mcts_player,
                                                         previous_player,
                                                         start_player=i % 2,
                                                         is_shown=False, is_recorded=True)
                self.evalplay_state_buffer.append(state_his)
                win_cnt[winner] += 1
            # win_cnt[-1] is tie.
            win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[-1]) / n_games
            # print("Current vs Previous, win: {}, lose: {}, tie:{}".format(
            #         win_cnt[1], win_cnt[2], win_cnt[-1]))
        return win_ratio

    def model_callback(self, batch, callbacks=None):
        _callbacks = []
        _callbacks += (callbacks or [])
        callbacks = cbks.CallbackList(_callbacks)
        callbacks.set_model(self.model_gomoku)
        callbacks.on_module_updated(batch)

    def save_states(self):
        selfplay_filepath = '{}.selfplay.states'.format(self.filepath_config.filepath)
        pickle.dump(self.selfplay_state_buffer, open(selfplay_filepath, 'wb'), protocol=2)
        evalplay_filepath = '{}.evalplay.states'.format(self.filepath_config.filepath)
        pickle.dump(self.evalplay_state_buffer, open(evalplay_filepath, 'wb'), protocol=2)

    def save_training_history(self):
        filepath = '{}.history'.format(self.filepath_config.filepath)
        pickle.dump(self.history_buffer, open(filepath, 'wb'), protocol=2)

    def save_session_state(self, batch):
        filepath = '{}.session'.format(self.filepath_config.filepath)
        session_state = {'batch': batch}
        pickle.dump(session_state, open(filepath, 'wb'), protocol=2)

    def run(self):
        """ Start training
        """
        model_checkpoint = cbks.ModelCheckpoint()
        for i in range(self.start_batch, self.game_batch_num):
            self.collect_selfplay_data(self.selfplay_per_iter)
            # print("batch i:{}, episode_len:{}".format(
            #         i + 1, self.episode_len))
            if len(self.data_buffer) > self.batch_size:
                self.policy_update()
                self.save_session_state(i + 1)
                self.save_training_history()
                print('current batch: ' + str(i))
                # loss, entropy = self.policy_update()
                # losses.append(loss)
            # check the performance of the current model,
            # and save the model params
            if (i + 1) % self.check_freq == 0:
                # print("current self-play batch: {}".format(i + 1))
                win_ratio = self.policy_evaluate()
                # history = {'loss': losses}
                # save_model_history(his_path, history)
                if win_ratio > 0.5 or self.previous_model is None:
                    model = self.model_gomoku.model
                    self.previous_model = clone_model(model)
                    self.previous_model.set_weights(model.get_weights())
                    # print("New best policy!!!!!!!!")
                    # update the best_policy
                    self.model_callback(i, [model_checkpoint])
                    self.save_states()
                    # self.model_gomoku.save_model(best_policy_path)


if __name__ == '__main__':
    train = Train()
    train.run()
