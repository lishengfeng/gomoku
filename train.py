import os.path
import pickle
import random
from collections import defaultdict, deque

import sys
import numpy as np
from keras.models import clone_model

import callbacks as cbks
from board import Board
from configs import TrainConfig, BoardConfig, FilepathConfig
from game import Game
from mcts import MCTSPlayer
from model_gomoku import GomokuModel
import time


class Train:
    def __init__(self, is_mpi=False):
        if is_mpi:
            global MPI
            mpi4py = __import__('mpi4py.MPI', globals(), locals())
            MPI = mpi4py.MPI
        self.is_mpi = is_mpi

        self.filepath_config = FilepathConfig(is_mpi)
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
        self.model_gomoku = GomokuModel(is_mpi)
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
        self.remaining_game_batch = self.game_batch_num - self.start_batch

        benchmark_file = '{}.benchmark'.format(filepath)
        if os.path.isfile(benchmark_file):
            benchmark_state = pickle.load(open(benchmark_file, 'rb'))
            self.time_selfplay = benchmark_state['selfplay']
            self.time_model_fit = benchmark_state['model_fit']
            self.time_evaluate = benchmark_state['evaluate']
        else:
            self.time_selfplay = []
            self.time_model_fit = []
            self.time_evaluate = []

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

    def collect_selfplay_data(self, selfplay_per_iter=1):
        """collect self-play data for training"""
        data_buffer = deque()
        selfplay_state_buffer = deque()
        for i in range(selfplay_per_iter):
            winner, play_data, state_his = self.game.start_self_play(self.mcts_player, is_shown=False, is_recorded=True)
            play_data = list(play_data)[:]
            # self.episode_len = len(play_data)
            # augment the data
            play_data = self.get_equi_data(play_data)
            data_buffer.extend(play_data)
            selfplay_state_buffer.append(state_his)
        return data_buffer, selfplay_state_buffer

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
        return loss, entropy

    def policy_evaluate(self):
        """
        Evaluate the trained policy by playing against the previous MCTS player
        Note: this is only for monitoring the progress of training
        """
        n_games = self.config.evaluate_match_num
        win_ratio = 1.0
        if self.previous_model is not None:
            previous_model = GomokuModel(model=self.previous_model, is_mpi=self.is_mpi)
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

    def save_benchmark_state(self):
        filepath = '{}.benchmark'.format(self.filepath_config.filepath)
        benchmark_state = {'selfplay': self.time_selfplay, 'model_fit': self.time_model_fit,
                           'evaluate': self.time_evaluate}
        pickle.dump(benchmark_state, open(filepath, 'wb'), protocol=2)

    def sync_model(self):
        comm = MPI.COMM_WORLD
        if comm.rank == 0:
            if self.remaining_game_batch > 0:
                weights = self.model_gomoku.model.get_weights()
            else:
                weights = None
        else:
            weights = None
        weights = comm.bcast(weights, root=0)
        if weights is None:
            print("Process %d exit." % comm.rank)
            sys.exit()
        if comm.rank != 0:
            self.model_gomoku.model.set_weights(weights)

    def run(self):
        """ Start training
        """
        model_checkpoint = cbks.ModelCheckpoint(is_mpi=self.is_mpi)

        if self.is_mpi:
            comm = MPI.COMM_WORLD
            while True:
                self.sync_model()
                selfplay_start_time = 0
                if comm.rank == 0:
                    selfplay_start_time = time.time()
                # All processes including root starts to selfplay
                data_buffer, selfplay_state_buffer = self.collect_selfplay_data(self.selfplay_per_iter)
                selfplay_data = {'data_buffer': data_buffer, 'selfplay_state_buffer': selfplay_state_buffer}
                selfplay_data_list = comm.gather(selfplay_data, root=0)
                if comm.rank == 0:
                    self.time_selfplay.append(time.time() - selfplay_start_time)
                    # self.time_selfplay += time.time() - selfplay_start_time
                    self.remaining_game_batch -= len(selfplay_data_list)
                    cur_i = self.game_batch_num - self.remaining_game_batch
                    for sd in selfplay_data_list:
                        self.data_buffer.extend(sd['data_buffer'])
                        self.selfplay_state_buffer.extend(sd['selfplay_state_buffer'])
                    if len(self.data_buffer) > self.batch_size:
                        model_fit_start_time = time.time()
                        self.policy_update()
                        self.time_model_fit.append(time.time() - model_fit_start_time)
                        # self.time_model_fit += time.time() - model_fit_start_time
                        self.save_session_state(cur_i)
                        self.save_training_history()
                        print('current batch: ' + str(cur_i))
                    # check the performance of the current model,
                    # and save the model params
                    if cur_i % self.check_freq == 0:
                        evaluate_start_time = time.time()
                        win_ratio = self.policy_evaluate()
                        self.time_evaluate.append(time.time() - evaluate_start_time)
                        # self.time_evaluate += time.time() - evaluate_start_time
                        if win_ratio > 0.5 or self.previous_model is None:
                            model = self.model_gomoku.model
                            self.previous_model = clone_model(model)
                            self.previous_model.set_weights(model.get_weights())
                            # update the best_policy
                            self.model_callback(cur_i, [model_checkpoint])
                            self.save_states()
                    self.save_benchmark_state()
                comm.Barrier()
        else:
            for i in range(self.start_batch, self.game_batch_num):
                selfplay_start_time = time.time()
                data_buffer, selfplay_state_buffer = self.collect_selfplay_data(self.selfplay_per_iter)
                self.data_buffer.extend(data_buffer)
                self.selfplay_state_buffer.extend(selfplay_state_buffer)
                self.time_selfplay.append(time.time() - selfplay_start_time)
                # self.time_selfplay += time.time() - selfplay_start_time

                if len(self.data_buffer) > self.batch_size:
                    model_fit_start_time = time.time()
                    self.policy_update()
                    self.time_model_fit.append(time.time() - model_fit_start_time)
                    # self.time_model_fit += time.time() - model_fit_start_time
                    self.save_session_state(i + 1)
                    self.save_training_history()
                    print('current batch: ' + str(i))
                # check the performance of the current model,
                # and save the model params
                if (i + 1) % self.check_freq == 0:
                    evaluate_start_time = time.time()
                    win_ratio = self.policy_evaluate()
                    self.time_evaluate.append(time.time() - evaluate_start_time)
                    # self.time_evaluate += time.time() - evaluate_start_time
                    if win_ratio > 0.5 or self.previous_model is None:
                        model = self.model_gomoku.model
                        self.previous_model = clone_model(model)
                        self.previous_model.set_weights(model.get_weights())
                        # update the best_policy
                        self.model_callback(i, [model_checkpoint])
                        self.save_states()
                self.save_benchmark_state()


if __name__ == '__main__':
    train = Train()
    train.run()
