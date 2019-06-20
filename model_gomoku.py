import pickle

import keras.backend as K
import numpy as np
from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.merge import Add
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.regularizers import l2

import configs

save_dir = './saved'
best_policy_path = save_dir + '/best_policy.model'
his_path = save_dir + '/model_history'


class GomokuModel:
    def __init__(self, model_file=None, model=None):
        self.model_config = configs.ModelConfig()
        self.board_config = configs.BoardConfig()
        if model:
            self.model = model
        else:
            self.model = self.build()

            # get the train op
            opt = Adam()
            # Three loss terms：
            # loss = (z - v)^2 + pi^T * log(p) + c||theta||^2
            losses = ['categorical_crossentropy', 'mean_squared_error']
            self.model.compile(optimizer=opt, loss=losses)

            if model_file:
                net_params = pickle.load(open(model_file, 'rb'))
                self.model.set_weights(net_params)

    def build(self):
        mc = self.model_config
        bc = self.board_config
        board_width = bc.width
        board_height = bc.height
        his_size = bc.his_size
        in_x = x = Input((2 * his_size + 3, board_width, board_height))
        x = Conv2D(filters=mc.cnn_filter_num,
                   kernel_size=mc.cnn_first_filter_size, padding="same",
                   data_format="channels_first", use_bias=False,
                   kernel_regularizer=l2(mc.l2_reg), name="input_conv-" + str(
                    mc.cnn_first_filter_size) + "-" + str(mc.cnn_filter_num))(
                x)
        x = BatchNormalization(axis=1, name="input_batchnorm")(x)
        x = Activation("relu", name="input_relu")(x)

        for i in range(mc.res_layer_num):
            x = self._build_residual_block(x, i + 1)

        res_out = x

        # for policy output
        x = Conv2D(filters=4, kernel_size=1, data_format="channels_first",
                   use_bias=False, kernel_regularizer=l2(mc.l2_reg),
                   name="policy_conv-1-2")(res_out)
        x = BatchNormalization(axis=1, name="policy_batchnorm")(x)
        x = Activation("relu", name="policy_relu")(x)
        x = Flatten(name="policy_flatten")(x)
        policy_out = Dense(board_width * board_height,
                           kernel_regularizer=l2(mc.l2_reg),
                           activation="softmax", name="policy_out")(x)

        # for value output
        x = Conv2D(filters=2, kernel_size=1, data_format="channels_first",
                   use_bias=False, kernel_regularizer=l2(mc.l2_reg),
                   name="value_conv-1-4")(res_out)
        x = BatchNormalization(axis=1, name="value_batchnorm")(x)
        x = Activation("relu", name="value_relu")(x)
        x = Flatten(name="value_flatten")(x)
        x = Dense(mc.value_fc_size, kernel_regularizer=l2(mc.l2_reg),
                  activation="relu", name="value_dense")(x)
        value_out = Dense(1, kernel_regularizer=l2(mc.l2_reg),
                          activation="tanh", name="value_out")(x)
        return Model(in_x, [policy_out, value_out], name="gomoku_model")

    def _build_residual_block(self, x, index):
        mc = self.model_config
        in_x = x
        res_name = "res" + str(index)
        x = Conv2D(filters=mc.cnn_filter_num, kernel_size=mc.cnn_filter_size,
                   padding="same",
                   data_format="channels_first", use_bias=False,
                   kernel_regularizer=l2(mc.l2_reg),
                   name=res_name + "_conv1-" + str(
                           mc.cnn_filter_size) + "-" + str(mc.cnn_filter_num))(
                x)
        x = BatchNormalization(axis=1, name=res_name + "_batchnorm1")(x)
        x = Activation("relu", name=res_name + "_relu1")(x)
        x = Conv2D(filters=mc.cnn_filter_num, kernel_size=mc.cnn_filter_size,
                   padding="same",
                   data_format="channels_first", use_bias=False,
                   kernel_regularizer=l2(mc.l2_reg),
                   name=res_name + "_conv2-" + str(
                           mc.cnn_filter_size) + "-" + str(mc.cnn_filter_num))(
                x)
        x = BatchNormalization(axis=1,
                               name="res" + str(index) + "_batchnorm2")(x)
        x = Add(name=res_name + "_add")([in_x, x])
        x = Activation("relu", name=res_name + "_relu2")(x)
        return x

    def policy_value(self, state_input):
        """
        :param state_input:
        :return: policy and value
        """
        state_input_union = np.array(state_input)
        results = self.model.predict_on_batch(state_input_union)
        return results

    def policy_value_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        bc = self.board_config
        board_width = bc.width
        board_height = bc.height
        his_size = bc.his_size
        legal_positions = board.available_positions
        current_state = board.current_state()
        # reshape -1 :The criterion to satisfy for providing the new shape
        # is that 'The new shape should be compatible with the original shape'
        act_probs, value = self.policy_value(
                current_state.reshape(-1, 2 * his_size + 3, board_width,
                                      board_height))
        act_probs = zip(legal_positions, act_probs.flatten()[legal_positions])
        return act_probs, value[0][0]

    def train_step(self, state_input, mcts_probs, winner, learning_rate):
        state_input_union = np.array(state_input)
        mcts_probs_union = np.array(mcts_probs)
        winner_union = np.array(winner)
        loss = self.model.evaluate(state_input_union,
                                   [mcts_probs_union, winner_union],
                                   batch_size=len(state_input), verbose=0)
        action_probs, _ = self.model.predict_on_batch(state_input_union)
        entropy = -np.mean(
                np.sum(action_probs * np.log(action_probs + 1e-10), axis=1))
        K.set_value(self.model.optimizer.lr, learning_rate)
        self.model.fit(state_input_union, [mcts_probs_union, winner_union],
                       validation_split=0.2, batch_size=len(state_input),
                       verbose=0)
        return loss[0], entropy


def save_model_history(history_path, history):
    with open(history_path, 'w+b') as file_pi:
        pickle.dump(history, file_pi)
