class ModelConfig:
    cnn_filter_num = 128
    cnn_first_filter_size = 3
    cnn_filter_size = 3
    res_layer_num = 7
    l2_reg = 1e-4
    value_fc_size = 128


class BoardConfig:
    width = 19
    height = 19
    n_in_row = 5
    his_size = 5
    first_player = 0


class MCTSConfig:
    """
    n_playout: num of simulations for each move
    c_put: a number in (0, inf) that controls how quickly exploration
    converges to the maximum-value policy. A higher value means relying on
    the prior more.
    temp: temperature parameter in (0, 1] controls the level of exploration
    """
    n_playout = 1600
    c_put = 5
    # with the default temp=1e-3, it is almost equivalent
    # to choosing the move with the highest prob
    temperature = 1e-3


class TrainConfig:
    game_batch_num = 10000
    play_batch_size = 1
    # mini-batch size for training
    batch_size = 512
    # num of train_steps for each updateh
    epochs = 5
    # adaptively adjust the learning rate based on KL
    learn_rate = 1e-3
    lr_multiplier = 1.0
    kl_targ = 0.025
    check_freq = 1000
    evaluate_match_num = 100
