import os
import pickle
import sys
import numpy as np
from statistics import mean

import matplotlib.pyplot as plt
from keras.callbacks import History


# Model accuracy and loss plots
def plot_benchmark(benchmark):
    # Create sub-plots

    index = len(benchmark['selfplay'])

    plt.bar(np.arange(index) + 1, benchmark['selfplay'], color='orange')
    plt.bar(np.arange(index) + 1, benchmark['model_fit'], color='green')

    plt.title('Benchmark for one hour monitor with one process')
    plt.ylabel('Time (s)')
    plt.xlabel('Runs')
    plt.xticks(np.arange(index) + 1)
    plt.legend(['selfplay', 'training'], loc='best')
    plt.show()


if __name__ == '__main__':
    # if len(sys.argv) < 2:
    #     raise ValueError("Please specify the file to be benchmark.")
    file_path = 'saved/gomoku-board11x11x00-model128x03x03x01-train5000.benchmark'
    # file_path = 'saved/' + sys.argv[1]
    if not os.path.isfile(file_path):
        raise ValueError("File does not exist.")
    with open(file_path, 'rb') as file_pi:
        # model_history = History()
        # model_history.history = pickle.load(file_pi)
        plot_benchmark(pickle.load(file_pi))

# if __name__ == '__main__':
#     with open('saved/gomoku-board11x11x00-model128x03x03x01-train5000.benchmark', 'rb') as f1:
#         with open('saved/gomoku-board11x11x00-model128x03x03x01-train5000-mpi.benchmark', 'rb') as f2:
#             # model_history = History()
#             # model_history.history = pickle.load(file_pi)
#             b1 = pickle.load(f1)
#             b2 = pickle.load(f2)
#
#             avg_selfplay = mean(b1['selfplay'])
#             avg_training = mean(b1['model_fit'])
#             comm_time = (b2['selfplay'][0] + b2['model_fit'][0]) - (avg_selfplay + avg_training)
#
#             plt.bar(1, comm_time, color='red')
#             plt.bar(1, avg_selfplay, color='orange')
#             plt.bar(1, avg_training, color='green')
#
#             # plt.hist(benchmark['model_fit'], color=['orange'])
#             plt.title('Benchmark for one hour monitor with one process')
#             plt.ylabel('Time (s)')
#             plt.xlabel('Avg')
#             plt.legend(['communication','selfplay', 'training'], loc='best')
#             plt.show()
