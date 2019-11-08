import os
import pickle
import sys

import matplotlib.pyplot as plt
from keras.callbacks import History


# Model accuracy and loss plots
def plot_model_history(model_details):
    # Create sub-plots

    his_size = len(model_details.history)
    loss = []
    # Summarize history for accuracy
    for his in model_details.history:
        loss.append(his['loss'])
    plt.plot(range(1, his_size + 1), loss)
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='best')
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise ValueError("Please specify the file to be plotted.")
    file_path = 'saved/' + sys.argv[1]
    if not os.path.isfile(file_path):
        raise ValueError("File does not exist.")
    with open(file_path, 'rb') as file_pi:
        model_history = History()
        model_history.history = pickle.load(file_pi)
        plot_model_history(model_history)

