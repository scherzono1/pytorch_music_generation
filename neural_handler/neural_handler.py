import numpy as np


class NeuralHandler:

    @staticmethod
    def get_batch(vectorized_songs, seq_length, batch_size):
        # the length of the vectorized songs string
        n = vectorized_songs.shape[0] - 1
        # randomly choose the starting indices for the examples in the training batch
        idx = np.random.choice(n - seq_length, batch_size)

        input_batch = [vectorized_songs[i: i + seq_length] for i in idx]
        output_batch = [vectorized_songs[i + 1: i + seq_length + 1] for i in idx]

        # x_batch, y_batch provide the true inputs and targets for network training
        x_batch = np.reshape(input_batch, [batch_size, seq_length])
        y_batch = np.reshape(output_batch, [batch_size, seq_length])
        return x_batch, y_batch
