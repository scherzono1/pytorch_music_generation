import mitdeeplearning as mdl
import numpy as np


class MusicHandler:

    @staticmethod
    def load_songs():
        songs = mdl.lab1.load_training_data()
        return songs

    @staticmethod
    def join_songs(songs):
        return "\n\n".join(songs)

    @staticmethod
    def get_vocab(songs_joined):
        vocab = sorted(set(songs_joined))
        return vocab

    @staticmethod
    def get_char2idx(vocab):
        char2idx = {u: i for i, u in enumerate(vocab)}
        return char2idx

    @staticmethod
    def get_idx2char(vocab):
        idx2char = np.array(vocab)
        return idx2char

    @staticmethod
    def vectorize_string(songs_joined, char2idx):
        vectorized_output = np.array([char2idx[char] for char in songs_joined])
        return vectorized_output
