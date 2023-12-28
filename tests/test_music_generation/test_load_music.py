import torch

from music.music_handler import MusicHandler
from neural_handler.neural_handler import NeuralHandler
from neural_handler.rnn_lstm_model import RnnLSTMModel


def test_load_data():

    # Download the dataset
    songs = MusicHandler.load_songs()

    # Join our list of song strings into a single string containing all songs
    songs_joined = MusicHandler.join_songs(songs)

    # Find all unique characters in the joined string
    vocab = MusicHandler.get_vocab(songs_joined)

    # Map every unique character to a unique integer
    char2idx = MusicHandler.get_char2idx(vocab)
    idx2char = MusicHandler.get_idx2char(vocab)

    # Vectorize the songs string
    vectorized_songs = MusicHandler.vectorize_string(songs_joined, char2idx)

    seq_length = 100
    x_batch, y_batch = NeuralHandler.get_batch(vectorized_songs, seq_length=seq_length, batch_size=32)

    model = RnnLSTMModel(output_size=len(vocab), hidden_size=256, rnn_units=1024, input_size=seq_length)
    model.eval()

    pred, _ = model(torch.from_numpy(x_batch).long())

    # Convert logits to probabilities
    probabilities = torch.softmax(pred[-1], dim=-1)

    sampled_indices = torch.multinomial(probabilities, 1, replacement=True)
    sampled_indices = sampled_indices.squeeze().detach().numpy()

    input_str = repr("".join(idx2char[x_batch[-1]]))
    predictions = repr("".join(idx2char[sampled_indices]))

    assert len(input_str) == len(predictions)
