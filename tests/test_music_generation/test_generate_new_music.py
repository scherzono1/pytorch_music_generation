import numpy as np
import torch

from music.music_handler import MusicHandler
from neural_handler.rnn_lstm_model import RnnLSTMModel

num_training_iterations = 2  # Increase this to train longer
batch_size = 64  # Experiment between 1 and 64
seq_length = 100  # Experiment between 50 and 500
learning_rate = 5e-3  # Experiment between 1e-5 and 1e-1

# Model parameters:
embedding_dim = 256
rnn_units = 1024  # Experiment between 1 and 2048

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_generate_new_music():

    # Download the dataset
    songs = MusicHandler.load_songs()

    # Join our list of song strings into a single string containing all songs
    songs_joined = MusicHandler.join_songs(songs)

    # Find all unique characters in the joined string
    vocab = MusicHandler.get_vocab(songs_joined)
    vocab_size = len(vocab)

    # Map every unique character to a unique integer
    char2idx = MusicHandler.get_char2idx(vocab)
    idx2char = MusicHandler.get_idx2char(vocab)

    # load model
    model = RnnLSTMModel(output_size=vocab_size, hidden_size=embedding_dim, rnn_units=rnn_units, input_size=seq_length)
    model.load_state_dict(torch.load('model_1.pt'))
    model.eval()
    model.to(device)

    generation_length = 10000
    start_string = 'X'

    input_eval = [char2idx[s] for s in start_string]*seq_length
    input_eval = torch.tensor(input_eval, dtype=torch.long).unsqueeze(0)

    # Empty string to store our results
    text_generated = []

    hidden = None
    cell = None

    for i in range(generation_length):

        input_eval = input_eval.to(device)

        with torch.no_grad():

            predictions, (hidden, cell) = model(input_eval, hidden, cell)

        last_char_logits = predictions[0, -1, :]
        p = torch.nn.functional.softmax(last_char_logits, dim=0).detach().cpu().numpy()
        predicted_char_index = np.random.choice(len(p), p=p)
        predicted_char = idx2char[predicted_char_index]

        text_generated.append(predicted_char)

        # Update the current sequence for the next iteration
        input_eval = input_eval.cpu().numpy().squeeze(0)
        input_eval = np.roll(input_eval, -1)
        input_eval[-1] = predicted_char_index
        input_eval = torch.tensor(input_eval).unsqueeze(0)

    generated_text = 'X' + ''.join(text_generated)

    # save as file
    with open('generated_text.txt', 'w') as f:
        f.write(generated_text)
