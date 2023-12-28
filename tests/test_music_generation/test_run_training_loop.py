import torch
from torch.utils.data import DataLoader

from music.music_handler import MusicHandler
from neural_handler.rnn_lstm_model import RnnLSTMModel
from neural_handler.rolling_dataset import RollingDataset

# Optimization parameters:
num_training_iterations = 2  # Increase this to train longer
batch_size = 64  # Experiment between 1 and 64
seq_length = 100  # Experiment between 50 and 500
learning_rate = 5e-3  # Experiment between 1e-5 and 1e-1

# Model parameters:
embedding_dim = 256
rnn_units = 1024  # Experiment between 1 and 2048

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_run_training_loop():

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

    # Vectorize the songs string
    vectorized_songs = MusicHandler.vectorize_string(songs_joined, char2idx)

    model = RnnLSTMModel(output_size=vocab_size, hidden_size=embedding_dim, rnn_units=rnn_units, input_size=seq_length)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = torch.nn.CrossEntropyLoss()

    # Load the data in batches
    train_loader = DataLoader(dataset=RollingDataset(vectorized_songs, seq_length), batch_size=batch_size, shuffle=True)

    for epoch in range(num_training_iterations):
        print(f'Epoch [{epoch + 1}/{num_training_iterations}]')
        for i, (features, labels) in enumerate(train_loader):

            # ensure the shapes are correct
            if features.shape[0] != batch_size or labels.shape[0] != batch_size:
                continue

            features = torch.tensor(features, dtype=torch.long)
            labels = torch.tensor(labels, dtype=torch.long)

            # Move data to the same device as the model
            features, labels = features.to(device), labels.to(device)

            # Forward pass
            outputs, _ = model(features)
            # outputs = torch.softmax(outputs, dim=-1)
            # sampled_indices = [torch.multinomial(output, 1, replacement=True) for output in outputs]
            # sampled_indices = torch.stack(sampled_indices).squeeze(-1)

            loss = loss_function(outputs.view(-1, 83), labels.view(-1))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'ITERATION [{i + 1}], Loss: {loss.item():.4f}')

        model.save_model(f"model_{epoch}.pt")
