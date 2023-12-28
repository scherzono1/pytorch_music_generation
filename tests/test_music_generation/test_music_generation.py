from music.music_handler import MusicHandler


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
