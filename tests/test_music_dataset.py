import glob
import os
import unittest

from miditok import MuMIDI
from miditoolkit import MidiFile

from music_dataset import MusicDataset

"""
This file contains the tests for the music_dataset.py file.

The music dataset is a dataset of symbolic music notation that is created from midi files.
"""


class TestMusicDataset(unittest.TestCase):
    """
    This class contains the tests for the music_dataset.py file.
    """

    def test_mu_midi(self):
        """
        We use the MuMIDI tokenizer from miditok to tokenize the midi files.
        Here we test if the tokenizer works correctly.
        """

        # We create a MuMIDI object and a MusicDataset object.
        midi = MuMIDI()
        dataset = MusicDataset('../musescore')

        # We get the first midi file.
        midi_sample = "../musescore/_Dolphin_Shoals_MK8.mid"

        # We load the midi file.
        midi_file = MidiFile(midi_sample)
        tokens = midi.midi_to_tokens(midi_file)

        # We tokenize the midi file.
        dataset.tokenize_midi(midi_sample)

        # We check if the tokenizer works correctly.
        self.assertEqual(dataset.tokenized_midi, tokens)

    def test_load_data(self):
        """
        We test if the load_data function works correctly.
        """

        # We create a MusicDataset object.
        dataset = MusicDataset('../musescore')

        # We count how many midi files are in the dataset.
        files = glob.glob(os.path.join('../musescore', '*.mid'))

        # We load the data.
        dataset.load_data()

        # We check if the data is loaded correctly.
        self.assertEqual(dataset.length, len(files))

    def test_get_data(self):
        """
        We test if the get_data function works correctly.
        """

        # We create a MusicDataset object.
        dataset = MusicDataset('../musescore')

        # We load the data.
        dataset.load_data()

        # We get the data.
        data = dataset[0]

        # We check if the data is loaded correctly.
        self.assertEqual(len(data[0]), len(dataset.data[0]))


if __name__ == '__main__':
    unittest.main()
