"""
We use the MuMIDI tokenizer from miditok to tokenize the midi files with the MusicDataset class.
"""

import os
import sys
import glob
from typing import List, Dict

import numpy as np
import torch
import torch.utils.data as data
from miditok import MuMIDI
from miditoolkit import MidiFile

tokenizer = MuMIDI()


def tokenize_midi(midi_file) -> List[List[int]]:
    """
    Tokenize the midi file.
    :param midi_file: The midi file to be tokenized.
    :return: The tokenized midi file.
    """
    global tokenizer
    try:
        midi_file = MidiFile(midi_file)
        tokens = tokenizer.midi_to_tokens(midi_file)
    except Exception as e:
        print(e)
        return None
    return tokens


def decode_tokens(tokens):
    """
    Decode the token.
    :param token: The token to be decoded.
    :return: The decoded token.
    """
    global tokenizer
    return tokenizer.tokens_to_midi(tokens)


class MusicDataset(data.Dataset):
    """
    This class is used to load the data from the midi files.
    """

    def __init__(self, root, transform=None, target_transform=None):
        """
        Initialize the dataset.
        :param root: The root directory of the dataset.
        :param transform: The transform to be applied to the data.
        :param target_transform: The transform to be applied to the target.
        """
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.data: List[List[List[int]]] = []
        self.targets = []
        self.tokenized_midi = None
        self.length = 0
        self.skipped = 0
        self.vocab : Dict[str, List[int]] = tokenizer.vocab
        self.vocab_size = len(self.vocab)

    def load_data(self):
        """
        Load the data from the midi files.
        """
        self.length = 0
        self.skipped = 0
        for file in glob.glob(os.path.join(self.root, '*.mid')):
            tokens = tokenize_midi(file)
            if tokens is not None:
                self.length += 1
                self.data.append(tokens)
            else:
                self.skipped += 1
            self.targets.append(os.path.basename(file).split('.')[0])

    def __getitem__(self, index):
        """
        Get the data and target at the given index.
        :param index: The index of the data and target.
        :return: The data and target at the given index.
        """
        if self.transform is not None:
            data = self.transform(self.data[index])
        else:
            data = self.data[index]
        target = self.targets[index]
        if self.target_transform is not None:
            target = self.target_transform(target)
        return data, target

    def __len__(self):
        """
        Get the length of the dataset.
        :return: The length of the dataset.
        """
        return self.length

    def tokenize_midi(self, midi_file):
        """
        Tokenize the midi file.
        :param midi_file: The midi file to be tokenized.
        :return: The tokenized midi file.
        """
        self.tokenized_midi = tokenize_midi(midi_file)

    def decode_tokens(self, tokens):
        """
        Decode the token.
        :param token: The token to be decoded.
        :return: The decoded token.
        """
        return decode_tokens(tokens)

    def play_midi(self, tokens):
        """
        Play the tokenized midi file.
        :param tokens: The tokenized midi file.
        """
        # delete previous temporary midi file if exists
        if os.path.exists('temp.mid'):
            os.remove('temp.mid')
        midi = decode_tokens(tokens)
        # Create temporary midi file
        midi.dump('temp.mid')
        # Play the midi file using a synth available in windows
        os.system('temp.mid')
