import math
import random
from typing import Optional, Union, Tuple

import gym
import numpy as np
import torch
from gym import spaces
from torch import nn

from music_dataset import MusicDataset


def custom_loss(inputs, targets):
    """
    Calculates the loss of the current generated midi sample. Both are tensors of dtype long
    :param inputs: The current generated midi sample.
    :param targets: The target midi sample.
    :return: The loss of the current generated midi sample.
    """
    # get the mean of the sum of the square of the difference between the two tensors
    loss = torch.sum(torch.pow(inputs - targets, 2)) / inputs.shape[0]
    return loss


def get_tensor_of_sample(token_dim: int, tokens: dict):
    """
    :param token_dim: dimension of token
    :param tokens: The tokens of the midi sample.
    :param tokens:
    :return: The tensor of the midi sample.
    """
    flat_list = []
    for i in range(token_dim):
        if isinstance(tokens[i], int):
            flat_list.append(tokens[i])
        else:
            for j in range(len(tokens[i])):
                if isinstance(tokens[i][j], int):
                    flat_list.append(tokens[i][j])
    return torch.tensor(flat_list)


def get_repetition_penalty(token, window):
    """
    Count the number of repetitions of a token in a window and return the penalty.
    The higher the penalty the more likely the token is to be repeated.
    :param token: The token to calculate the repetition penalty for.
    :param window: The window to calculate the repetition penalty for.
    :return: The repetition penalty for the given token.
    """
    repetition = len(window) - window.count(token)
    penalty = math.pow(repetition, 2) / len(window)
    return penalty


class MusicEnvironment(gym.Env):
    """
    Music Environment for a dataset of MuMIDI tokenized MIDI files.
    A time step is a list of tokens where:
            (list index: token type)
            0: Pitch / Position / Bar / Program / (Chord)
            (1: Velocity)
            (2: Duration)
            1 or 3: Current Bar embedding
            2 or 4: Current Position embedding
            (-1: Tempo)
    """

    def __init__(self, dataset: MusicDataset, *, seq_length, seed: Optional[int] = None, return_info: bool = False,
                 options: Optional[dict] = None):
        self.frequency_penalty = 0.2
        self.presence_penalty = 0.2
        self.repetition_penalty = 1.2
        self.seq_length = seq_length
        self.token_dim = 5
        self.observation_space = spaces.Box(low=0, high=dataset.vocab_size, shape=(seq_length, self.token_dim),
                                            dtype=np.int32)
        self.action_space = spaces.Discrete(dataset.vocab_size * self.token_dim)
        self.return_info = return_info
        self.dataset = dataset
        self.options = options
        self.initial_seed = seed
        self.midi_sample = {i: [] for i in range(self.token_dim)}
        self.done = False
        self.is_playing = False
        self.is_learning = True
        self.target_midi_sample = {i: [] for i in range(self.token_dim)}
        self.current_template = None
        self.get_target_sample(dataset, seq_length)
        self.reset()

    def get_target_sample(self, dataset, seq_length):
        target_sample = random.choice(dataset)
        self.current_template = target_sample[1]
        sample = target_sample[0]
        for s in range(seq_length):
            if len(sample) > s:
                for i in range(self.token_dim):
                    if len(sample[s]) > i:
                        self.target_midi_sample[i].append(sample[s][i])

    def close(self):
        pass

    def seed(self, seed: Optional[int] = None):
        np.random.seed(seed)

    def step(self, action):
        """
        :param action: The next mumidi token to add to the generated midi sample.
        :return: (observation, reward, done, info)
        """
        # check modulo of action to find out which dimension in the midi sample to update
        self.add_action_to_sample(action)
        self.done = False
        if len(self.midi_sample) == self.seq_length:
            self.done = True
            self.update_sample()
        info = {}
        reward = 0
        if self.is_learning:
            reward = self.get_reward()
            if self.return_info:
                info = {'reward': reward}
        return self.observation(self.midi_sample), reward, self.done, info

    def add_action_to_sample(self, action):
        for i in range(self.token_dim):
            if action % self.token_dim == i:
                self.midi_sample[i].append(action // self.token_dim)

    def update_sample(self, reset=False):
        if reset:
            self.midi_sample = {i: [] for i in range(self.token_dim)}
        elif len(self.midi_sample[0][1:]) > self.seq_length:
            self.midi_sample = {i: self.midi_sample[i][1:] for i in range(self.token_dim)}
        if reset or len(self.midi_sample[0][1:]) < self.seq_length:
            self.get_target_sample(self.dataset, self.seq_length)
        else:
            self.target_midi_sample = {i: self.target_midi_sample[i][1:] for i in range(self.token_dim)}

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None):
        """
        :param seed: The seed for the random number generator.
        :param return_info: Whether to return the observation and info.
        :param options: The options for the environment.
        :return: The observation of the current generated midi sample.
        """
        self.target_midi_sample = {i: [] for i in range(self.token_dim)}
        self.midi_sample = {i: [] for i in range(self.token_dim)}
        self.seed(seed)
        self.return_info = return_info
        self.options = options
        self.update_sample(reset=True)
        return self.observation(self.midi_sample)

    def render(self, mode="human"):
        """
        Play the current generated midi sample if the episode is done.
        :param mode: The mode of rendering.
        """
        if self.done and not self.is_playing:
            print("Playing generated midi sample of template: {}".format(self.current_template))
            self.dataset.play_midi(self.midi_sample)
            self.is_playing = True

    def observation(self, midi_sample, as_tensor=False):
        """
        :param as_tensor: Whether to return the observation as a tensor.
        :param midi_sample: The current generated midi sample.
        :return: The observation of the current generated midi sample.
        """
        arr = np.resize(get_tensor_of_sample(self.token_dim, midi_sample).numpy(), (self.seq_length, self.token_dim))
        return torch.tensor(arr) if as_tensor else arr.tolist()

    def get_reward(self):
        """
        The reward is the loss between the generated midi sample and the target midi sample.
        Since the reward needs to get higher the closer the generated midi sample is to the target midi sample,
        the reward is negative.
        Additionally the reward gets multiplied by the following factors:
        -- repetition_penalty (the reward gets smaller the more the same notes are repeated way too often)
        -- frequency_penalty (the more often the same pattern verbatim is found the smaller the reward)
        -- presence_penalty (the reward gets higher the more notes which are less frequent are present)
        :return: The reward of the current generated midi sample.
        """
        base_reward = -self.get_loss()
        repetition_penalty = self.repetition_penalty * self.get_repetition_penalty(self.midi_sample)
        frequency_penalty = self.frequency_penalty * self.get_frequency_penalty(self.midi_sample)
        presence_penalty = self.presence_penalty * self.get_presence_penalty(self.midi_sample)
        penalized_reward = base_reward + (-base_reward * repetition_penalty * frequency_penalty * presence_penalty)
        return penalized_reward / 2.0

    def write_midi_sample(self, filename):
        """
        Writes the current generated midi sample to a midi file.
        :param filename: The name of the midi file.
        """
        token_dict = self.midi_sample
        midi_sample = [[] for s in range(self.seq_length)]
        for s in range(self.seq_length):
            for i in range(self.token_dim):
                if len(token_dict[i]) > s:
                    midi_sample[s].append(token_dict[i][s])
        midi_file = self.dataset.decode_tokens(midi_sample)
        midi_file.dump(filename)

    def get_repetition_penalty(self, midi_sample, repetition_window=8):
        """
        the more the same notes are repeated way too often the smaller the value this function returns
        :param repetition_window: The window size for the repetition penalty.
        :param midi_sample: The current generated midi sample.
        :return: The repetition penalty of the current generated midi sample.
        """
        repetition_penalty = 0
        for i in range(self.token_dim):
            for j in range(self.seq_length):
                if j + repetition_window < self.seq_length:
                    # make sure no index out of range error occurs
                    if len(midi_sample) > j and len(midi_sample[j]) > i:
                        token = midi_sample[j][i]
                        if len(midi_sample) > j + repetition_window:
                            window = midi_sample[j + repetition_window]
                            repetition_penalty += self.repetition_penalty * (1 - get_repetition_penalty(token, window))
        return 1 / (repetition_penalty + 1)

    def get_frequency_penalty(self, midi_sample, pattern_size=4):
        """
        the more often the same pattern verbatim is found the smaller the value this function returns
        to implement this we need to count the number of times the same pattern is found in the generated midi sample.
        The difference to the repetition penalty is that the frequency penalty is only calculated for the same sequence
        of tokens, not for the whole generated midi sample.
        :param midi_sample: The current generated midi sample.
        :param pattern_size: The size of the pattern to check for.
        :return: The frequency penalty of the current generated midi sample.
        """
        frequency_penalty = 0
        for i in range(self.token_dim):
            for j in range(self.seq_length - pattern_size + 1):
                try:
                    pattern = [midi_sample[i][j + k] for k in range(pattern_size)]
                    frequency_penalty += self.get_frequency_count(pattern)
                except IndexError:
                    pass
                except KeyError:
                    pass
        return 1 / (frequency_penalty + 1)

    def get_presence_penalty(self, midi_sample, rarity=0.5):
        """
        The value of this function gets higher the more of less frequent notes are present.
        To calculate this we need to count the number of notes which are present in the generated midi sample.
        Then we need to divide the number of notes by the total number of notes.
        :param rarity: the factor that filters out non-rare notes.
        :param midi_sample: The current generated midi sample.
        :return: The presence penalty of the current generated midi sample.
        """
        note_count = 0
        for i in range(self.token_dim):
            for j in range(self.seq_length):
                if len(midi_sample) > j and len(midi_sample[j]) > i:
                    note_count += 1
        return 1 - (note_count / (self.token_dim * self.seq_length)) * rarity

    def get_loss(self):
        """
        Calculates the loss of the current generated midi sample.
        :return: The loss of the current generated midi sample.
        """
        inputs = self.observation(self.midi_sample, as_tensor=True)
        targets = self.observation(self.target_midi_sample, as_tensor=True)
        assert inputs.shape == targets.shape
        return custom_loss(inputs, targets).item()

    def get_frequency_count(self, pattern):
        """
        Counts the number of times the given pattern is found in the generated midi sample.
        :param pattern: The pattern to check for.
        :return: The number of times the given pattern is found in the generated midi sample.
        """
        frequency_count = 0
        for i in range(self.seq_length - len(pattern) + 1):
            for token_list in list(self.midi_sample.values()):
                if token_list[i:i + len(pattern)] == pattern:
                    frequency_count += 1
        return frequency_count
