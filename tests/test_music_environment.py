import os
import random
import unittest

from miditoolkit import MidiFile
from tqdm import tqdm

from music_dataset import MusicDataset
from music_environment import MusicEnvironment

"""
Tests for the openai gym music environment for the music generation project using symbolic music data tokenized
from midi files with MuMIDI tokenizer for multitrack music generation.
"""


class TestMusicEnvironment(unittest.TestCase):
    dataset = MusicDataset('../musescore')

    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset.load_data()

    def test_music_environment_initialization(self):
        """
        Test the initialization of the music environment.
        """
        seq_length = 32
        env = MusicEnvironment(self.dataset, seq_length=seq_length)
        self.assertEqual(env.action_space.n, self.dataset.vocab_size * env.token_dim)
        self.assertEqual(env.observation_space.shape, (seq_length, env.token_dim))
        self.assertEqual(env.dataset.vocab_size, self.dataset.vocab_size)
        # Test the midi sample to be equal to a dict with the expected keys
        self.assertEqual(env.midi_sample.keys(), {i for i in range(env.token_dim)})
        # Test the midi sample to be equal to a dict with the expected values
        self.assertEqual(list(env.midi_sample.values()), [[] for _ in range(env.token_dim)])
        # Check that done is False
        self.assertFalse(env.done)
        # Check that is_playing is False
        self.assertFalse(env.is_playing)
        # Test the target midi sample length to be equal to seq_length
        self.assertEqual(len(env.target_midi_sample[0]), seq_length)
        # Test the target midi sample to be equal to a dict with the expected keys
        self.assertEqual(env.target_midi_sample.keys(), {i for i in range(env.token_dim)})

    def test_music_environment_reset(self):
        """
        Test the reset of the music environment.
        """
        seq_length = 32
        env = MusicEnvironment(self.dataset, seq_length=seq_length)
        env.reset()
        # Test the midi sample to be equal to a dict with the expected keys
        self.assertEqual(list(env.midi_sample.keys()), [i for i in range(env.token_dim)])
        # Test the midi sample to be equal to a dict with the expected values
        self.assertEqual(list(env.midi_sample.values()), [[] for _ in range(env.token_dim)])
        # Check that done is False
        self.assertFalse(env.done)
        # Check that is_playing is False
        self.assertFalse(env.is_playing)
        # Test the target midi sample length to be equal to seq_length
        self.assertEqual(len(env.target_midi_sample[0]), seq_length)
        # Test the target midi sample to be equal to a dict with the expected keys
        self.assertEqual(env.target_midi_sample.keys(), {i for i in range(env.token_dim)})

    def test_music_environment_step(self):
        """
        Test the step of the music environment.
        """
        seq_length = 32
        action = 0
        env = MusicEnvironment(self.dataset, seq_length=seq_length)
        env.reset()
        env.step(action)
        # Test the midi sample to be equal to a dict with the expected keys
        self.assertEqual(list(env.midi_sample.keys()), [i for i in range(env.token_dim)])
        # Test the midi sample to be equal to a dict with the expected values
        self.assertEqual([len(v) for v in list(env.midi_sample.values())],
                         [1 if action % env.token_dim == i else 0 for i in range(env.token_dim)])
        # Test the target midi sample length to be equal to seq_length
        self.assertEqual(len(env.target_midi_sample[0]), seq_length)
        # Test the target midi sample to be equal to a dict with the expected keys
        self.assertEqual(env.target_midi_sample.keys(), {i for i in range(env.token_dim)})

    def test_music_environment_render(self):
        """
        Test the render of the music environment.
        """
        seq_length = 32
        env = MusicEnvironment(self.dataset, seq_length=seq_length)
        env.reset()
        env.render()

    def test_music_environment_close(self):
        """
        Test the close of the music environment.
        """
        seq_length = 32
        env = MusicEnvironment(self.dataset, seq_length=seq_length)
        env.close()

    def test_music_environment_seed(self):
        """
        Test the seed of the music environment.
        """
        seq_length = 32
        env = MusicEnvironment(self.dataset, seq_length=seq_length)
        env.seed(0)

    def test_add_action_to_sample(self):
        """
        Test the add_action_to_sample method of the music environment.
        """
        seq_length = 32
        env = MusicEnvironment(self.dataset, seq_length=seq_length)
        env.reset()
        env.add_action_to_sample(0)
        # Test the midi sample to be equal to a dict with the expected keys
        self.assertEqual(list(env.midi_sample.keys()), [i for i in range(env.token_dim)])
        # Test the midi sample to be equal to a dict with the expected values
        self.assertEqual(sum([len(v) for v in list(env.midi_sample.values())]), 1)
        # Test the target midi sample length to be equal to seq_length
        self.assertEqual(len(env.target_midi_sample[0]), seq_length)
        # Test the target midi sample to be equal to a dict with the expected keys
        self.assertEqual(env.target_midi_sample.keys(), {i for i in range(env.token_dim)})

    def test_update_sample(self):
        """
        Test the update_sample method of the music environment.
        """
        seq_length = 32
        env = MusicEnvironment(self.dataset, seq_length=seq_length)
        env.reset()
        actions = []
        for _ in range(seq_length + 1):
            actions.append(random.randint(0, self.dataset.vocab_size - 1))
            env.step(actions[-1])
        if env.done:
            # Test the midi sample to be equal to a dict with the expected keys
            self.assertEqual(env.midi_sample.keys(), {i for i in range(env.token_dim)})
            # Test the midi sample to be equal to a dict with the expected values
            self.assertEqual([len(v) for v in list(env.midi_sample.values())],
                             [env.seq_length for _ in range(env.token_dim)])
            # Test the target midi sample length to be equal to seq_length
            self.assertEqual(len(env.target_midi_sample[0]), seq_length)
            # Test the target midi sample to be equal to a dict with the expected keys
            self.assertEqual(env.target_midi_sample.keys(), {i for i in range(env.token_dim)})
        else:
            # Test the midi sample to be equal to a dict with the expected keys
            self.assertEqual(env.midi_sample.keys(), {i for i in range(env.token_dim)})
            # Test the midi sample to be equal to a dict with the expected values
            self.assertEqual(sum([len(v) for v in list(env.midi_sample.values())]), env.seq_length + 1)
            # Test the target midi sample length to be equal to seq_length
            self.assertEqual(len(env.target_midi_sample[0]), seq_length)
            # Test the target midi sample to be equal to a dict with the expected keys
            self.assertEqual(env.target_midi_sample.keys(), {i for i in range(env.token_dim)})

    def test_get_reward(self):
        """
        Test the get_reward method of the music environment.
        """
        seq_length = 32
        env = MusicEnvironment(self.dataset, seq_length=seq_length)
        env.reset()
        actions = []
        for _ in range(seq_length + 1):
            actions.append(random.randint(0, self.dataset.vocab_size - 1))
            env.step(actions[-1])
        # Test the reward to be smaller than 0
        reward = env.get_reward()
        print(reward)
        self.assertLess(reward, 0)

    def test_penalties(self):
        """
        Test the presence penalty, frequency_penalty and repetition_penalty methods of the music environment.
        """
        seq_length = 32
        env = MusicEnvironment(self.dataset, seq_length=seq_length)
        env.reset()
        actions = []
        for _ in range(seq_length + 1):
            actions.append(random.randint(0, self.dataset.vocab_size - 1))
            env.step(actions[-1])
        reward = env.get_reward()
        presence_penalty = env.get_presence_penalty(env.midi_sample)
        frequency_penalty = env.get_frequency_penalty(env.midi_sample)
        repetition_penalty = env.get_repetition_penalty(env.midi_sample)
        # print all
        print(reward, presence_penalty, frequency_penalty, repetition_penalty)
        # test reward to be smaller than the penalties
        self.assertLess(reward, presence_penalty + frequency_penalty + repetition_penalty)
        # test all penalties to be different from the reward
        self.assertNotEqual(presence_penalty, reward)
        self.assertNotEqual(frequency_penalty, reward)
        self.assertNotEqual(repetition_penalty, reward)
        # test all penalties to be different from each other
        self.assertNotEqual(presence_penalty, frequency_penalty)
        self.assertNotEqual(presence_penalty, repetition_penalty)
        self.assertNotEqual(frequency_penalty, repetition_penalty)

    def test_write_midi_sample(self):
        """
        Test the write_midi_sample method of the music environment.
        """
        # choose a large sequence length to check if the midi file will contain music one can play
        seq_length = 1024
        env = MusicEnvironment(self.dataset, seq_length=seq_length)
        env.reset()
        env.is_learning = False
        actions = []
        for _ in tqdm(range(seq_length + 1)):
            actions.append(random.randint(0, self.dataset.vocab_size - 1))
            env.step(actions[-1])
        env.write_midi_sample('test_write_midi_sample.mid')
        # Test that the midi file exists
        self.assertTrue(os.path.isfile('test_write_midi_sample.mid'))
        # Test that the midi file is not empty and larger than 40 bytes
        self.assertGreater(os.path.getsize('test_write_midi_sample.mid'), 40)
        midi_file = MidiFile('test_write_midi_sample.mid')
        # Test that the midi file is a valid midi file
        self.assertGreater(midi_file.ticks_per_beat, 0)
        # Test that the midi file contains the expected number of notes
        self.assertGreater(sum([len(track.notes) for track in midi_file.instruments]), 0)
        # Test that the midi file contains at least 1 track
        self.assertTrue(len(midi_file.instruments) > 0)
        # Test the max tick of the midi file is equal or larger than the 0
        self.assertGreaterEqual(midi_file.max_tick, 0)
        # Remove the midi file
        os.remove('test_write_midi_sample.mid')


if __name__ == '__main__':
    unittest.main()
