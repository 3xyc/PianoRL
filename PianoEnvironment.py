import random

import fluidsynth
import librosa
import numpy
import numpy as np
import gymnasium as gym
import pretty_midi
import pyaudio
from fluidsynth import Synth
from matplotlib import pyplot as plt
import pretty_midi as pm
from midi_util import synthesize_fluidsynth
from stable_baselines3.common.env_checker import check_env


class MLPPianoEnv(gym.Env):
    def __init__(self, audio_file, sound_file, sample_rate=44100, hop_length=512, number_of_keys: int = 10,
                 max_velocity: int = 120):
        super().__init__()

        self.sample_rate = sample_rate
        self.hop_length = hop_length

        self.target_states = {}

        self.sound_file = sound_file
        self.fl_synth = fluidsynth.Synth(samplerate=self.sample_rate)
        # Load in the soundfont
        sfid = self.fl_synth.sfload(sound_file)
        self.fl_synth.program_select(0, sfid, 0, 0)

        # self._target_state, sr = librosa.load(audio_file, sr=self.sample_rate)

        self.target_note = pretty_midi.Note(60, 60, 0, 1)
        self._target_state = synthesize_fluidsynth(self.fl_synth, [self.target_note], sample_rate=self.sample_rate)

        self.number_of_keys = number_of_keys
        self.max_velocity = max_velocity
        self._agent_state = np.zeros(self._target_state.shape)

        # Define the action space as changes in notes, where each value represents a change in the note
        self.action_space = gym.spaces.MultiDiscrete([number_of_keys, 2])

        # Define observation_space
        self.observation_space = gym.spaces.Box(low=-1., high=1.,
                                                shape=(len(self._agent_state) + len(self._target_state),),
                                                dtype=np.float64)

        mid = pm.PrettyMIDI()
        self.agent_mid = mid
        agent_instrument = pm.Instrument(0, False, name='Agent Piano')
        self.agent_instrument = agent_instrument
        self.agent_mid.instruments.append(agent_instrument)

    def _get_obs(self):
        obs = np.hstack((self._agent_state, self._target_state))
        return obs

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_state - self._target_state, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset agent state to zeros, target state might be a nother song ? but loading takes long..
        self._agent_state = np.zeros(self._target_state.shape, dtype=np.float64)
        pitch = self.np_random.integers(0, self.number_of_keys)

        self.target_note = pretty_midi.Note(60, pitch, 0, 1)
        print("pitch: ", self.target_note)
        if self.target_note.pitch not in self.target_states.keys():
            self._target_state = synthesize_fluidsynth(self.fl_synth, [self.target_note],
                                                       sample_rate=self.sample_rate)
            self.target_states[self.target_note.pitch] = self._target_state
        else:
            self._target_state = self.target_states[self.target_note.pitch]

        # Return observation and info
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        old_distance = self._get_info()["distance"]
        # Apply the action (which represents changes in MIDI notes) to the current agent state

        midi_note = action[0]
        velocity = 60
        start = 0
        end = 1

        note = pretty_midi.Note(velocity, midi_note, start, end)
        print("note:", note)
        wav = synthesize_fluidsynth(self.fl_synth, [note], sample_rate=self.sample_rate)
        self._agent_state[:len(wav):] = wav[::]

        new_distance = self._get_info()["distance"]
        # Check if the agent has matched the target state
        terminated = (new_distance == 0) or self.target_note.pitch == note.pitch
        if terminated:
            print("terminated, distance= ", new_distance, note, self.target_note)
            # librosa.display.waveshow(wav, sr=self.sample_rate)
            # plt.show()
            reward = 1
        else:
            reward = -1

        # Truncated condition: not used for now
        truncated = False

        # Get the updated observation and info
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, bool(terminated), truncated, info


# Register the environment with Gym
gym.register(id="gymnasium_env/MLPPianoEnv-v0",
             entry_point=MLPPianoEnv)


class CNNPianoEnv(gym.Env):
    def __init__(self, audio_file, sound_file, sample_rate=44100, hop_length=512, number_of_keys: int = 10,
                 max_velocity: int = 120):
        super().__init__()

        self.sample_rate = sample_rate
        self.hop_length = hop_length

        self.target_states = {}

        self.sound_file = sound_file
        self.fl_synth = fluidsynth.Synth(samplerate=self.sample_rate)
        # Load in the soundfont
        sfid = self.fl_synth.sfload(sound_file)
        self.fl_synth.program_select(0, sfid, 0, 0)

        # self._target_state, sr = librosa.load(audio_file, sr=self.sample_rate)

        self.target_note = pretty_midi.Note(60, 60, 0, 1)
        wav = synthesize_fluidsynth(self.fl_synth, [self.target_note], sample_rate=self.sample_rate)
        stft = librosa.stft(wav, hop_length=self.hop_length)
        self._target_state = np.abs(stft)
        print("stft:", stft.shape, stft)


        self.number_of_keys = number_of_keys
        self.max_velocity = max_velocity
        self._agent_state = np.zeros(self._target_state.shape)

        # Define the action space as changes in notes, where each value represents a change in the note
        self.action_space = gym.spaces.MultiDiscrete([number_of_keys, 2])

        # Define observation_space
        self.observation_space = gym.spaces.Box(low=0., high=200.,
                                                shape=(2, self._target_state.shape[0],self._target_state.shape[1]),
                                                dtype=np.float64)

        mid = pm.PrettyMIDI()
        self.agent_mid = mid
        agent_instrument = pm.Instrument(0, False, name='Agent Piano')
        self.agent_instrument = agent_instrument
        self.agent_mid.instruments.append(agent_instrument)

    def _get_obs(self):
        obs = np.stack((self._agent_state, self._target_state), axis=0)
        print(obs.shape)
        return obs

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_state - self._target_state, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset agent state to zeros, target state might be a nother song ? but loading takes long..
        self._agent_state = np.zeros(self._target_state.shape, dtype=np.float64)
        pitch = self.np_random.integers(0, self.number_of_keys)

        self.target_note = pretty_midi.Note(60, pitch, 0, 1)
        print("pitch: ", self.target_note)
        if self.target_note.pitch not in self.target_states.keys():
            wav = synthesize_fluidsynth(self.fl_synth, [self.target_note],
                                                       sample_rate=self.sample_rate)
            stft = librosa.stft(wav, hop_length=self.hop_length)
            self._target_state = np.abs(stft)
            self.target_states[self.target_note.pitch] = self._target_state
        else:
            self._target_state = self.target_states[self.target_note.pitch]

        # Return observation and info
        observation = self._get_obs()
        info = self._get_info()
        return observation, info


    def step(self, action):
        old_distance = self._get_info()["distance"]
        # Apply the action (which represents changes in MIDI notes) to the current agent state

        midi_note = action[0]
        velocity = 60
        start = 0
        end = 1

        note = pretty_midi.Note(velocity, midi_note, start, end)
        print("note:", note)
        wav = synthesize_fluidsynth(self.fl_synth, [note], sample_rate=self.sample_rate)
        stft = librosa.stft(wav, hop_length=self.hop_length)
        self._agent_state[:len(wav):] = np.abs(stft[::])

        new_distance = self._get_info()["distance"]
        # Check if the agent has matched the target state
        terminated = (new_distance == 0) or self.target_note.pitch == note.pitch
        if terminated:
            print("terminated, distance= ", new_distance, note, self.target_note)
            # librosa.display.waveshow(wav, sr=self.sample_rate)
            # plt.show()
            reward = 1
        else:
            reward = -1

        # Truncated condition: not used for now
        truncated = False

        # Get the updated observation and info
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, bool(terminated), truncated, info




gym.register(id="gymnasium_env/CNNPianoEnv-v0",
                 entry_point=CNNPianoEnv)

if __name__ == "__main__":
    env = gym.make("gymnasium_env/MLPPianoEnv-v0", audio_file='resources/wav/c_eb_c_eb_c_eb_c.wav',
                   sound_file='resources/soundfiles/[GD] Clean Grand Mistral.sf2', sample_rate=44100, number_of_keys=10)
    check_env(env)
    print(env)

    env = gym.make("gymnasium_env/CNNPianoEnv-v0", audio_file='resources/wav/c_eb_c_eb_c_eb_c.wav',
                   sound_file='resources/soundfiles/[GD] Clean Grand Mistral.sf2', sample_rate=44100, number_of_keys=10)
    check_env(env)
    print(env)
