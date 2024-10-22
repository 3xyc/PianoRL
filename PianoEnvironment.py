import fluidsynth
import librosa
import numpy
import numpy as np
import gymnasium as gym
import pyaudio
from fluidsynth import Synth
from matplotlib import pyplot as plt
from mido import MidiFile, MidiTrack, Message


class PianoEnv(gym.Env):
    def __init__(self, audio_file, sound_file, sample_rate=44100, hop_length=512,  number_of_keys: int = 1, max_velocity: int = 120):
        super().__init__()
        self.number_of_keys = number_of_keys
        self.max_velocity = max_velocity
        self._agent_state = np.zeros(number_of_keys, dtype=np.int64)
        self._target_state = np.zeros(number_of_keys, dtype=np.int64)

        # Define the action space as changes in notes, where each value represents a change in the note
        self.action_space = gym.spaces.MultiDiscrete([number_of_keys, 2])

        # Define observation space: agent's current state (MIDI notes between 0 and 127)
        self.observation_space = gym.spaces.Box(low=0, high=max_velocity, shape=(2, number_of_keys), dtype=np.int64)

        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.wav_data, sr = librosa.load(audio_file, sr=self.sample_rate)
        self.cqt_data = librosa.cqt(self.wav_data, hop_length=hop_length, sr=self.sample_rate)

        print(self.wav_data.shape)
        print(self.cqt_data.shape)
        librosa.display.specshow(librosa.amplitude_to_db(np.abs(self.cqt_data), ref=np.max),
                                 sr=sr, x_axis='time', y_axis='cqt_note')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Constant-Q power spectrum')
        plt.show()

        mid = MidiFile()
        track = MidiTrack()
        mid.tracks.append(track)
        track.append(Message('note_on', note=60, velocity=64))
        track.append(Message('note_off', note=60, velocity=64, time=1000))

        mid.save('new_song.mid')

    def _get_obs(self):
        obs = np.vstack((self._agent_state, self._target_state))
        print(obs.shape)
        return obs

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_state - self._target_state, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Reset agent state to zeros, target state is set to random MIDI note values between 0 and 127
        self._agent_state = np.zeros(self.number_of_keys, dtype=np.int64)
        self._target_state = self.np_random.integers(0, self.max_velocity, size=self.number_of_keys, dtype=np.int64)
        print("agent_state", self._agent_state)
        print("target_state", self._target_state)

        # Return observation and info
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        old_distance = self._get_info()["distance"]
        # Apply the action (which represents changes in MIDI notes) to the current agent state
        step = 0
        if action[1] == 0 and self._agent_state[action[0]] < self.max_velocity:
            step = 1
        elif action[1] == 1 and self._agent_state[action[0]] > -self.max_velocity:
            step = -1
        print(action)
        print("performing, step:", self._agent_state[action[0]], step)
        self._agent_state[action[0]] += step

        new_distance = self._get_info()["distance"]
        # Check if the agent has matched the target state
        terminated = new_distance == 0
        print(new_distance)
        reward = 1 if new_distance == 0 else 0
        reward = -10 if step == 0 else old_distance - new_distance
        # reward = -self._get_info()["distance"]/self._get_info()["distance"]
        print("reward", reward)

        # Truncated condition: not used for now
        truncated = False

        # Get the updated observation and info
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info


# Register the environment with Gym
gym.register(id="gymnasium_env/PianoEnv-v0",
             entry_point=PianoEnv)


if __name__ == "__main__":
    env = gym.make("gymnasium_env/PianoEnv-v0", audio_file='resources/wav/c_eb_c_eb_c_eb_c.wav', sound_file='resources/soundfiles/[GD] Clean Grand Mistral.sf2', sample_rate=44100, number_of_keys=1)
    print(env)

    mid = MidiFile(r"new_song.mid")

