import struct

import librosa
import musicpy.musicpy
import numpy as np
import gymnasium as gym
import sf2_loader as sf
from matplotlib import pyplot as plt
from musicpy.daw import daw
from musicpy.structures import chord, note


class PianoEnv(gym.Env):
    def __init__(self, audio_file, sound_file, sample_rate=44100, number_of_keys: int = 1, max_velocity: int = 120):
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
        self.wav_data, _ = librosa.load(audio_file, sr=self.sample_rate)
        self.cqt_data = librosa.cqt(self.wav_data, sr=self.sample_rate)
        self.agent_daw = daw(1, name='piano song')
        print(self.wav_data.shape)
        print(self.cqt_data.shape)


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

    def to_wav(self, piece):
        wav = self.agent_daw.export(piece, mode='wav', action='get', show_msg=True)
        raw_data = wav.raw_data
        # Convert the raw data to integers (16-bit PCM)
        # 'h' format is used for signed 16-bit (2 bytes)
        num_samples = len(raw_data) // 2  # Each sample is 2 bytes (16-bit PCM)
        audio_samples = struct.unpack('<' + 'h' * num_samples, raw_data)  # Little-endian format

        # Convert to numpy array for easier manipulation
        audio_samples_np = np.array(audio_samples, dtype=np.float32)
        return audio_samples_np
    def to_midi_stream(self, piece):
        midi_stream = musicpy.musicpy.write(piece,
                                            bpm=120,
                                            channel=0,
                                            start_time=None,
                                            save_as_file=False)
        print(midi_stream.getvalue())
        piece_from_midi_stream = musicpy.musicpy.read(midi_stream, is_file=True)
        return midi_stream

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

    new_song = daw(1, name='piano song')
    new_song.load(0, r'resources/soundfiles/[GD] Clean Grand Mistral.sf2')
    # export piece object to wav
    wav = new_song.export(note('C', 5), mode='wav', action='get', show_msg=True)
    # export piece object to midi stream
    midi_stream = musicpy.musicpy.write(note('C', 5),
          bpm=120,
          channel=0,
          start_time=None,
          save_as_file=False)
    print(midi_stream.getvalue())

    midi_object = musicpy.musicpy.read(midi_stream, is_file=True)
    print(midi_object)
    print(wav)
    print(wav[0:1000])
    raw_data = wav.raw_data
    # Convert the raw data to integers (16-bit PCM)
    # 'h' format is used for signed 16-bit (2 bytes)
    num_samples = len(raw_data) // 2  # Each sample is 2 bytes (16-bit PCM)
    audio_samples = struct.unpack('<' + 'h' * num_samples, raw_data)  # Little-endian format

    # Convert to numpy array for easier manipulation
    audio_samples_np = np.array(audio_samples, dtype=np.float32)

    # Apply librosa's CQT or other processing directly (assuming 44100Hz sample rate)
    sr = 44100  # Sample rate (you should adjust this based on your audio file)
    cqt = librosa.cqt(audio_samples_np, sr=sr)
    print(cqt)
    # Display the CQT spectrogram
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(cqt), ref=np.max),
                             sr=sr, x_axis='time', y_axis='cqt_note')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Constant-Q power spectrum')
    plt.show()
    print(audio_samples_np)
    print(wav)