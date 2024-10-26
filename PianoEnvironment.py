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

class PianoEnv(gym.Env):
    def __init__(self, audio_file, sound_file, sample_rate=44100, hop_length=512, number_of_keys: int = 1,
                 max_velocity: int = 120):
        super().__init__()
        self.sound_file = sound_file
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self._target_state, sr = librosa.load(audio_file, sr=self.sample_rate)
        print(self._target_state.shape)

        self.number_of_keys = number_of_keys
        self.max_velocity = max_velocity
        self._agent_state = np.zeros(self._target_state.shape)

        # Define the action space as changes in notes, where each value represents a change in the note
        self.action_space = gym.spaces.MultiDiscrete([number_of_keys, 2])

        # Define observation_space
        print(self._target_state.max())
        self.observation_space = gym.spaces.Box(low=-1., high=1., shape=(2, self._target_state.shape[0]), dtype=np.float64)

        mid = pm.PrettyMIDI()
        self.agent_mid = mid
        agent_instrument = pm.Instrument(0, False, name='Agent Piano')
        self.agent_instrument = agent_instrument
        self.agent_mid.instruments.append(agent_instrument)

        self.fluidsynth = fluidsynth.Synth(samplerate=self.sample_rate)
        # Load in the soundfont
        sfid = self.fluidsynth.sfload(sound_file)
        self.fluidsynth.program_select(0, sfid, 0, 0)

        note = pretty_midi.Note(60, 62, 0, 4)
        agent_instrument.notes.append(note)
        #wav_data = synthesize_fluidsynth(self.fluidsynth, [note, pretty_midi.Note(60, 66, 4, 5)])
        wav_data = agent_instrument.fluidsynth(sample_rate, sound_file)


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

        # Reset agent state to zeros, target state might be a nother song ? but loading takes long..
        self._agent_state = np.zeros(self._target_state.shape, dtype=np.float64)
        #self._target_state = self.np_random.integers(0, self.max_velocity, size=self.number_of_keys, dtype=np.int64)
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
        terminated = (new_distance == 0)
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

        return observation, reward, bool(terminated), truncated, info


# Register the environment with Gym
gym.register(id="gymnasium_env/PianoEnv-v0",
             entry_point=PianoEnv)

if __name__ == "__main__":
    env = gym.make("gymnasium_env/PianoEnv-v0", audio_file='resources/wav/c_eb_c_eb_c_eb_c.wav',
                   sound_file='resources/soundfiles/[GD] Clean Grand Mistral.sf2', sample_rate=44100, number_of_keys=1)
    check_env(env)
    print(env)
