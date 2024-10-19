import numpy as np
import gymnasium as gym


class PianoEnv(gym.Env):
    def __init__(self, number_of_keys: int = 1, max_velocity: int = 120):
        super().__init__()
        self.number_of_keys = number_of_keys
        self.max_velocity = max_velocity
        self._agent_state = np.zeros(number_of_keys, dtype=np.int64)
        self._target_state = np.zeros(number_of_keys, dtype=np.int64)

        # Define the action space as changes in notes, where each value represents a change in the note
        self.action_space = gym.spaces.MultiDiscrete([number_of_keys, 2])

        # Define observation space: agent's current state (MIDI notes between 0 and 127)
        self.observation_space = gym.spaces.Box(low=0, high=max_velocity, shape=(2, number_of_keys), dtype=np.int64)

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
