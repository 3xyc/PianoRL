import numpy as np
import gymnasium as gym

class PianoEnv(gym.Env):
    def __init__(self, size: int = 24):
        super().__init__()
        self.size = size
        self._agent_state = np.zeros(size)
        self._target_state = np.zeros(size)

        # define action space as the 88 Keys of the Piano
        self.action_space = gym.spaces.Discrete(size)

        # define Observation space,
        # for now as notes played by the piano, and the notes seen from the original piece
        self.observation_space = gym.spaces.Dict({
            "agent": gym.spaces.Box(low=0, high=1, shape=(size,), dtype=np.float32),
            "target": gym.spaces.Box(low=0, high=1, shape=(size,), dtype=np.float32),
        })


    def _get_obs(self):
        return {"agent": self._agent_state, "target": self._target_state}

    def _get_info(self):
        # manhatten distanz der Vektoren, representiert distanz / Ähnlichkeit beider lösungen
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self):#
        self._agent_state = np.zeros(self.size)
        self._target_state = np.zeros(self.size)
        # Reset the environment state
        return self._get_obs(), self._get_info() # Initial observation, info

    def step(self, action):
        # set note defined by action to 1
        self._agent_state[action] = 1

        # Completed only if the agent plays the same notes as the target
        # TODO adjust to some form of similarity calculation between both
        terminated = np.array(self._agent_state, self._target_state) == 0
        reward = 1 if terminated else 0


        # TODO understand what truncate does
        truncated = False

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

gym.register(id="gymnasium_env/PianoEnv-v0",
                 entry_point=PianoEnv)