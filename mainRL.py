import gymnasium as gym
from PianoEnvironment import PianoEnv

if __name__ == '__main__':


    env = gym.make("gymnasium_env/PianoEnv-v0")
    print(env)