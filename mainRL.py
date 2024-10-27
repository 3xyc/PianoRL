import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

import PianoEnvironment

def playout_policy(model, env):
    obs, info = env.reset()
    initial_state = obs
    done = False
    truncated = False
    i = 0
    while not (done or truncated):
        i += 1
        # Predict the best action using the trained model
        action, _states = model.predict(obs, deterministic=False)
        print("action", action)
        print("states", _states)
        # Take the action in the environment
        obs, reward, done, truncated, info = env.step(action)
        print(obs, reward, done, truncated, info)
        # Optionally, render the environment (if implemented)
        # env.render()
        if i > 3000:
            break
    return initial_state, abs(initial_state[0] - initial_state[1]), i


if __name__ == '__main__':
    env = gym.make("gymnasium_env/PianoEnv-v0", audio_file='resources/wav/c_eb_c_eb_c_eb_c.wav',
                   sound_file='resources/soundfiles/[GD] Clean Grand Mistral.sf2', sample_rate=44100, number_of_keys=20)
    print(env)
    # Box(4,) means that it is a Vector with 4 components
    print("Observation space:", env.observation_space)
    # Discrete(2) means that there is two discrete actions
    print("Action space:", env.action_space)

    # The reset method is called at the beginning of an episode
    obs, info = env.reset()
    print("obs", obs)
    print("info", info)

    check_env(env.unwrapped)
    # Sample a random action
    action = env.action_space.sample()
    print("Sampled action:", action)
    obs, reward, terminated, truncated, info = env.step(action)
    # Note the obs is a numpy array
    # info is an empty dict for now but can contain any debugging info
    # reward is a scalar
    print(obs, reward, terminated, truncated, info)

    tmp_path = "./logs"
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    # Initialize the PPO agent with a Multi-Layer Perceptron (MLP) policy
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./")
    model.set_logger(new_logger)
    # Train the agent for a specified number of timesteps
    model.learn(total_timesteps=10000, tb_log_name="first_run")

    input()
    results = []
    for i in range(10):
        initial_state, step_count, best_step_count = playout_policy(model, env)
        results.append([initial_state, step_count, best_step_count])

    for r in results:
        print("initial state", r[0], "step_count", r[1], "best_step_count", r[2])


    env.close()
    model.save("MlpPolicy")