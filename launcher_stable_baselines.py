import gym
import os

import numpy as np

from stable_baselines.common.policies import MlpPolicy, CnnLstmPolicy, CnnPolicy, CnnLnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
from stable_baselines import PPO2
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from functools import partial

from environments.env_gym import AresEnvGym


# Create log dir
log_dir = "log/"
os.makedirs(log_dir, exist_ok=True)


def make_sc2env(env_id=0, **kwargs):
    return Monitor(AresEnvGym((64, 64, 3), env_id), log_dir, allow_early_resets=True)

best_mean_reward, n_steps = -np.inf, 0

def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_steps, best_mean_reward
    # Print stats every 1000 calls
    if (n_steps + 1) % 10000 == 0:
        # Evaluate policy performance
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        if len(x) > 0:
            mean_reward = np.mean(y[-100:])
            print(x[-1], 'timesteps')
            print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

            # New best model, you could save the agent here
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                # Example for saving best model
        print("Saving new best model")
        _locals['self'].save(log_dir + 'best_model.pkl')
    n_steps += 1
    return False


class CustomPolicy(CnnLnLstmPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, **kwargs):
        super(CustomPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, **kwargs, layers=[512, 400], n_lstm=512)


def main():
        
    env_args = dict()

    pysc2_env_vec = VecFrameStack(SubprocVecEnv([partial(make_sc2env, id=i, **env_args) for i in range(10)]), 8)

    #pysc2_env = AresEnvGym((64, 64, 3), 1)
    #envTwo = AresEnvGym((64, 64, 3))
    #pysc2_env_vec = DummyVecEnv([lambda: pysc2_env])  # The algorithms require a vectorized environment to run
    #model = PPO2.load("ppo_model_lstm", env=pysc2_env_vec)

    model = PPO2(CustomPolicy, pysc2_env_vec, verbose=1, nminibatches=1, cliprange=0.2)
    model.learn(total_timesteps=600000, callback=callback)
    model.save("ppo_model_lstm")
    # obs = pysc2_env_vec.reset()
    # for i in range(1000):
    #     action, _states = model.predict(obs)
    #     obs, rewards, dones, info = pysc2_env_vec.step(action)


if __name__ == '__main__':   
    main()