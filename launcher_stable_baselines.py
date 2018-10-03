import gym
import os

import numpy as np

from stable_baselines.common.policies import MlpPolicy, CnnLstmPolicy, CnnPolicy, CnnLnLstmPolicy, CnnLnLstmPolicyLargerNetwork, CnnPolicyLargerLayer
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
    return Monitor(AresEnvGym((64, 64, 1), env_id), None, allow_early_resets=True)

best_mean_reward, n_steps = -np.inf, 0

def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_steps, best_mean_reward
    print("n steps " + str(n_steps))
    #print("best mean reward " + str(n_steps))
    # Print stats every 1000 calls
    if (n_steps + 1) % 10 == 0:
        print("Saving new best model")
        _locals['self'].save(str(n_steps) + '_model.pkl')
    n_steps += 1
    return False

def load(load_path, env=None, **kwargs):
    data, params = PPO2._load_from_file(load_path)
    data['policy'] = CnnPolicyLargerLayer
    model = PPO2(policy="CnnPolicyLargerLayer", env=None, _init_setup_model=False)
    model.__dict__.update(data)
    model.__dict__.update(kwargs)
    model.set_env(env)
    model.setup_model()

    restores = []
    for param, loaded_p in zip(model.params, params):
        restores.append(param.assign(loaded_p))
    model.sess.run(restores)

    return model


def main():
        
    env_args = dict()

    pysc2_env_vec = VecFrameStack(SubprocVecEnv([partial(make_sc2env, id=i, **env_args) for i in range(6)]), 4)
    #pysc2_env_vec = VecFrameStack(DummyVecEnv([partial(make_sc2env, id=0, **env_args) ]), 4)
    #pysc2_env = AresEnvGym((64, 64, 3), 1)
    #envTwo = AresEnvGym((64, 64, 3))
    #pysc2_env_vec = DummyVecEnv([lambda: pysc2_env])  # The algorithms require a vectorized environment to run

    #model = PPO2(CnnPolicyLargerLayer, pysc2_env_vec, verbose=1, nminibatches=16, lam=0.95, gamma=0.99, cliprange=0.2, tensorboard_log=log_dir)
    #model = PPO2.load("109_model.pkl", env=pysc2_env_vec, tensorboard_log=log_dir)

    # model.learn(total_timesteps=800, callback=callback)
    # model.save("ppo_model_lstm")
    model = load("729_model.pkl", env=pysc2_env_vec, tensorboard_log=log_dir)
    model.learn(total_timesteps=1000000, callback=callback)
    model.save("ppo_model_ppo_cnn")
    # obs = pysc2_env_vec.reset()
    # for i in range(1000):
    #     action, _states = model.predict(obs)
    #     obs, rewards, dones, info = pysc2_env_vec.step(action)


if __name__ == '__main__':   
    main()