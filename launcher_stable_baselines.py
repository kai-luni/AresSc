import gym

from stable_baselines.common.policies import MlpPolicy, CnnLstmPolicy, CnnPolicy, CnnLnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines import PPO2
from functools import partial

from environments.env_gym import AresEnvGym

# import tensorflow as tf
# # Creates a graph.
# a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
# b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
# c = tf.matmul(a, b)
# # Creates a session with log_device_placement set to True.
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# # Runs the op.
# print(sess.run(c))
def make_sc2env(id=0, **kwargs):
    return AresEnvGym((64, 64, 3), id)


def main():
        
    env_args = dict()

    envs = SubprocVecEnv([partial(make_sc2env, id=i, **env_args) for i in range(8)])

    #pysc2_env = AresEnvGym((64, 64, 3))
    #envTwo = AresEnvGym((64, 64, 3))
    #pysc2_env_vec = DummyVecEnv([lambda: pysc2_env])  # The algorithms require a vectorized environment to run
    #model = PPO2.load("ppo_model", env=pysc2_env_vec)
    model = PPO2(CnnLstmPolicy, envs, verbose=1, nminibatches=1, cliprange=0.2)
    model.learn(total_timesteps=200000)
    model.save("ppo_model_lstm")

    # obs = env.reset()
    # for i in range(1000):
    #     action, _states = model.predict(obs)
    #     obs, rewards, dones, info = env.step(action)
    #     env.render()

if __name__ == '__main__':   
    main()