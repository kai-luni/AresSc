import tensorflow as tf
from baselines.common import policies, models, cmd_util
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.run import get_learn_function
from baselines.ppo2.ppo2 import learn as ppo_learn
from functools import partial


from environments.env_gym import AresEnvGym


def make_sc2env(env_id=0, **kwargs):
    return AresEnvGym((64, 64, 3), env_id)

def main():
    env_args = dict()
    network_kwargs = dict(nlstm=384)

    # create vectorized environment
    pysc2_env_vec = SubprocVecEnv([partial(make_sc2env, id=i, **env_args) for i in range(4)])
    #policy = policies.build_policy(pysc2_env_vec, models.cnn_small())(nbatch=1, nsteps=1)
    # kwargs = dict(value_network='copy')
    # learn_fn = lambda e: get_learn_function('ppo2')(env=pysc2_env_vec, **kwargs)
    model = ppo_learn(network="cnn_lstm", env=pysc2_env_vec, total_timesteps=1500000, gamma=0.995, nsteps=256, nminibatches=1, **network_kwargs)
    model.save("lstm_ppo")

if __name__ == '__main__':   
    main()
