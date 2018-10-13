import tensorflow as tf
from baselines.common import policies, models, cmd_util
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.run import get_learn_function
from baselines.bench import Monitor
from baselines.ppo2.ppo2 import Model as ppo_model
from functools import partial

from baselines_ares.custom_ppo2 import learn as ppo_learn
from environments.env_gym import AresEnvGym

def make_sc2env(env_id=0, **kwargs):
    import sys
    from absl import flags
    FLAGS = flags.FLAGS
    FLAGS(sys.argv)
    return Monitor(AresEnvGym((64, 64, 3), env_id), 'log.csv', allow_early_resets=True)

def train():
    env_args = dict()
    network_kwargs = dict(nlstm=512)
    number_envs = 8
    pysc2_env_vec = SubprocVecEnv([partial(make_sc2env, id=i, **env_args) for i in range(number_envs)])
    model = ppo_learn(network="cnn_lstm", env=pysc2_env_vec, total_timesteps=1500000, gamma=0.995, nsteps=192, nminibatches=number_envs, load_path="1420_ppo_cnn_lstm_384_easy", **network_kwargs)
    model.save("lstm_ppo")

def play():
    env_args = dict()
    network_kwargs = dict(nlstm=384)

    # create vectorized environment
    pysc2_env_vec = SubprocVecEnv([partial(make_sc2env, id=i, **env_args) for i in range(1)])

    policy = policies.build_policy(pysc2_env_vec, "cnn_lstm", **network_kwargs)
    nenvs = pysc2_env_vec.num_envs
    # Calculate the batch_size
    nsteps=256
    nminibatches=1
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches
    ent_coef=0.0
    vf_coef=0.5
    max_grad_norm=0.5

    make_model = lambda : ppo_model(policy=policy, ob_space=(64, 64, 3), ac_space=65, nbatch_act=nenvs, nbatch_train=nbatch_train,
                    nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm)
    model = make_model()
    model.load("3780_ppo_cnn_lstm_384_easy")

    ob = pysc2_env_vec.reset()
    state = model.initial_state
    done = [False]
    step_counter = 0

    # run a single episode until the end (i.e. until done)
    while True:
        #print(step_counter)
        action, _, state, _ = model.step(ob, S=state, M=done)
        ob, reward, done, _ = pysc2_env_vec.step(action)
        step_counter += 1





    # policy = policies.build_policy(pysc2_env_vec, models.cnn_small())(nbatch=1, nsteps=1)
    # model.load(load_path)
    
    # kwargs = dict(value_network='copy')
    # learn_fn = lambda e: get_learn_function('ppo2')(env=pysc2_env_vec, **kwargs)
    #model = ppo_learn(network="cnn_lstm", env=pysc2_env_vec, total_timesteps=1500000, gamma=0.995, nsteps=256, nminibatches=1, **network_kwargs)
    # model.save("lstm_ppo")

if __name__ == '__main__':   
    train()
