from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common import policies, models, cmd_util
from baselines.ppo2.ppo2 import Model as ppo_model
from baselines.bench import Monitor
from functools import partial

from environments.env_gym import AresEnvGym
from pysc2.env import sc2_env





import imageio
import numpy as np


# model = A2C(MlpPolicy, "LunarLander-v2").learn(100000)

# images = []
# obs = model.env.reset()
# img = model.env.render(mode='rgb_array')
# for i in range(350):
#     images.append(img)
#     action, _ = model.predict(obs)
#     obs, _, _ ,_ = model.env.step(action)
#     img = model.env.render(mode='rgb_array')

# imageio.mimsave('lander_a2c.gif', [np.array(img[0]) for i, img in enumerate(images) if i%2 == 0], fps=29)
difficulty = sc2_env.Difficulty.medium


def make_sc2env(env_id=0, **kwargs):
    import sys
    from absl import flags
    FLAGS = flags.FLAGS
    FLAGS(sys.argv)
    return Monitor(AresEnvGym((64, 64, 3), env_id, difficulty=difficulty), 'log.csv', allow_early_resets=True)

def play():
    env_args = dict()
    network_kwargs = dict(nlstm=512)

    # create vectorized environment
    pysc2_env_vec = DummyVecEnv([partial(make_sc2env, id=i, **env_args) for i in range(1)])

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
    model.load("4860_ppo_cnn_lstm_512_medium")

    images = []
    ob = pysc2_env_vec.reset()
    state = model.initial_state
    done = [False]
    step_counter = 0

    # run a single episode until the end (i.e. until done)
    while True:
        #print(step_counter)
        images.append(ob)
        action, _, state, _ = model.step(ob, S=state, M=done)
        ob, _, done, stats = pysc2_env_vec.step(action)
        step_counter += 1
        if(done[0]):
            imageio.mimsave(str(stats[0]["final_reward"]) + "_" + str(difficulty) + '.gif', [np.array(img[0]) for i, img in enumerate(images) if i%2 == 0], fps=4)
            images = []
    
    

if __name__ == '__main__':   
    play()