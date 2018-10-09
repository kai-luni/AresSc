
from environments.env_gym import AresEnvGym

if __name__ == '__main__': 
    import sys
    from absl import flags
    FLAGS = flags.FLAGS
    FLAGS(sys.argv)
    #main()

env = AresEnvGym((64, 64, 3), 0)

while(True):
    env.step(15)