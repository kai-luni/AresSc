from rl.core import Processor
from PIL import Image

import numpy as np

class AresProcessor(Processor):
    def __init__(self, input_shape):
        self.input_shape = input_shape
    def process_observation(self, obs):
        rgb_minimap = obs.observation['rgb_minimap']
        assert rgb_minimap.ndim == 3  # (height, width, channel)
        # img = Image.fromarray(rgb_minimap.astype('uint8'))
        # img = img.convert('L')  # convert to grayscale
        # processed_observation = np.array(img)
        assert rgb_minimap.shape == self.input_shape
        return rgb_minimap.astype('uint8')  # saves storage in experience memory

    def process_state_batch(self, batch):
        # We could perform this processing step in `process_observation`. In this case, however,
        # we would need to store a `float32` array instead, which is 4x more memory intensive than
        # an `uint8` array. This matters if we store 1M observations.
        # for i in range(len(batch)):
        #     test = batch[i]
        #     print(test)
        processed_batch = (batch.astype('float32') / 128.)-1.
        return processed_batch

    def process_reward(self, reward):
        #return np.clip(reward, -1., 1.)
        return reward