from absl import app

import matplotlib.pyplot as plt

import os
import pickle
import numpy as np

def norm(x):
    return np.sqrt(np.dot(x, x))

def main(arg):
    if(not os.path.isfile('model/episodes.p') or not os.path.isfile('model/losses.p') or not os.path.isfile('model/game_scores.p')):
        print("at least one important file is missing")
        return
    
    episodes = np.asarray(pickle.load(open('model/episodes.p', 'rb')))
    losses = np.asarray(pickle.load(open('model/losses.p', 'rb')))
    game_scores = np.array(pickle.load(open('model/game_scores.p', 'rb')))
    print(type(episodes))
    print(type(losses))
    loss_list = []
    for i in range(len(losses)):
        loss_list.append(losses[i][1])

    plt.plot(episodes, game_scores)

    #plt.axis([0, episodes[len(losses)-1], 0, 0.004])
    plt.show()

if __name__ == "__main__":
  app.run(main)

