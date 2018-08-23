from absl import app

import matplotlib.pyplot as plt

import os
import pickle

def main(arg):
    if(not os.path.isfile('model/episodes.p') or not os.path.isfile('model/losses.p') or not os.path.isfile('model/game_scores.p')):
        print("at least one important file is missing")
        return
    
    episodes = pickle.load(open('model/episodes.p', 'rb'))
    losses = pickle.load(open('model/losses.p', 'rb'))
    game_scores = pickle.load(open('model/game_scores.p', 'rb'))

    plt.plot(episodes, losses)

    plt.axis([0, episodes[len(losses)-1], 0, 0.004])
    plt.show()

if __name__ == "__main__":
  app.run(main)