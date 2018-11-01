# AresSc - A Deep Learning Approach for a Star Craft 2 bot

The goals of this project are the following:
* See how Unsupervised Deep Learning can be utilized for complex tasks
* Use creativity to get impressive results out of these new technologies
* Only source of information for the network(s): image data
* The final goal would be to actually use filmed data from a cam and let the network play

## Version

Check the documentation folder for more details, right now this is **Alpha 3**

## How does it work

Right now there are two bots playing by turns on one side:
* The first one is taking care of building the base and is completely scripted
* The second one is a neural network that gets feeded the minimap and hast an action space of 65 to attack 64 points on the 
minimap or do nothing

## Next steps

A third bot which imput is the game screen and can directly atach ot this. It can also decide when to stop fighting and the 
algorithm will go back to 'Macro Mode'.

## Try it out

You will need to look it up in launcher_baselines.py. There are loads dependencies. For training the headless agent for linux is 
recommended.

Another note, I ran into trouble to install atari-py under windows, which is a dependency in gym. You can install gym without 
atari-py as for this program just the gym-environment is need.
