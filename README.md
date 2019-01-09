# PPO-demo
Repository uses Unity-ML Tennis as environment for Proximal Policy Optimization agent 

This project is my sample implementation of Proximal Policy Optimization with Beta distribution algorithm described in detail:
https://arxiv.org/abs/1707.06347

Please note the original paper described the PPO with Gaussain distribution but Beta is commonly accepted as a superior for RL learning. 

The environment used to present the algorithm is Multi agent Tennis from Unity-ML
You don’t have to build the environment yourself the prebuilt one included in the project will work fine - please note it’s only compatible with Unity-ML 0.4.0b NOT the current newest version. I don’t have access to the source of the environment as it was prebuilt by Udacity. 

## Environment details:
A reward of -0.01 is returned when ball touches the ground, and a reward of 0.1 if the agent hits the ball over the net. Thus, the goal of the agents is to keep the ball in play for as long as possible.
Each racket is controlled by it's own agent - because the observations of the agents overlap I decided to use a single agent in code but it consumes stacked observations from both and outputs actions for both.  

The state space has 8 dimensions:
* Vector Observation space: 8 variables corresponding to the ball and agent positions
* Vector Action space: (Continuous) Size of 2, corresponding to agent movement (forward / back / up / down).
* Visual Observations: None.
* Observations are stacked in the vector of 3 - this results in the total size of space being 24 (see model.py for details)

Two continunous actions are available, corresponding to agent movement on the tennis field. 

The problem is considered solved when the agents achieve average score of at least 0.5 over 100 episodes. 

## Installation: 
Please run pip install . in order to ensure you got all dependencies needed

To start up the project:
python -m train.py 

All hyper-paramters are in: 
config.py 

The config includes PLAY_ONLY argument which decides whether to start Agent with pre-trained weights or spend a few hours and train it from scratch :) 

More details on the project can be found in:  
[Report](/Report.md)




