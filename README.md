This repository contains the solution for the third project of the Udacity Deep Reinforcement Learning Nanodegree.

# The Environment

The environment for this project is the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

* After each episode,  the rewards that each agent received are added up(without discounting), to get a score for each agent. This yields 2 (potentially different) scores. Then, the maximum of these 2 scores is taken.
* This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

# How to

Install the dependancies first

* **Option 1: pipenv (recommended).** Initialize a pipenv environment by running `pipenv --three install` inside the root directory of the repo. [Pipenv](http://docs.pipenv.org/) will automatically locate the [Pipfiles](https://github.com/pypa/pipfile), create a new virtual environment and install the necessary packages.

* **Option 2: pip.** Install the needed dependencies by running `pip install -r requirements.txt` 

A solution of the environment can be obtained by running the `DDPG_Collaboration_And_Competition.ipynb`  notebook.

## Repository structure

*  `DDPG_Collaboration_And_Competition.ipynb` notebook contains a solution of the environment with single agent
*  `Report.md` contains a description of the implementation.
*  `trainer.py` contains the code for running the training of the agent over a given number of episodes
*  `ddpg_agent.py` contains the implementation of the DDPG agent
*  `model.py` contains the definition of the Actor and Critic networks
*  `actor_0/1_solution.pth` and `critic_0/1_solution.pth` contain the model weights of actor and critic networks after environment was solved, for the 2 agents





