# Learning Algorithm

The learning algorithm I used for this project is a **Deep Deterministic Policy Gradient**, whose full description can be found in [the original paper](https://arxiv.org/pdf/1509.02971.pdf).

DDPG is an actor-critic algorithm, and its basically an extension of DQN to continuous actions spaces.

The actor is used to approximate the optimal policy deterministically, differently from other actor-critic algorithms that learn stochastic policies (i.e. a probability distribution over the actions).

The critic learns to evaluate the optimal action-value functions by using the actor estimated best actions.

## Hyperparameters and model architecture

The environment has been solved the following hyperparameters, for both agents:

```
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 512        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
N_RANDOM_EPISODES = 400 # n episodes with random actions
```

With respect to previous project, here I found critical for the solution to the environment the introduction of a few random actions at the beginning of the training, for a number `N_RANDOM_EPISODES` of episodes.

The model architecture was the following, for both agents:

* Actor: 1 hidden layer for the actor, with 256 units
* Critic: 2 hidden layers, with [256, 256] units respectively

## Plot of Rewards

Environment was solved in 1477 episodes, with a quite stable learning trajectory.

![DDPG_Solution_Scores](/Users/max/Google_Drive/learning/MOOCs/DRLND-Udacity/drlnd_p3_collaboration_and_competition/images/scores.png)

```
Episode 100	Average Score: 0.02
Episode 200	Average Score: 0.03
Episode 300	Average Score: 0.02
Episode 400	Average Score: 0.03
Episode 500	Average Score: 0.02
Episode 600	Average Score: 0.03
Episode 700	Average Score: 0.08
Episode 800	Average Score: 0.10
Episode 900	Average Score: 0.13
Episode 1000	Average Score: 0.18
Episode 1100	Average Score: 0.42
Episode 1200	Average Score: 0.40
Episode 1300	Average Score: 0.41
Episode 1400	Average Score: 0.31
Episode 1477	Average Score: 0.50
Environment solved in 1477 episodes!	Average Score: 0.50
```




## Ideas for Future Work

- Study the impact of Hyperparameters
- Trying other algorithms like PPO, A3C or D4PG
- Implement an evolutionary strategies and compare it with Actor-Critic gradient-based approaches