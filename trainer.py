import numpy as np
from collections import deque
import torch

def trainer(agents, env, brain_name, 
            n_episodes=2000, max_t=1000, n_random_episodes=0, score_solved=0.5,
            save_model=True, model_filename='checkpoint.pth'):
    """Deep Q-Learning.
    
    Params
    ======
        agent: the agent
        env: the environment
        brain_name: unity environment brain_name
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        score_solved (float): score (averaged on the last 100 episodes) at which we consider the environment solved
        save_model (bool): if we save the model weights or not
        model_filename (str): path for saving the model weights
    """

    scores = []                        
    scores_window = deque(maxlen=100)                      
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name] 
        states = env_info.vector_observations            
        score = 0.0
        for t in range(max_t):
            if i_episode <= n_random_episodes:
                actions = np.random.randn(2, 2) # select an action (for each agent)
            else:
                # Choose action
                actions = np.zeros([2,2])
                actions[0, :] = agents[0].act(states[0])
                actions[1, :] = agents[1].act(states[1])
            
            # Send action to env, get state and reward
            env_info = env.step(actions)[brain_name]        
            next_states = env_info.vector_observations  
            rewards = env_info.rewards      
            dones = env_info.local_done
            
            # Update the agent
            for i in range(len(agents)):
                agents[i].step(states[i], actions[i], rewards[i], next_states[i], dones[i])
            
            states = next_states
            score += np.max(rewards)
            if np.any(dones):
                break 
        
        scores_window.append(score)       
        scores.append(score)              
        
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=score_solved:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            if save_model:
                torch.save(agents[0].actor_local.state_dict(), 'actor_0 ' + model_filename)
                torch.save(agents[0].critic_local.state_dict(), 'critic_0 ' + model_filename)
                torch.save(agents[1].actor_local.state_dict(), 'actor_1 ' + model_filename)
                torch.save(agents[1].critic_local.state_dict(), 'critic_1 ' + model_filename)
            break
    
    return scores, i_episode