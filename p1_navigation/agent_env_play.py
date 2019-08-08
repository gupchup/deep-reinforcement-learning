from dqn_agent import Agent
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import torch

def dqn(agent, env, n_episodes=20, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, model_file="checkpoint.pth", debug=False):
    """Deep Q-Learning.
    
    Params
    ======
        agent: The agent that interacts with the environment.
        env: The environment in which actions are taking.
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    brain_name = env.brain_names[0]
    
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        state = env_info.vector_observations[0]            # get the current state
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps).astype(int)
            if debug:
                print("Episode: {0}, t: {1}, Action: {2} \r".format(i_episode, t, action))
            env_action_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_action_info.vector_observations[0]   # get the next state
            reward = env_action_info.rewards[0]                   # get the reward
            done = env_action_info.local_done[0]                  # see if episode has finished
            # next_state, reward, done, _ = env.step(action)[brain_name]
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=14.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), model_file)
            break
            
    return scores