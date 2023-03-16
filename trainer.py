import numpy as np
import random
from tqdm import tqdm
from utils.epsilon_greedy_policy import epsilon_greedy_policy

class Trainer():
    
    '''
    Class to train the agents. Contains the loop to train the agents and track the parameters.
    '''
    
    def __init__(self, episodes, n_steps, lr, min_epsilon, max_epsilon, decay_rate, gamma, seed):
        self.episodes = episodes
        self.n_steps = n_steps
        self.lr = lr
        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon
        self.decay_rate = decay_rate
        self.gamma = gamma
        self.seed = seed
    
    def train(self, env, agents, action_space):
        
        random.seed(self.seed)
        
        for _ in range(self.episodes):
            state = env.reset()

        for agent in agents:
            agent.update(state)

        step = 0
        done = False

        self.epsilon_list = []
        for step in tqdm(range(self.n_steps)):
            # Reduce epsilon (because we need less and less exploration)
            epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate * step)
            self.epsilon_list.append(epsilon)

            # Choose the action At using epsilon greedy policy
            action = [epsilon_greedy_policy(agent.Qtable, agent._obs_to_state[state], epsilon, action_space) for agent in agents]

            # Take action At and observe Rt+1 and St+1
            # Take the action (a) and observe the outcome state(s') and reward (r)
            new_state, reward, done, info = env.step(action)

            for idx in range(len(agents)):

                # Agregamos registro no observado
                agents[idx].update(new_state)

                # Update de Q-table
                state_idx = agents[idx]._obs_to_state[state]
                newstate_idx = agents[idx]._obs_to_state[new_state]
                Qtable_idx = agents[idx].Qtable
                Qtable_idx[state_idx][action[idx]] += self.lr * (reward[idx] + self.gamma * np.max(Qtable_idx[newstate_idx]) - Qtable_idx[state_idx][action[idx]])
                agents[idx].Qtable = Qtable_idx

            # If done, finish the episode
            if done:
                break
            
            # Our next state is the new state
            state = new_state