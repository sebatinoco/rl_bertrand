import numpy as np
from tqdm import tqdm
from utils.epsilon_greedy_policy import epsilon_greedy_policy

def train(env, agents, episodes, n_steps, lr, min_epsilon, max_epsilon, decay_rate, gamma, action_space):

    for episode in range(episodes):
        state = env.reset()

    for agent in agents:
        agent.update(state)

    step = 0
    done = False

    for step in tqdm(range(n_steps)):
        # Reduce epsilon (because we need less and less exploration)
        epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*step)

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
            Qtable_idx[state_idx][action[idx]] += lr * (reward[idx] + gamma * np.max(Qtable_idx[newstate_idx]) - Qtable_idx[state_idx][action[idx]])
            agents[idx].Qtable = Qtable_idx

        # If done, finish the episode
        if done:
            break
        
        # Our next state is the new state
        state = new_state