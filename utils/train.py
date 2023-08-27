#from tqdm import tqdm
from utils.export_results import export_results
import sys
import numpy as np
import pandas as pd

def train(env, agents, buffer, N, episodes, timesteps, update_steps, inflation_start, trigger_deviation, exp_name = 'experiment'):
    
    prices_history = np.zeros((episodes, timesteps, N))
    actions_history = np.zeros((episodes, timesteps, N))
    monopoly_history = np.zeros((episodes, timesteps))
    nash_history = np.zeros((episodes, timesteps))
    rewards_history = np.zeros((episodes, timesteps, N))
    metric_history = np.zeros((episodes, timesteps))
    
    for episode in range(episodes):
        trigger_steps = 0
        ob_t = env.reset()
        for t in range(timesteps):
            actions = [agent.select_action(ob_t) for agent in agents]    
            
            # trigger deviation
            if (t > (timesteps * 2) // 3) & (trigger_deviation):
                if trigger_steps < 1000: 
                    actions[0] = env.pN
                    trigger_steps += 1
            
            ob_t1, rewards, done, info = env.step(actions)
            
            experience = (ob_t, actions, rewards, ob_t1, done)
            
            buffer.store_transition(*experience)
            
            if (t % update_steps == 0) & (t >= buffer.sample_size):
                for agent_idx in range(N):
                    agent = agents[agent_idx]
                    sample = buffer.sample(agent_idx)
                    agent.update(*sample)
            
            sys.stdout.write(f"\rExperiment: {exp_name} \t Episode: {episode + 1}/{episodes} \t Episode completion: {100 * t/timesteps:.2f} % \t Delta: {info:.2f}")
            
            ob_t = ob_t1
        
        #export_results(env.prices_history[env.k:], env.quantities_history,
        #            env.monopoly_history[1:], env.nash_history[1:], 
        #            env.rewards_history, env.metric_history, 
        #            env.pi_N_history[1:], env.pi_M_history[1:],
        #            env.costs_history[env.v+1:], exp_name)
        
        prices_history[episode] = np.array(env.prices_history)[env.k:]
        monopoly_history[episode] = np.array(env.monopoly_history)
        nash_history[episode] = np.array(env.nash_history)
        rewards_history[episode] = np.array(env.rewards_history)
        metric_history[episode] = np.array(env.metric_history)
    
    # export   
    prices_history = np.mean(prices_history, axis = 0)
    actions_history = np.mean(actions_history, axis = 0)
    monopoly_history = np.mean(monopoly_history, axis = 0)
    nash_history = np.mean(nash_history, axis = 0)
    rewards_history = np.mean(rewards_history, axis = 0)
    metric_history = np.mean(metric_history, axis = 0)
    
    results = pd.DataFrame({'monopoly': monopoly_history,
                        'nash': nash_history,
                        'metric': metric_history
                        })

    for agent in range(env.N):
        results[f'prices_{agent}'] = prices_history[:, agent]
        results[f'actions_{agent}'] = actions_history[:, agent]
        results[f'rewards_{agent}'] = rewards_history[:, agent]
        
    results.to_csv(f'metrics/{exp_name}.csv')