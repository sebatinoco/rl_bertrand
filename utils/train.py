from tqdm import tqdm
import pandas as pd
from utils.export_results import export_results

def train(env, agents, buffer, N, timesteps, learning_start, update_steps, exp_name = 'experiment'):

    ob_t = env.reset()

    for t in tqdm(range(timesteps)):
        actions = [agent.select_action(ob_t) for agent in agents]    
        
        ob_t1, rewards, done, info = env.step(actions)
        
        experience = (ob_t, actions, rewards, ob_t1, done)
        
        buffer.store_transition(*experience)
        
        if (t % update_steps == 0) & (t >= learning_start):
            for agent_idx in range(N):
                agent = agents[agent_idx]
                sample = buffer.sample(agent_idx)
                agent.update(*sample)
        
        ob_t = ob_t1
    
    # export results
    export_results(env.prices_history[1:], env.monopoly_history, env.nash_history, env.rewards_history, env.metric_history, exp_name)