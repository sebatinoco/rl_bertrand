#from tqdm import tqdm
from utils.export_results import export_results
import sys

def train(env, agents, buffer, N, timesteps, update_steps, inflation_start, trigger_deviation, exp_name = 'experiment'):

    ob_t = env.reset()

    for t in range(timesteps):
        actions = [agent.select_action(ob_t) for agent in agents]    
        
        # trigger deviation
        if (t > timesteps // 2) & (trigger_deviation):
            actions[0] = env.pN
        
        # start changing prices        
        if t > inflation_start:
            env.inflation_start = True
        
        ob_t1, rewards, done, info = env.step(actions)
        
        experience = (ob_t, actions, rewards, ob_t1, done)
        
        buffer.store_transition(*experience)
        
        if (t % update_steps == 0) & (t >= buffer.sample_size):
            for agent_idx in range(N):
                agent = agents[agent_idx]
                sample = buffer.sample(agent_idx)
                agent.update(*sample)
        
        sys.stdout.write(f"\rExperiment: {exp_name} \t Training completion: {100 * t/timesteps:.2f} % \t Delta: {info:.2f}")
        
        ob_t = ob_t1
    
    # export results
    #export_results(env.prices_history[env.k:], env.quantities_history,
    #               env.monopoly_history[1:], env.nash_history[1:], 
    #               env.rewards_history, env.metric_history, 
    #               #env.pi_N_history, env.pi_M_history,
    #               env.costs_history[env.k:], exp_name)
    
    export_results(env.prices_history[env.k:], env.quantities_history,
                   env.monopoly_history[1:], env.nash_history[1:], 
                   env.rewards_history, env.metric_history, 
                   env.pi_N_history[1:], env.pi_M_history[1:],
                   env.costs_history[env.v+1:], exp_name)