from tqdm import tqdm

def train(env, agents: list, n_steps):
    
    prices_t = env.reset()
    #inflation_t = 0
    
    ob_t = prices_t
    
    for step in tqdm(range(n_steps)):
        
        action = [agent.select_action(ob_t) for agent in agents]
        new_state, reward, done, info = env.step(action)
        
        # update
        for agent in agents:
            agent.update()
        
        