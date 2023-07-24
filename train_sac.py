import torch
import numpy as np
from tqdm import tqdm

from envs.SimpleBertrandInflation import BertrandEnv
from replay_buffer import ReplayBuffer
from agents.sac import SACAgent

N = 2
total_timesteps = 20_000
rho = 5e-2
expected_shocks = int(total_timesteps * rho)
adjust_range = True

env = BertrandEnv(N = N, k = 2, rho = rho, A = 0.1, c = 0.01)
ob_t = env.reset()

inflation, past_prices, past_inflation = ob_t
dim_states = np.prod(inflation.shape) + np.prod(past_prices.shape) + np.prod(past_inflation.shape)
dim_actions = 1

action_low = env.price_low
action_high = env.price_high * (1.05 ** expected_shocks)

agents = [SACAgent(dim_states, dim_actions, action_low, action_high) for _ in range(N)]
buffer = ReplayBuffer(N, buffer_size = 1000000, sample_size = 64)

for timestep in tqdm(range(total_timesteps)):
    
    actions = [agent.get_action(ob_t) for agent in agents]    
    ob_t1, rewards, done, info = env.step(actions)
    
    if (ob_t1[0].item() > 0) & adjust_range:
        [agent.update_scale(env.price_low, env.price_high) for agent in agents]
    
    experience = (ob_t, actions, rewards, ob_t1, done)
    
    buffer.store_transition(*experience)
    
    if timestep > buffer.sample_size:
        for agent_idx in range(N):
            agent = agents[agent_idx]
            sample = buffer.sample(agent_idx)
            agent.update(*sample)
            
    ob_t = ob_t1