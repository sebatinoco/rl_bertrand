import matplotlib.pyplot as plt
from tqdm import tqdm
from agents.ddpg import DDPGAgent
from envs.new_env import BertrandEnv
from replay_buffer import ReplayBuffer
from utils.parse_args import parse_args
from utils.plot_metrics import plot_metrics

if __name__ == '__main__':
    
    args = parse_args()

    env = BertrandEnv(N = args['N'], k = args['k'], rho = args['rho'], v = args['v'], xi = args['xi'])
    ob_t = env.reset()

    agents = [DDPGAgent(args['N'], env.price_high, env.price_low) for _ in range(args['N'])]
    buffer = ReplayBuffer(N = args['N'], buffer_size = args['buffer_size'], sample_size = args['sample_size'])

    fig, axes = plt.subplots(1, figsize = (12, 4))
    axes = [axes]

    for t in tqdm(range(args['timesteps'])):
        actions = [agent.select_action(ob_t) for agent in agents]    
        
        ob_t1, rewards, done, info = env.step(actions)
        
        experience = (ob_t, rewards, actions, ob_t1, done)
        
        buffer.store_transition(*experience)
        
        if (t % args['update_steps'] == 0) & (t >= args['learning_start']):
            for agent_idx in range(args['N']):
                agent = agents[agent_idx]
                sample = buffer.sample(agent_idx)
                agent.update(*sample)
        
            if t % args['plot_steps'] == 0:
                plot_metrics(fig, axes, env.prices_history, env.monopoly_history, env.nash_history, args['rolling'])
        
        ob_t = ob_t1