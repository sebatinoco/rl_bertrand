import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
#from agents.ddpg import DDPGAgent
from agents.ddpg_epsilongreedy import DDPGAgent
from envs.BertrandInflation import BertrandEnv
from replay_buffer import ReplayBuffer
from utils.parse_args import parse_args
from utils.plot_metrics import plot_metrics

if __name__ == '__main__':
    
    args = parse_args()

    env = BertrandEnv(N = args['N'], k = args['k'], rho = args['rho'], v = args['v'], xi = args['xi'])
    ob_t = env.reset()

    agents = [DDPGAgent(args['N'], args['actor_lr'], args['Q_lr'], args['gamma'], args['tau']) for _ in range(args['N'])]
    buffer = ReplayBuffer(N = args['N'], buffer_size = args['buffer_size'], sample_size = args['sample_size'])

    #plot_dim = (1, 1+2) if args['plot_loss'] else (1, 2)
    plot_dim = (2, 2) if args['plot_loss'] else (1, 2)
    fig, axes = plt.subplots(*plot_dim, figsize = (16, 6))
    axes = np.array(axes, ndmin = 2)

    for t in tqdm(range(args['timesteps'])):
        actions = [agent.select_action(ob_t, env.price_high, env.price_low) for agent in agents]    
        
        ob_t1, rewards, done, info = env.step(actions)
        
        experience = (ob_t, rewards, actions, ob_t1, done)
        
        buffer.store_transition(*experience)
        
        if (t % args['update_steps'] == 0) & (t >= args['learning_start']) & (t >= args['sample_size']):
            for agent_idx in range(args['N']):
                agent = agents[agent_idx]
                sample = buffer.sample(agent_idx)
                agent.update(*sample)
        
            if t % args['plot_steps'] == 0:
                plot_args = (fig, axes, env.prices_history, env.monopoly_history, env.nash_history, env.rewards_history, args['rolling'])
                plot_metrics(*plot_args, agent.actor_loss, agent.Q_loss) if args['plot_loss'] else plot_metrics(*plot_args)     
        
        ob_t = ob_t1
        
    plt.savefig(f"figures/{args['exp_name']}.pdf")