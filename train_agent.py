import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from replay_buffer import ReplayBuffer
from utils.parse_args import parse_args
from utils.plot_metrics import plot_metrics

#from envs.BertrandInflation import BertrandEnv
#from envs.BertrandInflation_cost import BertrandEnv
#from envs.SimpleBertrandInflation import BertrandEnv
from envs.SimpleBertrandInflation_final import BertrandEnv

from agents.ddpg import DDPGAgent
from agents.dqn import DQNAgent # precios no se estan graficando bien
from agents.sac import SACAgent
#from agents.ddpg_cost import DDPGAgent

models_dict = {'sac': SACAgent, 'ddpg': DDPGAgent, 'dqn': DQNAgent}

if __name__ == '__main__':
    
    args = parse_args()

    env = BertrandEnv(N = args['N'], k = args['k'], rho = args['rho'], v = args['v'], xi = args['xi'])
    ob_t = env.reset()
    
    dim_states = env.N if args['use_lstm'] else env.k * env.N + env.k + 1
    dim_actions = args['n_actions'] if args['n_actions'] is not None else 1

    model = models_dict[args['model']]
    agents = [model(dim_states, dim_actions, env.price_low, env.price_high) for _ in range(args['N'])]
    
    buffer = ReplayBuffer(N = args['N'], buffer_size = args['buffer_size'], sample_size = args['sample_size'])

    plot_dim = (2, 3) if args['plot_loss'] else (1, 3)
    fig, axes = plt.subplots(*plot_dim, figsize = (16, 6) if args['plot_loss'] else (16, 4))
    axes = np.array(axes, ndmin = 2)

    for t in tqdm(range(args['timesteps'])):
        actions = [agent.select_action(ob_t) for agent in agents]    
        
        ob_t1, rewards, done, info = env.step(actions)
        
        experience = (ob_t, actions, rewards, ob_t1, done)
        
        buffer.store_transition(*experience)
        
        if (t % args['update_steps'] == 0) & (t >= args['learning_start']) & (t >= args['sample_size']):
            for agent_idx in range(args['N']):
                agent = agents[agent_idx]
                sample = buffer.sample(agent_idx)
                agent.update(*sample)
        
            if t % args['plot_steps'] == 0:
                plot_args = (fig, axes, env.prices_history, env.monopoly_history, env.nash_history, env.rewards_history, env.metric_history, args['window'])
                plot_metrics(*plot_args, agent.actor_loss, agent.Q_loss) if args['plot_loss'] else plot_metrics(*plot_args)     
        
        ob_t = ob_t1
        
    plt.savefig(f"figures/{args['exp_name']}.pdf")