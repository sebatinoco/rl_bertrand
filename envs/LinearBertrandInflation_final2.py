import numpy as np
import gymnasium as gym
import torch

class LinearBertrandEnv():
    def __init__(self, N, k, rho, A = 3, e = 1, c = 1, v = 3, xi = 0.2):
        
        self.N = N # number of agents
        self.k = k # past periods to observe
        self.rho = rho # probability of changing prices
        self.c = c # marginal cost
        self.v = v # length of past inflations to predict current inflation
        self.xi = xi # price limit deflactor
        assert v >= k, 'v must be greater or equal than k'
        
        self.A = A # highest disposition to pay
        self.e = e # elasticity of demmand
        
        self.inflation_history = [] # inflation history
        self.prices_history = [] # prices history
        self.quantities_history = [] # quantities history
        self.costs_history = [] # costs history
        self.nash_history = [] # nash prices history
        self.monopoly_history = [] # monopoly prices history
        self.rewards_history = [] # intrinsic rewards history
        self.metric_history = [] # collussion metric history
        self.pi_N_history = [] # nash utilities history
        self.pi_M_history = [] # monopoly utilities history
        self.inflation_start = False
        
        self.inflation_model = torch.jit.load('inflation/inflation_model.pt')
        self.inflation_model.eval()
        
        # set action boundaries
        self.pN = self.c # get nash price
        self.pM = (self.A + self.c) / 2 # get monopoly price
        
        self.nash_history += [self.pN]
        self.monopoly_history += [self.pM]
        
        self.pi_N = (self.pN - self.c) * self.demand([self.pN])[0]
        self.pi_M = (self.pM - self.c) * self.demand([self.pM])[0]
        
        self.pi_N_history += [self.pi_N]
        self.pi_M_history += [self.pi_M]

        assert self.pi_M > self.pi_N, f'monopoly profits should be higher than nash profits: {self.pi_N} vs {self.pi_M}'
        
        self.price_high = self.pM * (1 + self.xi)
        self.price_low = self.pN * (1 - self.xi)
        
    def step(self, action):
        
        '''
        Computes a step over the environment. Receives an action (array of prices) and return a tuple of (observation, reward, done, _)
        action: array of prices (np.array)
        '''
        
        # compute quantities
        quantities = self.demand(action)
        self.quantities_history.append(quantities)
        
        # intrinsic reward: (p - c) * q
        reward = [(action[agent] - self.c) * quantities[agent] for agent in range(self.N)]
        self.rewards_history.append(reward)
        
        # update price history
        self.prices_history.append(action)
        
        # obtain inflation
        inflation = self.get_inflation()
        self.inflation_history.append(inflation)
        
        # gather observation
        inflation = np.array(inflation, ndmin = 2, dtype = 'float32')
        cost = np.array(self.c, ndmin = 2, dtype = 'float32')
        past_prices = np.array(self.prices_history[-self.k:], dtype = 'float32')
        past_inflation = np.array(self.inflation_history[-self.k:], ndmin = 2, dtype = 'float32').T
        past_costs = np.array(self.costs_history[-self.k:], ndmin = 2, dtype = 'float32').T
        ob_t1 = (inflation, cost, past_prices, past_inflation, past_costs)
         
        done = False
        info = self.get_metric(reward)
        
        return ob_t1, reward, done, info
    
    def reset(self):
        
        '''
        Resets the environment.
        '''
        
        self.prices_space = gym.spaces.Box(low = self.price_low, high = self.price_high, shape = (self.k, self.N), dtype = float) # prices space
        self.inflation_space = gym.spaces.Box(low = 0.015, high = 0.035, shape = (self.v,), dtype = float) # inflation space
        
        self.prices_history = [list(prices) for prices in self.prices_space.sample()] # init prices
        self.inflation_history = list(self.inflation_space.sample()) # init inflation
        
        self.costs_history = [self.c]
        for inflation in self.inflation_history[::-1]:
            self.costs_history = [self.costs_history[0] / (1 + inflation)] + self.costs_history
        
        ob_t = (
            np.array(inflation, ndmin = 2, dtype = 'float32'), 
            np.array(self.c, ndmin = 2, dtype = 'float32'), 
            np.array(self.prices_history, ndmin = 2, dtype = 'float32'), 
            np.array(self.inflation_history[-self.k:], ndmin = 2, dtype = 'float32').T, 
            np.array(self.costs_history[-self.k:], ndmin = 2, dtype = 'float32').T
            )
        
        return ob_t
    
    def demand(self, prices):

        '''
        Returns the sold quantity in function of the prices set.
        prices: Array of prices offered by agents (np.array)
        '''

        p_min = np.min(prices)
        q_min = self.A - p_min * self.e

        quantities = [q_min if p == p_min and p < self.A else 0 for p in prices]
        
        #eq_count = np.count_nonzero(prices == p_min) # count p_min ocurrences
        #quantities = [q / eq_count for q in quantities]

        return quantities
    
    def get_inflation(self):
        sample = np.random.rand()
        
        inflation_t = 0
        if (sample < self.rho) & (self.inflation_start):
            
            with torch.no_grad():
                inflation_values = np.array(self.inflation_history) # transform to array
                inflation_values = inflation_values[inflation_values != 0][-self.v]
                inflation_values = torch.tensor(inflation_values).reshape(1, -1, 1).float()
                inflation_t = float(self.inflation_model(inflation_values).squeeze())
            
            dc = self.c * (inflation_t)
            
            self.c += dc # adjust marginal cost
            self.A += dc # dc = dA
            
            #print('Calculating new equilibria...')
            self.pN = self.c # get nash price
            self.pM = (self.A + self.c) / 2 # get monopoly price
            
            self.pi_N = (self.pN - self.c) * self.demand([self.pN])[0]
            self.pi_M = (self.pM - self.c) * self.demand([self.pM])[0]
            assert self.pi_M > self.pi_N, "monopoly profits should be higher than nash profits"
            
            self.price_high = self.pM * (1 + self.xi)
            self.price_low = self.pN * (1 - self.xi)
            
            #self.inflation_history.append(inflation_t) # store inflation
        
        self.costs_history += [self.c]
        self.nash_history += [self.pN]
        self.monopoly_history += [self.pM]
        self.pi_N_history += [self.pi_N]
        self.pi_M_history += [self.pi_M]
            
        return inflation_t
    
    def get_metric(self, rewards, window = 1000):
        
        metric = (np.mean(rewards) - self.pi_N) / (self.pi_M - self.pi_N)
        self.metric_history.append(metric)
        
        return np.mean(self.metric_history[-window:])