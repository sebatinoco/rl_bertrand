import numpy as np
import gymnasium as gym
from scipy.optimize import minimize, fsolve
import torch

import optuna
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.WARNING)

class BertrandEnv():
    def __init__(self, N, k, rho, mu = 1, a = None, a_0 = 0, a_index = 1, c = 1, v = 3, xi = 0.2):
        
        self.N = N # number of agents
        self.k = k # past periods to observe
        self.rho = rho # probability of changing prices
        self.a_0 = a_0 # base vertical differentiation index
        self.mu = mu # horizontal differentiation index
        self.a_index = a_index # vertical differentiation indexes
        self.c = c # marginal cost
        self.v = v # length of past inflations to predict current inflation
        self.xi = xi # price limit deflactor
        
        assert v >= k, 'v must be greater or equal than k'

        # vertical diff
        self.a = np.array(a)
        if a is None:
            self.a = np.array([c + a_index] * N)
        assert len(self.a) == N, 'self.a must be of equal size as N'
        
        self.inflation_history = [] # inflation history
        self.prices_history = [] # prices history
        self.nash_history = [] # nash prices history
        self.monopoly_history = [] # monopoly prices history
        self.rewards_history = []
        
        self.inflation_model = torch.jit.load('inflation/inflation_model.pt')
        self.inflation_model.eval()
        
    def get_nash(self):
        def nash(p):

            '''
            Nash problem. Containes the derivatives of each agent with respect to its price.
            '''
            
            assert len(self.a) == len(p), "a must be equal size to p"

            sum_denominator = np.exp(self.a_0 / self.mu)
            for i in range(len(p)):
                sum_denominator += np.exp((self.a[i] - p[i]) / self.mu)

            result = []
            for i in range(len(p)):
                first_term = np.exp((self.a[i] - p[i]) / self.mu) / sum_denominator
                second_term = (np.exp((self.a[i] - p[i]) / self.mu) * (p[i] - self.c)) / (self.mu * sum_denominator)
                third_term = (p[i] - self.c) / self.mu

                fn = first_term * (1 + second_term - third_term)
                result.append(fn)

            return result
        
        nash_solution = fsolve(nash, x0 = [1.0] * self.N)
        
        assert all(round(price, 4) == round(nash_solution[0], 4) for price in nash_solution), \
        f"Nash price should be unique: {nash_solution}" # all prices are the same
        
        pN = nash_solution[0] # float
        
        return pN
    
    def get_monopoly(self):
        
        def monopoly(p):
            return -(p - self.c) * self.demand(p)
        
        def objective(trial):
            solution = minimize(monopoly, x0 = trial.suggest_float('x0', 0, 10, step = 0.1)).x
            return monopoly(solution)

        direction = 'minimize'
        study = optuna.create_study(direction = direction, sampler = TPESampler())

        study.optimize(objective, n_trials = 100)

        x0 = study.best_trial.params['x0']
        
        pM = minimize(monopoly, x0 = x0).x[0] # float
        
        #print(f'Monopoly Price: {pM:.2f}')

        return pM
    
    def step(self, action):
        
        '''
        Computes a step over the environment. Receives an action (array of prices) and return a tuple of (observation, reward, done, _)
        action: array of prices (np.array)
        '''
        
        # compute quantities
        reward = self.demand(p = action)
        # intrinsic reward: (p - c) * q
        reward = [(action[agent] - self.c) * reward[agent] for agent in range(self.N)]
        self.rewards_history.append(reward)
        
        # update price history
        self.prices_history.append(action)
        
        # obtain inflation
        inflation = self.get_inflation()
        self.inflation_history.append(inflation)
        
        # gather observation
        inflation = np.array(inflation, ndmin = 2, dtype = 'float32')
        past_prices = np.array(self.prices_history[-self.k:], dtype = 'float32')
        past_inflation = np.array(self.inflation_history[-self.k:], ndmin = 2, dtype = 'float32').T
        ob_t1 = (inflation, past_prices, past_inflation)
         
        done = False
        info = None
        
        return ob_t1, reward, done, info
    
    def reset(self):
        
        '''
        Resets the environment.
        '''
        
        self.pN = self.get_nash() # get nash price
        self.pM = self.get_monopoly() # get monopoly price
        
        self.price_high = self.pM * (1 + self.xi)
        self.price_low = self.pN * (1 - self.xi)
        
        self.prices_space = gym.spaces.Box(low = self.price_low, high = self.price_high, shape = (self.k, self.N), dtype = float) # prices space
        self.inflation_space = gym.spaces.Box(low = 1.5, high = 3.5, shape = (self.v,), dtype = float) # inflation space
        
        past_prices = [list(prices) for prices in self.prices_space.sample()] # init prices
        past_inflation = list(self.inflation_space.sample()) # init inflation
        
        self.prices_history = past_prices # store prices
        self.inflation_history = past_inflation # store inflation
        
        inflation = self.get_inflation() # obtain inflation
        
        ob_t = (np.array(inflation, ndmin = 2, dtype = 'float32'), 
                np.array(past_prices, ndmin = 2, dtype = 'float32'), 
                np.array(past_inflation[-self.k:], ndmin = 2, dtype = 'float32').T)
        
        return ob_t
    
    def demand(self, p):

        '''
        Returns the sold quantity in function of the vertical and horizontal differentiation, as per the prices set.
        prices: Array of prices offered by agents (np.array)
        '''

        denominator = np.sum(np.exp((self.a - p) / self.mu)) + np.exp(self.a_0 / self.mu)

        q = [np.exp((self.a[agent] - p[agent]) / self.mu) / denominator for agent in range(len(p))]

        return q
    
    def get_inflation(self):
        sample = np.random.rand()
        
        inflation_t = 0
        if sample < self.rho:
            
            with torch.no_grad():
                inflation_values = np.array(self.inflation_history) # transform to array
                inflation_values = inflation_values[inflation_values != 0][-self.v]
                inflation_values = torch.tensor(inflation_values).reshape(1, -1, 1).float()
                inflation_t = float(self.inflation_model(inflation_values).squeeze())
            
            self.c *= (1 + inflation_t) # adjust marginal cost
            
            #print('Calculating new equilibria...')
            self.pN = self.get_nash() # get nash price
            self.pM = self.get_monopoly() # get monopoly price
            
            self.price_high = self.pM * (1 + self.xi)
            self.price_low = self.pN * (1 - self.xi)
            
            self.inflation_history.append(inflation_t) # store inflation
            
        self.nash_history += [self.pN]
        self.monopoly_history += [self.pM]
            
        return inflation_t