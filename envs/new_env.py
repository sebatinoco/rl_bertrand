import numpy as np
import gymnasium as gym
from scipy.optimize import minimize, fsolve

import optuna
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.WARNING)

class BertrandEnv():
    def __init__(self, N, k, rho, mu = 1, a = None, a_0 = 0, a_index = 1, c = 1):
        
        self.N = N # number of agents
        self.k = k # past periods to observe
        self.rho = rho # probability of changing prices
        self.a_0 = a_0 # base vertical differentiation index
        self.mu = mu # horizontal differentiation index
        self.a_index = a_index # vertical differentiation indexes
        self.c = c # marginal cost

        # vertical diff
        self.a = np.array(a)
        if a is None:
            self.a = np.array([c + a_index] * N)
        assert len(self.a) == N, 'self.a must be of equal size as N'
        
        self.inflation_history = [] # inflation history
        self.prices_history = [] # prices history
        
        self.prices_space = gym.spaces.Box(low = self.pN, high = self.pM, shape = (k, N), dtype = float) # prices space
        self.inflation_space = gym.spaces.Box(low = 1.5, high = 3.5, shape = (k,), dtype = float) # inflation space
        
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
        
        assert all(round(price, 4) == round(nash_solution[0], 4) for price in nash_solution), f"Nash price should be unique: {nash_solution}" # all prices are the same
        #assert nash(nash_solution) == [0.0] * self.N, f'Nash price should be a root: {nash(nash_solution)}' # nash price is a root
        
        pN = nash_solution[0] # float
        print(f'Nash Price: {pN:.2f}')
        
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
        
        print(f'Monopoly Price: {pM:.2f}')

        return pM
    
    def step(self, action):
        
        '''
        Computes a step over the environment. Receives an action (array of prices) and return a tuple of (observation, reward, done, _)
        action: array of prices (np.array)
        '''
        
        # compute quantities
        reward = self.demand(p = action)
        
        # update price history
        self.prices_history.append(action)
        
        # gather observation
        inflation = self.get_inflation()
        past_prices = self.prices_history[-self.k:]
        past_inflation = self.inflation_history[-self.k:]
        ob_t1 = (inflation, past_prices, past_inflation)
         
        done = False
        info = None
        
        # update history
        self.inflation_history.append(inflation)
        
        return ob_t1, reward, done, info
    
    def reset(self):
        
        '''
        Resets the environment.
        '''
        
        self.pN = self.get_nash() # get nash price
        self.pM = self.get_monopoly() # get monopoly price
        
        inflation = self.get_inflation()
        past_prices = self.prices_space.sample()
        past_inflation = self.inflation_space.sample()
        
        ob_t = (inflation, past_prices, past_inflation)
        
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
            inflation_t = 0.03 # inflation_model(INFLATION NOT NULL)
            
            self.c *= (1 + inflation_t) # adjust marginal cost
            
            print('Calculating new equilibria...')
            self.pN = self.get_nash() # get nash price
            self.pM = self.get_monopoly() # get monopoly price
            
            self.inflation_history.append(inflation_t) # store inflation
            
        return inflation_t