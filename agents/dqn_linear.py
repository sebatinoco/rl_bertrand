import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam
import numpy as np

class DQN(nn.Module):
    def __init__(self, dim_states, dim_actions, hidden_size = 256):
        super().__init__()
        
        self.fc1 = nn.Linear(dim_states, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, dim_actions)
        
    def forward(self, x):
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

class DQNAgent():
    def __init__(self, dim_states, dim_actions, action_low, action_high, lr = 1e-2, gamma = 0.99, target_steps = 200, hidden_size = 256, epsilon = 0.9, epsilon_decay = 0.99):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.dim_actions = dim_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.target_steps = target_steps
        self.target_count = 0
        
        # initiate network
        self.network = DQN(dim_states, dim_actions, hidden_size)
        self.target_network = DQN(dim_states, dim_actions, hidden_size)
        
        self.target_network.load_state_dict(self.network.state_dict()) # load target network params
        
        self.optimizer = Adam(self.network.parameters(), lr = lr) # optimizer
        
        self.action_low = action_low
        self.action_high = action_high
        
        self.action2price = np.linspace(self.action_low, self.action_high, dim_actions)
        
    def select_action(self, state):
        
        inflation, past_prices, past_inflation = state
 
        inflation = torch.FloatTensor(inflation).flatten()
        past_prices = torch.FloatTensor(past_prices).flatten()
        past_inflation = torch.FloatTensor(past_inflation).flatten()
        
        state = torch.cat([inflation, past_prices, past_inflation]).unsqueeze(0)

        if np.random.random() > self.epsilon:
            with torch.no_grad():
                action = torch.argmax(self.network(state), dim = 1).item()
        else:
                action = np.random.randint(0, self.dim_actions)
                self.epsilon *= self.epsilon_decay
                
        return self.rescale_action(action)
    
    def rescale_action(self, action):
        
        scaled_action = self.action2price[action]
        scaled_action = np.exp(scaled_action)
        
        return scaled_action
    
    def update(self, states, actions, rewards, next_states, dones):
        
        inflation = torch.FloatTensor(np.array([s[0] for s in states])).flatten(start_dim = 1)
        past_prices = torch.FloatTensor(np.array([s[1] for s in states])).flatten(start_dim = 1)
        past_inflation = torch.FloatTensor(np.array([s[2] for s in states])).flatten(start_dim = 1)

        next_inflation = torch.FloatTensor(np.array([s[0] for s in next_states])).flatten(start_dim = 1)
        next_past_prices = torch.FloatTensor(np.array([s[1] for s in next_states])).flatten(start_dim = 1)
        next_past_inflation = torch.FloatTensor(np.array([s[2] for s in next_states])).flatten(start_dim = 1)

        states = torch.cat([inflation, past_prices, past_inflation], dim = 1)
        next_states = torch.cat([next_inflation, next_past_prices, next_past_inflation], dim = 1)
        
        # to torch
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        dones = dones.view(dones.size(0), -1)
        
        self.target_count += 1
        self.update_network(states, actions, rewards, next_states, dones)
        if (self.target_count % self.target_steps) == 0:
            for param, target_param in zip(self.network.parameters(), self.target_network.parameters()):
                target_param.data.copy_(param.data)
        
    def update_network(self, states, actions, rewards, next_states, dones):
        
        with torch.no_grad():
            target_max = torch.max(self.target_network(next_states), dim = 1).values # max of Q values on t1
            td_target = rewards.squeeze() + self.gamma * target_max * (1 - dones.squeeze()) # fix the target
        
        old_val = self.network(states).gather(1, actions.long()).squeeze() # prediction of network
        
        Q_loss = F.mse_loss(td_target, old_val)
        
        self.optimizer.zero_grad()
        Q_loss.backward()
        self.optimizer.step()
        
    def update_scale(self, action_low, action_high):
        
        self.action_low = action_low
        self.action_high = action_high
        
        self.action2price = np.linspace(self.action_low, self.action_high, self.dim_actions)