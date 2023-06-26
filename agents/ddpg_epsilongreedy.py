import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Normal, Uniform
from torch.autograd import Variable
import numpy as np
    
class Actor(nn.Module):
    def __init__(self, input_size, output_size, hidden_size = 256, num_layers = 2, dropout = 0.1):
        super().__init__()
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        #self.action_high = action_high
        
        self.embedding = nn.Linear(1, hidden_size)
        # input: N x seq_large (k) x features (N)
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers,
                            batch_first = True, dropout = dropout)
        
        self.fc = nn.Linear(hidden_size * 2, output_size)
        
        #self.action_scale = (action_high - action_low) / 2.0
        #self.bias = (action_high + action_low) / 2.0
        
    def forward(self, state):
        
        # unpack
        inflation, past_prices, past_inflation = state
        
        history = torch.cat([past_prices, past_inflation], dim = 2)
        
        # inflation embedding
        inflation = F.relu(self.embedding(inflation))
        
        # history lstm
        h_0 = Variable(torch.randn(
            self.num_layers, history.size(0), self.hidden_size))
        
        c_0 = Variable(torch.randn(
            self.num_layers, history.size(0), self.hidden_size))
        
        history, hidden = self.lstm(history, (h_0, c_0))
        history = F.relu(history[:, -1, :])
        
        # concatenate
        x = torch.cat([inflation, history], dim = 1)
        
        # output -1 to 1
        x = F.tanh(self.fc(x))
        
        return x
        
        # scale output
        #return x * self.action_scale + self.bias
    
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size = 256, num_layers = 2, dropout = 0.1):
        super().__init__()
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        self.embedding = nn.Linear(2, hidden_size)
        # input: N x seq_large (k) x features (N)
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers,
                            batch_first = True, dropout = dropout)
        
        self.fc = nn.Linear(hidden_size * 2, output_size)
        
    def forward(self, state, action):
        
        # unpack
        inflation, past_prices, past_inflation = state
        
        history = torch.cat([past_prices, past_inflation], dim = 2)
        
        # inflation embedding
        numeric = torch.cat([inflation, action], dim = 1)
        numeric = F.relu(self.embedding(numeric))
        
        # history lstm
        h_0 = Variable(torch.zeros(
            self.num_layers, history.size(0), self.hidden_size))
        
        c_0 = Variable(torch.zeros(
            self.num_layers, history.size(0), self.hidden_size))
        
        history, hidden = self.lstm(history, (h_0, c_0))
        history = F.relu(history[:, -1, :])
        
        # concatenate
        x = torch.cat([numeric, history], dim = 1)
        
        return self.fc(x) 
    
    
class DDPGAgent():
    def __init__(self, N, actor_lr = 1e-3, Q_lr = 1e-2, gamma = 0.99, tau = 0.9, 
                 hidden_size = 256, Q_updates = 1, epsilon = 0.9, epsilon_decay = 0.99):
        
        # actor and critic
        self.actor = Actor(N + 1, 1, hidden_size)
        self.Q = QNetwork(N + 1, 1, hidden_size)
        
        # actor and critic targets
        self.actor_target = Actor(N + 1, 1, hidden_size)
        self.Q_target = QNetwork(N + 1, 1, hidden_size)
        
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.Q_target.load_state_dict(self.Q.state_dict())
        
        # optimizers
        self.actor_optimizer = Adam(self.actor.parameters(), lr = actor_lr)
        self.Q_optimizer = Adam(self.Q.parameters(), lr = Q_lr)
        
        self.gamma = gamma
        self.tau = tau
        self.Q_updates = Q_updates
        
        self.actor_loss = []
        self.Q_loss = []
        
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        
    def select_action(self, state, action_high, action_low, epsilon_greedy = True, action_noise = 0.2):
        inflation, past_prices, past_inflation = state
        inflation = torch.tensor(inflation)
        past_prices = torch.tensor(past_prices).unsqueeze(0)
        past_inflation = torch.tensor(past_inflation).unsqueeze(0)
        
        state = (inflation, past_prices, past_inflation)
        
        with torch.no_grad():
            action = self.actor(state).squeeze(0)
            
            action_scale = (action_high - action_low) / 2.0
            bias = (action_high + action_low) / 2.0
            action = action * action_scale + bias # scale
            
        if epsilon_greedy: # epsilon greedy
            sample = np.random.random_sample()
            if sample > self.epsilon:
                with torch.no_grad():
                    action = self.actor(state).squeeze(0)
                    action_scale = (action_high - action_low) / 2.0
                    bias = (action_high + action_low) / 2.0
                    action = action * action_scale + bias # scale
            else:
                action = Uniform(action_low, action_high).sample()
                self.epsilon *= self.epsilon_decay
        else: # gaussian noise
            with torch.no_grad():
                action = self.actor(state).squeeze(0)
                action_scale = (action_high - action_low) / 2.0
                bias = (action_high + action_low) / 2.0
                action = action * action_scale + bias # scale
            noise = Normal(0, action_noise * action_high).sample() if action_noise > 0 else torch.zeros(1)
            action += noise # add noise
        
        action = torch.clamp(action, action_low, action_high) # clamp

        return action.item()

    
    def update(self, state, action, reward, state_t1, done):
        
        inflation = torch.tensor(np.array([s[0] for s in state])).squeeze(2)
        past_prices = torch.tensor(np.array([s[1] for s in state]))
        past_inflation = torch.tensor(np.array([s[2] for s in state]))
        
        inflation_t1 = torch.tensor(np.array([s[0] for s in state_t1])).squeeze(2)
        past_prices_t1 = torch.tensor(np.array([s[1] for s in state_t1]))
        past_inflation_t1 = torch.tensor(np.array([s[2] for s in state_t1]))    
        
        state = (inflation, past_prices, past_inflation)
        action = torch.tensor(action).unsqueeze(1)
        reward = torch.tensor(reward).unsqueeze(1)
        state_t1 = (inflation_t1, past_prices_t1, past_inflation_t1)
        done = torch.tensor(done).unsqueeze(dim = 1)
        
        for _ in range(self.Q_updates):
            self.update_Q(state, action, reward, state_t1, done)
        
        self.update_actor(state)
        
        # update the target networks
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def update_actor(self, state):
        
        action = self.actor(state) # add clamp?
        actor_loss = -self.Q(state, action).mean() 
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        self.actor_loss.append(actor_loss.item())
    
    def update_Q(self, state, action, reward, state_t1, done):
        
        next_Q = reward + self.Q(state, action) * (1 - done) * self.gamma
        
        with torch.no_grad():
            action_t1 = self.actor_target(state_t1) # add clamp?
            Q_t1 = self.Q_target(state_t1, action_t1)
        
        next_Q = next_Q.float()
        Q_loss = F.mse_loss(next_Q, Q_t1)
        
        self.Q_optimizer.zero_grad()
        Q_loss.backward()
        self.Q_optimizer.step()
        
        self.Q_loss.append(Q_loss.item())