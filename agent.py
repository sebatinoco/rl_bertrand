class QLearning():
  def __init__(self, env):

    self.action_space = env.action_space.n # Action space
    self.Qtable = [] # Q-table

    self._obs_to_state = {} # obs to state

  def update(self, observation):
    
    '''
    Updates the _obs_to_state dictionary with the unseen observation.
    '''
    
    if observation not in self._obs_to_state.keys():
      self._obs_to_state.update({observation: len(self._obs_to_state)})
      self.Qtable.append([0] * self.action_space)