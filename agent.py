class QLearning():
  def __init__(self, env):

    self.action_space = env.action_space.n
    self.Qtable = []

    self._obs_to_state = {}

  def update(self, observation):
    if observation not in self._obs_to_state.keys():
      self._obs_to_state.update({observation: len(self._obs_to_state)})
      self.Qtable.append([0] * self.action_space)