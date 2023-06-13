import torch

class RolloutBuffer:
    """Save and store Agent's data"""
    def __init__(self) -> None:
        self.states      = []
        self.actions     = []
        self.values      = []
        self.probs       = []
        self.dones       = []
        self.action_mask = []

    def add_data(self,state,action,value,prob,done,valid_action):
        """Add data to rollout buffer"""
        self.states.append(state)
        self.actions.append(action)
        self.values.append(value)
        self.probs.append(prob)
        self.dones.append(done)
        self.action_mask.append(valid_action)

    def del_data(self):
        """Clear all data"""
        self.states      = []
        self.actions     = []
        self.values      = []
        self.probs       = []
        self.dones       = []
        self.action_mask = []


    def calc_advantages(self, last_value:torch.tensor, gamma:float, lamda:float) -> None:
        """Calculate the advantage"""
        with torch.no_grad():
            last_advantage = 0
            rewards = torch.tensor(self.rewards)
            for t in reversed(range(self.worker_steps)):
                last_value = last_value * (1 - self.dones[:, t])
                last_advantage = last_advantage * (1 - self.dones[:, t])
                delta = rewards[:, t] + gamma * last_value - self.values[:, t]
                last_advantage = delta + gamma * lamda * last_advantage
                self.advantages[:, t] = last_advantage
                last_value = self.values[:, t]