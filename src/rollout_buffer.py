import torch

class RolloutBuffer:
    """Save and store Agent's data"""
    def __init__(self) -> None:
        self.batch = {
            "states"     : [],
            "actions"    : [],
            "values"     : [],
            "probs"      : [],
            "dones"      : [],
            "action_mask": [],
            "rewards"    : [],
        }

    def add_data(self,state,action,value,reward,done,valid_action):
        """Add data to rollout buffer"""
        self.batch["states"].append(state)
        self.batch["actions"].append(action)
        self.batch["values"].append(value)
        self.batch["rewards"].append(reward)
        self.batch["dones"].append(done)
        self.batch["action_mask"].append(valid_action)

    def del_data(self):
        """Clear all data"""
        self.batch = {
            "states"     : [],
            "actions"    : [],
            "values"     : [],
            "probs"      : [],
            "dones"      : [],
            "action_mask": [],
            "rewards"    : [],
        }


    def cal_advantages(self, last_value:torch.tensor, gamma:float, lamda:float) -> None:
        """Calculate the Advantages"""
        with torch.no_grad():
            last_advantage = 0
            step = len(self.batch["rewards"])
            for t in reversed(range(step)):
                last_value = last_value * (1 - self.batch["dones"][t])
                last_advantage = last_advantage * (1 - self.batch["dones"][t])
                delta = self.batch["rewards"][t] + gamma * last_value - self.batch["values"][t]
                last_advantage = delta + gamma * lamda * last_advantage
                self.batch["advantages"][t] = last_advantage
                last_value = self.batch["values"][t]

    def mini_batch_loader(self,mini_batch_size):
        """Return the dictionary of the mini batch data."""
        # Prepare indices (shuffle)
        batch_size = len(self.batch["actions"])
        indices = torch.randperm(batch_size)
        for start in range(0, batch_size, mini_batch_size):
            # Compose mini batches
            end = start + mini_batch_size
            mini_batch_indices = indices[start: end]
            mini_batch = {}
            for key, value in self.batch.items():
                mini_batch[key] = value[mini_batch_indices].to(self.device)
            yield mini_batch