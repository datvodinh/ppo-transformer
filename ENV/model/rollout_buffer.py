import torch
import numpy as np

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
        self.batch["dones"].append(done)
        self.batch["action_mask"].append(valid_action)
        self.batch["rewards"].append(reward)

    def reset_data(self):
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


    def cal_advantages(self,
                       gamma:float,
                       gae_lambda:float):

        self.batch["advantages"] = torch.zeros_like(self.batch["rewards"])
        last_advantage           = 0
        last_value               = self.batch["values"][-1]

        for t in range(len(self.batch["rewards"])-1,-1,-1):
            mask                        = 1.0 - self.batch["dones"][t]
            last_value                  = last_value * mask
            last_advantage              = last_advantage * mask
            delta                       = self.batch["rewards"][t] + gamma * last_value - self.batch["values"][t]
            last_advantage              = delta + gamma * gae_lambda * last_advantage
            self.batch["advantages"][t] = last_advantage
            last_value                  = self.batch["values"][t]

        return self.batch["advantages"]

    def mini_batch_loader(self,mini_batch_size:int):
        """Return the dictionary of the mini batch data."""
        # Prepare indices (shuffle)
        batch_size = len(self.batch["actions"])
        indices    = torch.randperm(batch_size)
        for start in range(0, batch_size, mini_batch_size):
            # Compose mini batches
            end                 = start + mini_batch_size
            mini_batch_indices  = indices[start: end]
            mini_batch          = {}
            for key, value in self.batch.items():
                try:
                    mini_batch[key] = value[mini_batch_indices]
                except:
                    print(key,value)
            yield mini_batch

    def to_tensor(self):
        """Turn data into tensor"""
        for key,value in self.batch.items():
            if not torch.is_tensor(value):
                try:
                    self.batch[key] = torch.tensor(np.array(value),dtype=torch.float32)
                except:
                    print(key,value)