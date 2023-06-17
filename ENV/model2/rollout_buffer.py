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
        """
        Overview:
            Calculate GAE.

        Arguments:
            - gamma: (`float`): gamma discount.
            - gae_lambda: (`float`): gae_lambda discount.
        """

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

    def mini_batch_loader(self,config):
        """
        Overview:
            Mini batch data generator.

        Arguments:
            - mini_batch_size: (`int`): size of mini batch.
            
        Yield:
            - mini-batch: (`dict`): dictionary contain the training data.
        
        """
        idx = torch.randint(len(self.data["action"]) - config["memory_length"],size=(config["batch_size"],))
        while True:
            mini_batch          = {}
            for key,value in self.batch.items():
                mini_batch[key] = torch.stack([value[i:i + config["memory_length"]] for i in idx])
                yield mini_batch
                print(key,mini_batch[key].shape)

    def to_tensor(self):
        """Turn data into tensor"""
        for key,value in self.batch.items():
            if not torch.is_tensor(value):
                try:
                    self.batch[key] = torch.tensor(np.array(value),dtype=torch.float32)
                except:
                    print(key,value)
