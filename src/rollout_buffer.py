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
        self.memories = []
    def add_data(self,state,action,value,reward,done,valid_action):
        """Add data to rollout buffer"""
        self.batch["states"].append(state)
        self.batch["actions"].append(action)
        self.batch["values"].append(value)
        self.batch["rewards"].append(reward)
        self.batch["dones"].append(done)
        self.batch["action_mask"].append(valid_action)

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


    def cal_advantages(self,gamma,gae_lambda):

        self.batch["advantages"] = torch.zeros_like(self.batch["rewards"])
        last_advantage           = 0
        last_value               = self.batch["values"][-1]

        for t in range(len(self.batch["rewards"])-1,-1,-1):
            mask                        = 1.0 - self.batch["done"][t]
            last_value                  = last_value * mask
            last_advantage              = last_advantage * mask
            delta                       = self.batch["rewards"][t] + gamma * last_value - self.batch["values"][t]
            last_advantage              = delta + gamma * gae_lambda * last_advantage
            self.batch["advantages"][t] = last_advantage
            last_value                  = self.batch["values"][t]

        return self.batch["advantages"]

    def mini_batch_loader(self,mini_batch_size):
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
                if key == "memory_indices":
                    mini_batch[key] = 0#####NOTE
                mini_batch[key] = value[mini_batch_indices]
            yield mini_batch

    def to_tensor(self):
        """Turn data into tensor"""
        for key,value in self.batch.items():
            self.batch[key] = torch.tensor(value,dtype=torch.float32)

    def memory_index_select(memory, dim, index):
        """
        Selects values from the input tensor at the given indices along the given dimension.
        """
        for ii in range(1, len(input.shape)):
            if ii != dim:
                index = index.unsqueeze(ii)
        expanse      = list(input.shape)
        expanse[0]   = -1
        expanse[dim] = -1
        index        = index.expand(expanse)
        return torch.gather(input, dim, index) 

