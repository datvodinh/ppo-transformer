import torch
import numpy as np



class RolloutBuffer:
    """Save and store Agent's data"""
    def __init__(self,max_eps_length,num_game_per_batch,state_size,action_size) -> None:
        self.batch = {
            "states"     : torch.zeros((num_game_per_batch,max_eps_length,state_size)),
            "actions"    : torch.zeros((num_game_per_batch,max_eps_length)),
            "values"     : torch.zeros((num_game_per_batch,max_eps_length)),
            "probs"      : torch.zeros((num_game_per_batch,max_eps_length)),
            "dones"      : torch.zeros((num_game_per_batch,max_eps_length)),
            "action_mask": torch.zeros((num_game_per_batch,max_eps_length,action_size)),
            "rewards"    : torch.zeros((num_game_per_batch,max_eps_length)),
            "padding"    : torch.zeros((num_game_per_batch,max_eps_length)),
            "advantages" : torch.zeros((num_game_per_batch,max_eps_length)),
        }

        self.max_eps_length     = max_eps_length
        self.num_game_per_batch = num_game_per_batch
        self.state_size         = state_size
        self.action_size        = action_size

        self.game_count         = 0 #track current game
        self.step_count         = 0 #track current time step in game

        



    def add_data(self,state,action,value,reward,done,valid_action,prob):
        """Add data to rollout buffer"""
        try:
            self.batch["states"][self.game_count][self.step_count]      = state
            self.batch["actions"][self.game_count][self.step_count]     = action
            self.batch["values"][self.game_count][self.step_count]      = value
            self.batch["probs"][self.game_count][self.step_count]       = prob
            self.batch["dones"][self.game_count][self.step_count]       = done
            self.batch["action_mask"][self.game_count][self.step_count] = valid_action
            self.batch["rewards"][self.game_count][self.step_count]     = reward
            self.batch["padding"][self.game_count][self.step_count]     = 1
        except:
            pass

    def reset_data(self):
        """Clear all data"""
        self.batch = {
            "states"     : torch.zeros((self.num_game_per_batch,self.max_eps_length,self.state_size)),
            "actions"    : torch.zeros((self.num_game_per_batch,self.max_eps_length)),
            "values"     : torch.zeros((self.num_game_per_batch,self.max_eps_length)),
            "probs"      : torch.zeros((self.num_game_per_batch,self.max_eps_length)),
            "dones"      : torch.zeros((self.num_game_per_batch,self.max_eps_length)),
            "action_mask": torch.zeros((self.num_game_per_batch,self.max_eps_length,self.action_size)),
            "rewards"    : torch.zeros((self.num_game_per_batch,self.max_eps_length)),
            "padding"    : torch.zeros((self.num_game_per_batch,self.max_eps_length)),
            "advantages" : torch.zeros((self.num_game_per_batch,self.max_eps_length)),
        }
        self.game_count = 0 #track current game
        self.step_count = 0 #track current time step in game


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

        last_value               = self.batch["values"][:,-1]
        last_advantage           = torch.zeros_like(last_value)

        for t in range(self.max_eps_length-1,-1,-1):
            mask                          = 1.0 - self.batch["dones"][:,t]
            last_value                    = last_value * mask
            last_advantage                = last_advantage * mask
            delta                         = self.batch["rewards"][:,t] + gamma * last_value - self.batch["values"][:,t]
            last_advantage                = delta + gamma * gae_lambda * last_advantage
            self.batch["advantages"][:,t] = last_advantage
            last_value                    = self.batch["values"][:,t]

    def mini_batch_loader(self,config):
        """
        Overview:
            Mini batch data generator.

        Arguments:
            - config: (`dict`): config.
            
        Yield:
            - mini-batch: (`dict`): dictionary contain the training data.
        
        """
        idx        = torch.randperm(self.num_game_per_batch)
        mini_batch_size = self.num_game_per_batch // config["n_mini_batch"]
        for start in range(0,self.num_game_per_batch,mini_batch_size):
            end = start + mini_batch_size
            mini_batch_indices = idx[start: end]
            mini_batch = {}

            for key, value in self.batch.items():
                mini_batch[key] = value[mini_batch_indices]
                
            yield mini_batch

                

