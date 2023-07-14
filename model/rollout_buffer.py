import torch
import numpy as np



class RolloutBuffer:
    """Save and store Agent's data"""
    def __init__(self,config,state_size,action_size) -> None:

        self.max_eps_length     = config["max_eps_length"]
        self.num_game_per_batch = config["num_game_per_batch"]
        self.memory_length      = config["memory_length"]
        self.num_blocks         = config["num_blocks"]
        self.embed_dim          = config["embed_dim"]
        self.state_size         = state_size
        self.action_size        = action_size
        self.n_mini_batches     = config["n_mini_batches"]
        
        self.game_count         = 0 #track current game
        self.step_count         = 0 #track current time step in game
        # Generate episodic memory mask used in attention
        self.memory_mask = torch.tril(torch.ones((self.memory_length, self.memory_length)), diagonal=-1)
        """ e.g. memory mask tensor looks like this if memory_length = 6
        0, 0, 0, 0, 0, 0
        1, 0, 0, 0, 0, 0
        1, 1, 0, 0, 0, 0
        1, 1, 1, 0, 0, 0
        1, 1, 1, 1, 0, 0
        1, 1, 1, 1, 1, 0
        """         
        # Setup memory window indices to support a sliding window over the episodic memory
        repetitions = torch.repeat_interleave(torch.arange(0, self.memory_length).unsqueeze(0), self.memory_length - 1, dim = 0).long()
        self.memory_indices = torch.stack([torch.arange(i, i + self.memory_length) for i in range(self.max_eps_length - self.memory_length + 1)]).long()
        self.memory_indices = torch.cat((repetitions, self.memory_indices))
        """ e.g. the memory window indices tensor looks like this if memory_length = 4 and max_episode_length = 7:
        0, 1, 2, 3
        0, 1, 2, 3
        0, 1, 2, 3
        0, 1, 2, 3
        1, 2, 3, 4
        2, 3, 4, 5
        3, 4, 5, 6
        """
        self.memory_mask_batch    = torch.zeros((self.num_game_per_batch,self.max_eps_length,self.memory_length))
        self.memory_indices_batch = torch.zeros((self.num_game_per_batch,self.max_eps_length,self.memory_length)).int()
        for i in range(self.max_eps_length):
            self.memory_mask_batch[:,i]    = self.memory_mask[torch.clip(torch.tensor([[i]]), 0, self.memory_length - 1)]
            self.memory_indices_batch[:,i] = self.memory_indices[i].int()

        self.memory_index_batch = torch.arange(self.num_game_per_batch).unsqueeze(1).expand((self.num_game_per_batch,self.max_eps_length))

        self.batch = {
            "states"        : torch.zeros((self.num_game_per_batch,self.max_eps_length,self.state_size)),
            "actions"       : torch.zeros((self.num_game_per_batch,self.max_eps_length)),
            "values"        : torch.zeros((self.num_game_per_batch,self.max_eps_length)),
            "probs"         : torch.zeros((self.num_game_per_batch,self.max_eps_length)),
            "policy"        : torch.zeros((self.num_game_per_batch,self.max_eps_length,self.action_size)),
            "dones"         : torch.zeros((self.num_game_per_batch,self.max_eps_length)),
            "action_mask"   : torch.zeros((self.num_game_per_batch,self.max_eps_length,self.action_size)),
            "rewards"       : torch.zeros((self.num_game_per_batch,self.max_eps_length)),
            "padding"       : torch.zeros((self.num_game_per_batch,self.max_eps_length)),
            "advantages"    : torch.zeros((self.num_game_per_batch,self.max_eps_length)),
            "memory"        : torch.zeros((self.num_game_per_batch,self.max_eps_length, self.num_blocks, self.embed_dim)),
        }





    def add_data(self,state,action,value,reward,done,valid_action,prob,memory,policy):
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
            self.batch["memory"][self.game_count][self.step_count]      = memory
            self.batch["policy"][self.game_count][self.step_count]      = policy
        except:
            pass

    def reset_data(self):
        """Clear all data"""
        self.batch = {
            "states"        : torch.zeros((self.num_game_per_batch,self.max_eps_length,self.state_size)),
            "actions"       : torch.zeros((self.num_game_per_batch,self.max_eps_length)),
            "values"        : torch.zeros((self.num_game_per_batch,self.max_eps_length)),
            "probs"         : torch.zeros((self.num_game_per_batch,self.max_eps_length)),
            "policy"        : torch.zeros((self.num_game_per_batch,self.max_eps_length,self.action_size)),
            "dones"         : torch.zeros((self.num_game_per_batch,self.max_eps_length)),
            "action_mask"   : torch.zeros((self.num_game_per_batch,self.max_eps_length,self.action_size)),
            "rewards"       : torch.zeros((self.num_game_per_batch,self.max_eps_length)),
            "padding"       : torch.zeros((self.num_game_per_batch,self.max_eps_length)),
            "advantages"    : torch.zeros((self.num_game_per_batch,self.max_eps_length)),
            "memory"        : torch.zeros((self.num_game_per_batch,self.max_eps_length, self.num_blocks, self.embed_dim)),
        }

        self.game_count = 0
        self.step_count = 0
        


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

    def prepare_batch(self) -> None:
        """Flattens the training samples and stores them inside a dictionary. Due to using a recurrent policy,
        the data is split into episodes or sequences beforehand.
        """
        # Supply training samples
        samples = {
            "actions":        self.batch["actions"],
            "values":         self.batch["values"],
            "probs":          self.batch["probs"],
            "policy":         self.batch["policy"],
            "action_mask":    self.batch["action_mask"],
            "advantages":     self.batch["advantages"],
            "states":         self.batch["states"], 
            "memory_mask":    self.memory_mask_batch,
            "memory_indices": self.memory_indices_batch,
            "memory_index":   self.memory_index_batch
        }

        # Flatten all samples and convert them to a tensor except memories and its memory mask
        self.samples = {}
        padding = self.batch["padding"].reshape(-1)
        for key, value in samples.items():
            self.samples[key] = value.reshape(value.shape[0] * value.shape[1], *value.shape[2:])[padding!=0]
        self.batch_size = self.samples["actions"].shape[0]
    def mini_batch_generator(self):
        """
        Overview:
            A generator that returns a dictionary containing the data of a whole minibatch.
            This mini batch is completely shuffled.
            
        Yields:
            {dict} -- Mini batch data for training
        """
        # Prepare indices (shuffle)
        indices = torch.randperm(self.batch_size)
        mini_batch_size = self.batch_size // self.n_mini_batches
        for start in range(0, self.batch_size, mini_batch_size):
            # Compose mini batches
            end = start + mini_batch_size
            if end < self.batch_size:
                mini_batch_indices = indices[start: end]
                mini_batch = {}
                for key, value in self.samples.items():
                    if key == "memory_index":
                        # Add the correct episode memories to the concerned mini batch
                        mini_batch["memory"] = self.batch["memory"][value[mini_batch_indices]]
                    else:
                        mini_batch[key] = value[mini_batch_indices]
                yield mini_batch

    @staticmethod
    def batched_index_select(input, dim, index):
        """
        Selects values from the input tensor at the given indices along the given dimension.
        This function is similar to torch.index_select, but it supports batched indices.
        The input tensor is expected to be of shape (batch_size, ...), where ... means any number of additional dimensions.
        The indices tensor is expected to be of shape (batch_size, num_indices), where num_indices is the number of indices to select for each element in the batch.
        The output tensor is of shape (batch_size, num_indices, ...), where ... means any number of additional dimensions that were present in the input tensor.

        Arguments:
            input {torch.tensor} -- Input tensor
            dim {int} -- Dimension along which to select values
            index {torch.tensor} -- Tensor containing the indices to select

        Returns:
            {torch.tensor} -- Output tensor
        """
        for ii in range(1, len(input.shape)):
            if ii != dim:
                index = index.unsqueeze(ii)
        expanse = list(input.shape)
        expanse[0] = -1
        expanse[dim] = -1
        index = index.expand(expanse)
        index = index.to(torch.int64)
        return torch.gather(input, dim, index)

                

