import torch
import torch.nn.functional as F
import numpy as np
from src.rollout_buffer import RolloutBuffer
class Agent(RolloutBuffer):
    """Agent"""
    def __init__(self,env,model,config):
        self.env                = env
        self.model              = model
        self.max_episode_length = config["max_episode_length"]
        self.memory_length      = config["memory_length"]
        self.num_blocks         = config["num_blocks"]
        self.embed_dim          = config["embed_dim"]
        self.step_current_game  = 0
        #Inspire from https://github.com/MarcoMeter/episodic-transformer-memory-ppo/

        self.memory             = torch.zeros((self.max_episode_length, self.num_blocks, self.embed_dim), dtype=torch.float32)   
        # Generate episodic memory mask used in attention
        self.memory_mask        = torch.tril(torch.ones((self.memory_length, self.memory_length)), diagonal=-1)       
        # Setup memory window indices to support a sliding window over the episodic memory
        repetitions             = torch.repeat_interleave(torch.arange(0, self.memory_length).unsqueeze(0), self.memory_length - 1, dim = 0).long()
        self.memory_indices     = torch.stack([torch.arange(i, i + self.memory_length) for i in range(self.max_episode_length - self.memory_length + 1)]).long()
        self.memory_indices     = torch.cat((repetitions, self.memory_indices))

    def play(self,state,per):
        with torch.no_grad():
            policy,value,memory = self.model(state          = torch.tensor(state,dtype=torch.float32),
                                             memories       = self.memories,
                                             mask           = self.memory_mask[torch.clip(self.step_current_game, 0, self.memory_length - 1)],
                                             memory_indices = self.memory_indices[self.step_current_game])
            
            policy         = policy.squeeze().numpy()
            list_action    = self.env.getValidActions(state)
            actions        = np.where(list_action==1)[0]
            action         = np.random.choice(actions,p = self.stable_softmax(policy[actions]))

            self.memory[self.step_current_game] = memory 

            if self.env.getReward(state)==-1:
                self.step_current_game += 1
                self.add_data(state        = state,
                              action       = action,
                              value        = value,
                              reward       = 0.0,
                              done         = 0,
                              valid_action = list_action
                              )
            else:
                self.step_current_game = 0 #reset step
                self.memory            = torch.zeros((self.max_episode_length, self.num_blocks, self.embed_dim), dtype=torch.float32) #reset memory
                self.add_data(state        = state,
                              action       = action,
                              value        = value,
                              reward       = self.env.getReward(state) * 1.0,
                              done         = 1,
                              valid_action = list_action
                              )
        
        return action,per 
    
    @staticmethod
    def stable_softmax(x):
        max_val           = np.max(x)
        scaled_values     = x - max_val
        exp_scaled_values = np.exp(scaled_values)
        softmax           = exp_scaled_values / np.sum(exp_scaled_values)
        return softmax
    
    def run(self,num_games)->float:
        """Run custom environment and return win rate"""
        return self.env.run(self.play,num_games,np.array([0.]),1)[0] / num_games
