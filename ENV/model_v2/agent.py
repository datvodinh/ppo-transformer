import torch
import torch.nn.functional as F
import numpy as np
from model_v2.rollout_buffer import RolloutBuffer
from model_v2.memory import Memory
from torch.distributions import Categorical
class Agent(RolloutBuffer):
    """Agent"""
    def __init__(self,env,model,config):
        super().__init__()
        self.env                = env
        self.model              = model
        self.max_episode_length = config["max_episode_length"]
        self.memory_length      = config["memory_length"]
        self.num_blocks         = config["num_blocks"]
        self.embed_dim          = config["embed_dim"]
        self.step_current_game  = 0
        # self.memory             = Memory(memory_len=64,batch_size=1,embedding_dim=self.embed_dim,num_blocks=self.num_blocks)
        self.model.transformer.reset_memory(batch_size=1)
    def play(self,state,per):
        """
        Overview:
            Agent's play function

        Arguments:
            - state: (`np.array`): state
            - per: (`List`): per file

        Returns:
            - action: (`int`): Agent's action
            - per: (`List`): per file
        """
        with torch.no_grad():
            tensor_state = torch.tensor(state.reshape(1,1,-1),dtype=torch.float32)
            policy,value = self.model(tensor_state)
            policy       = policy.squeeze()
            list_action  = self.env.getValidActions(state)
            actions  = torch.tensor(list_action,dtype=torch.float32)
            categorical  = Categorical(logits=policy.masked_fill(actions==0,float('-inf')))
            action       = categorical.sample().item()
            if actions[action] != 1:
                action   = np.random.choice(np.where(list_action==1)[0])

            log_prob     = categorical.log_prob(torch.tensor([action]).view(1,-1)).squeeze()
            

            # print(action)

            if self.env.getReward(state)==-1:
                self.step_current_game += 1
                self.add_data(state        = state,
                              action       = action,
                              value        = value.item(),
                              reward       = 0.0,
                              done         = 0,
                              valid_action = list_action,
                              prob         = log_prob
                              )
            else:
                self.step_current_game = 0 #reset step
                self.model.transformer.reset_memory(batch_size=1)
                self.add_data(state        = state,
                              action       = action,
                              value        = value.item(),
                              reward       = self.env.getReward(state) * 1.0,
                              done         = 1,
                              valid_action = list_action,
                              prob         = log_prob
                              )
        
        return action,per 
    
    @staticmethod
    def stable_softmax(x):
        """
        Overview:
            Return the stable softmax

        Arguments:
            - x: (`Optional[torch.Tensor]`): input Logits.

        Returns:
            - softmax: (`Optional[torch.Tensor]`): stable softmax.
        """
        max_val           = np.max(x)
        scaled_values     = x - max_val
        exp_scaled_values = np.exp(scaled_values)
        softmax           = exp_scaled_values / np.sum(exp_scaled_values)
        return softmax
    
    def run(self,num_games:int)->float:
        """
        Overview:
            Run custom environment and return win rate.

        Arguments:
            - num_games: (`int`): number of games.
            
        """
        
        win_rate =  self.env.run(self.play,num_games,np.array([0.]),1)[0] / num_games
        # print(num_games,win_rate)
        return win_rate