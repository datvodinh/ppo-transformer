import torch
import torch.nn.functional as F
import numpy as np
from src.rollout_buffer import RolloutBuffer
class Agent(RolloutBuffer):
    """Agent"""
    def __init__(self,env,model):
        self.env = env
        self.model = model

    def play(self,state,per):
        with torch.no_grad():
            policy,value,_ = self.model(torch.tensor(state,dtype=torch.float32))
            policy = policy.squeeze().numpy()
            list_action = self.env.getValidActions(state)
            actions = np.where(list_action==1)[0]
            action = np.random.choice(actions,p = self.stable_softmax(policy[actions]))

            if self.env.getReward(state)==-1:
                self.add_data(state=state,
                              action=action,
                              value=value,
                              reward=0.0,
                              done=0,
                              valid_action=list_action
                              )
            else:
                self.add_data(state=state,
                              action=action,
                              value=value,
                              reward=self.env.getReward(state),
                              done=1,
                              valid_action=list_action
                              )
        
        return action,per 
    
    @staticmethod
    def stable_softmax(x):
        # Input validation
        if len(x) == 0:
            raise ValueError("Input array must not be empty.")
        max_val = np.max(x)
        scaled_values = x - max_val
        exp_scaled_values = np.exp(scaled_values)
        softmax = exp_scaled_values / np.sum(exp_scaled_values)
        return softmax
    
    def run(self,num_games)->float:
        """Run custom environment and return win rate"""
        return self.env.run(self.play,num_games,np.array([0.]),1)[0] / num_games

    def to_tensor(self):
        """Turn data into tensor"""
        for key,value in self.batch.items():
            self.batch[key] = torch.tensor(value,dtype=torch.float32)

    
    
    

    
    