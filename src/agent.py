import torch
import torch.nn.functional as F
import numpy as np

class Agent:
    """Agent"""
    def __init__(self,env,model):
        self.env = env
        self.model = model

    def play(self,state,per):
        with torch.no_grad():
            policy = self.model.getPolicy(torch.tensor(state,dtype=torch.float32))
            policy = policy.squeeze().numpy()
            actions = np.where(self.env.getValidActions(state)==1)[0]
            action = np.random.choice(actions,p = self.stableSoftmax(policy[actions]))
        
        return action,per 
    
    @staticmethod
    def stableSoftmax(x):
        # Input validation
        if len(x) == 0:
            raise ValueError("Input array must not be empty.")
        max_val = np.max(x)
        scaled_values = x - max_val
        exp_scaled_values = np.exp(scaled_values)
        softmax = exp_scaled_values / np.sum(exp_scaled_values)
        return softmax
    
    def run(self,num_games):
        self.env.run(self.play,num_games,np.array([0.]),1)

    
    
    

    
    