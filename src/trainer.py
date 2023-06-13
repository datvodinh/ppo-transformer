from torch.distributions import Categorical,kl_divergence
import torch
import numpy as np
import time
import os

from ENV.setup import make
from src.model import PPOTransformerModel
from src.agent import Agent
from src.rollout_buffer import RolloutBuffer
from src.writer import Writer

class Trainer:
    """Train the model"""
    def __init__(self,config,game_name,path) -> None:
        self.config        = config
        self.env           = make(game_name)

        self.model         = PPOTransformerModel(config,self.env.getStateSize(),self.env.getActionSize())
        self.optimizer     = torch.optim.AdamW(self.model.parameters(),lr=config['lr'])
        self.memory_length = config["memory_length"]

        self.writer        = Writer(path)
        self.buffer        = RolloutBuffer()
        self.agent         = Agent()
    
    def _cal_loss(self,value,value_new,entropy,log_prob,log_prob_new,rtgs):
        """Calculate Total Loss"""
        advantage       = rtgs - value.detach()
        ratios          = torch.exp(torch.clamp(log_prob_new-log_prob.detach(),min=-20.,max=5.))
        Kl              = kl_divergence(Categorical(logits=log_prob), Categorical(logits=log_prob_new))

        actor_loss      = -torch.where(
                            (Kl >= self.policy_kl_range) & (ratios >= 1),
                            ratios * advantage - self.policy_params * Kl,
                            ratios * advantage
                        ).mean()
        # print(actor_loss)
        value_clipped   = value + torch.clamp(value_new - value, -self.value_clip, self.value_clip)

        critic_loss     = 0.5 * torch.max((rtgs-value_new)**2,(rtgs-value_clipped)**2).mean()
        total_loss      = actor_loss + self.critic_coef * critic_loss - self.data["entropy_coef"] * entropy
        
        return total_loss

    def _mask(self):
        return torch.tril(torch.ones((self.memory_length, self.memory_length)), diagonal=-1)

    def train():
        pass
