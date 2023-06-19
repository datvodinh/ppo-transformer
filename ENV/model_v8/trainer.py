from torch.distributions import Categorical,kl_divergence
import torch
import torch.nn as nn
import numpy as np
import time
import os

from setup import make
from model_v8.model import PPOTransformerModel
from model_v8.agent import Agent
from model_v8.writer import Writer

class Trainer:
    """Train the model"""
    def __init__(self,config,game_name,writer_path=None) -> None:
        self.config        = config
        self.env           = make(game_name)

        self.model         = PPOTransformerModel(config,self.env.getStateSize(),self.env.getActionSize())
        self.optimizer     = torch.optim.AdamW(self.model.parameters(),lr=config['lr'])
        self.memory_length = config["memory_length"]
        if writer_path is not None:
            self.writer    = Writer(writer_path)
        self.agent         = Agent(self.env,self.model,config)
    
    def _cal_loss(self,value,value_new,entropy,log_prob,log_prob_new,advantage):
        """
        Overview:
            Calculate Total Loss

        Arguments:
            - value: (`torch.Tensor`):  
            - value_new: (`torch.Tensor`): 
            - entropy: (`torch.Tensor`): 
            - log_prob: (`torch.Tensor`): 
            - log_prob_new: (`torch.Tensor`): 
            - advantage: (`torch.Tensor`):  

        Return:
            - actor_loss: (`torch.Tensor`): 
            - critic_loss: (`torch.Tensor`): 
            - total_loss: (`torch.Tensor`): 

        
        """
        
        returns         = value + advantage
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        ratios          = torch.exp(torch.clamp(log_prob_new-log_prob.detach(),min=-20.,max=5.))
        Kl              = kl_divergence(Categorical(logits=log_prob), Categorical(logits=log_prob_new))

        actor_loss      = -torch.where(
                            (Kl >= self.config["policy_kl_range"]) & (ratios >= 1),
                            ratios * advantage - self.config["policy_params"] * Kl,
                            ratios * advantage
                        ).mean()

        value_clipped   = value + torch.clamp(value_new - value, -self.config["value_clip"], self.config["value_clip"])
        
        critic_loss     = 0.5 * torch.max((returns-value_new)**2,(returns-value_clipped)**2).mean()
        total_loss      = actor_loss + self.config["critic_coef"] * critic_loss - self.config["entropy_coef"] * entropy
        
        return actor_loss, critic_loss, total_loss

    def train(self,write_data=True):
        training = True

        step = 0
        while training:
            win_rate = self.agent.run(num_games=self.config["num_game_per_batch"])
            self.agent.to_tensor()
            self.agent.cal_advantages(self.config["gamma"],self.config["gae_lambda"])
            
            for _ in range(self.config["num_epochs"]):
                mini_batch_loader = self.agent.mini_batch_loader(self.config)
                for _ in range(self.config["n_mini_batch"]):
                    mini_batch      = next(mini_batch_loader)
                    pol_new,val_new = self.model(mini_batch["states"])
                    val_new         = val_new.squeeze(1)
                    # print(pol_new, mini_batch["action_mask"])
                    categorical_new = Categorical(logits=pol_new.masked_fill(mini_batch["action_mask"]==0,float('-inf')))
                    log_prob_new    = categorical_new.log_prob(mini_batch["actions"].view(1,-1)).squeeze(0)
                    entropy         = categorical_new.entropy().mean()

                    actor_loss, critic_loss, total_loss = self._cal_loss(
                        value        = mini_batch["values"],
                        value_new    = val_new,
                        entropy      = entropy,
                        log_prob     = mini_batch["probs"],
                        log_prob_new = log_prob_new,
                        advantage    = mini_batch["advantages"]
                    )

                    if not torch.isnan(total_loss).any():
                        self.optimizer.zero_grad()
                        total_loss.backward()
                        nn.utils.clip_grad_norm_(self.model.parameters(),max_norm=self.config["max_grad_norm"])
                        self.optimizer.step()
                    if write_data:
                        with torch.no_grad():
                            self.writer.add(
                                step        = step,
                                win_rate    = win_rate,
                                reward      = self.agent.batch["rewards"].mean(),
                                entropy     = entropy,
                                actor_loss  = actor_loss,
                                critic_loss = critic_loss,
                                total_loss  = total_loss
                            )
                            step+=1
            
            self.agent.reset_data()

    def _save_model(model:PPOTransformerModel,path):
        torch.save(model.state_dict(),path)

