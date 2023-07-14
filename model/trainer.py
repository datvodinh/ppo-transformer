from torch.distributions import Categorical,kl_divergence
import torch
import torch.nn as nn
import numpy as np
import time
import os
from model.model import PPOTransformerModel
from model.agent import Agent
from model.writer import Writer
from model.distribution import Distribution

class Trainer:
    """Train the model"""
    def __init__(self,config,env,writer_path=None) -> None:
        self.config        = config
        self.env           = env

        self.model         = PPOTransformerModel(config,self.env.getStateSize(),self.env.getActionSize())
        self.optimizer     = torch.optim.AdamW(self.model.parameters(),lr=config['lr'])
        self.memory_length = config["memory_length"]
        if writer_path is not None:
            self.writer    = Writer(writer_path)
        self.agent         = Agent(self.env,self.model,config)
        self.dist          = Distribution()

    def _truly_loss(self,value,value_new,entropy,log_prob,log_prob_new,Kl,advantage):
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
        #Calculate returns and advantage
        
        returns         = value + advantage

        if self.config["normalize_advantage"]:
            advantage   = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        ratios          = torch.exp(torch.clamp(log_prob_new-log_prob.detach(),min=-20.,max=5.))
        print(ratios)
        R_dot_A = ratios * advantage
        actor_loss      = -torch.where(
                            (Kl >= self.config["policy_kl_range"]) & (R_dot_A > advantage),
                            R_dot_A - self.config["policy_params"] * Kl,
                            R_dot_A - self.config["policy_kl_range"]
                        )

        value_clipped         = value + torch.clamp(value_new - value, -self.config["value_clip"], self.config["value_clip"])
        critic_loss           = 0.5 * torch.max((returns-value_new)**2,(returns-value_clipped)**2)
  
        total_loss            = actor_loss + self.config["critic_coef"] * critic_loss - self.config["entropy_coef"] * entropy

        return actor_loss.mean(), critic_loss.mean(), total_loss.mean(), entropy.mean()
    
    def _remove_padding(self,
                 t:torch.Tensor,
                 padding:torch.Tensor,
                 value = -100):
        """
        Overviews:
            Return tensor with padding for policy and value loss.
        Arguments:
            - t: (`torch.Tensor`): tensor
            - padding: (`torch.Tensor`): padding mask
        Returns:
            - Tensor t without padding
        """
        t =  t.masked_fill(padding==0,value=float(str(value)))
        return t[t!=value]

    def train(self,write_data=True):
        training = True

        step = 0
        while training:

            win_rate = self.agent.run(num_games=self.config["num_game_per_batch"])
            self.agent.rollout.cal_advantages(self.config["gamma"],self.config["gae_lambda"])
            self.agent.rollout.prepare_batch()
            self.model.train()
            
            for _ in range(self.config["num_epochs"]):
                mini_batch_loader   = self.agent.rollout.mini_batch_generator()
                for mini_batch in mini_batch_loader:

                    sliced_memory   = self.agent.rollout.batched_index_select(
                                        mini_batch["memory"],
                                        1,
                                        mini_batch["memory_indices"])
                    pol_new,val_new,_  = self.model(mini_batch["states"],sliced_memory,mini_batch["memory_mask"],mini_batch["memory_indices"])
                    val_new         = val_new.squeeze(1)
                    log_prob_new, entropy = self.dist.log_prob(pol_new,mini_batch["actions"].view(1,-1),mini_batch["action_mask"])

                    Kl = self.dist.kl_divergence(mini_batch["policy"],pol_new)
                    actor_loss, critic_loss, total_loss, entropy = self._truly_loss(
                        value        = mini_batch["values"].reshape(-1).detach(),
                        value_new    = val_new,
                        entropy      = entropy,
                        log_prob     = mini_batch["probs"].reshape(-1).detach(),
                        log_prob_new = log_prob_new,
                        Kl           = Kl,
                        advantage    = mini_batch["advantages"].reshape(-1).detach(),
                    )
                    with torch.autograd.set_detect_anomaly(self.config["set_detect_anomaly"]):
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
                                    reward      = self.agent.rollout.batch["rewards"].mean(),
                                    entropy     = entropy,
                                    actor_loss  = actor_loss,
                                    critic_loss = critic_loss,
                                    total_loss  = total_loss,
                                    kl_mean     = Kl.mean().item(),
                                    kl_max      = Kl.max().item(),
                                    kl_min      = Kl.min().item()
                                )
                            step+=1
            
            self.agent.rollout.reset_data()

    def _save_model(model:PPOTransformerModel,path):
        torch.save(model.state_dict(),path)

