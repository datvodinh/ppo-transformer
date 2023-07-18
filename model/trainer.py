from torch.distributions import Categorical,kl_divergence
import torch
import torch.nn as nn
import json
from model.model import PPOTransformerModel
from model.agent import Agent
from model.writer import Writer
from model.distribution import Distribution

class Trainer:
    """Train the model"""
    def __init__(self,config,env,writer_path=None,save_path=None) -> None:
        self.config        = config
        self.env           = env

        self.model         = PPOTransformerModel(config,self.env.getStateSize(),self.env.getActionSize())
        self.optimizer     = torch.optim.AdamW(self.model.parameters(),lr=config['lr'])
        self.memory_length = config["transformer"]["memory_length"]
        if writer_path is not None:
            self.writer    = Writer(writer_path)
        if save_path is not None:
            self.save_path = save_path

        self.agent         = Agent(self.env,self.model,config)
        self.dist          = Distribution()

        try:
            self.model.load_state_dict(torch.load(f'{save_path}model.pt'))
            with open(f"{save_path}stat.json","r") as f:
                self.data = json.load(f)
            print('PROGRESS RESTORED!')
        except:
            print("TRAIN FROM BEGINING!")
            self.data = {
                "step":0,
                "entropy_coef":config["entropy_coef"]["start"]
            }

        self.entropy_coef = self.data["entropy_coef"]
        self.entropy_coef_step = (config["entropy_coef"]["start"] - config['entropy_coef']['end']) / config['entropy_coef']['step']
        

    def _entropy_coef_schedule(self):
        self.entropy_coef -= self.entropy_coef_step
        if self.entropy_coef <= self.config['entropy_coef']['end']:
            self.entropy_coef = self.config['entropy_coef']['end']

    def _truly_loss(self,value,value_new,entropy,log_prob,log_prob_new,Kl,advantage):
        """
        Overview:
            Calculates the total loss using Truly PPO method.

        Arguments:
            - value: (`torch.Tensor`): The predicted values.
            - value_new: (`torch.Tensor`): The updated predicted values.
            - entropy: (`torch.Tensor`): The entropy.
            - log_prob: (`torch.Tensor`): The log probabilities of the actions.
            - log_prob_new: (`torch.Tensor`): The updated log probabilities of the actions.
            - Kl: (`torch.Tensor`): The KL divergence.
            - advantage: (`torch.Tensor`): The advantages.

        Returns:
            - actor_loss: (`torch.Tensor`): The actor loss.
            - critic_loss: (`torch.Tensor`): The critic loss.
            - total_loss: (`torch.Tensor`): The total loss.
            - entropy: (`torch.Tensor`): The entropy.
        """
        #Calculate returns and advantage
        
        returns         = value + advantage

        if self.config["normalize_advantage"]:
            advantage   = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        ratios          = torch.exp(torch.clamp(log_prob_new-log_prob.detach(),min=-20.,max=5.))
        R_dot_A = ratios * advantage
        actor_loss      = -torch.where(
                            (Kl >= self.config["PPO"]["policy_kl_range"]) & (R_dot_A > advantage),
                            R_dot_A - self.config["PPO"]["policy_params"] * Kl,
                            R_dot_A - self.config["PPO"]["policy_kl_range"]
                        )

        value_clipped         = value + torch.clamp(value_new - value, -self.config["PPO"]["value_clip"], self.config["PPO"]["value_clip"])
        critic_loss           = 0.5 * torch.max((returns-value_new)**2,(returns-value_clipped)**2)
  
        total_loss            = actor_loss + self.config["PPO"]["critic_coef"] * critic_loss - self.entropy_coef * entropy

        return actor_loss.mean(), critic_loss.mean(), total_loss.mean(), entropy.mean()
    
    def train(self,write_data=True):
        """
        Overview:
            Trains the model.

        Arguments:
            - write_data: (`bool`): Whether to write data to Tensorboard or not.
        """
        training = True

        while training:

            win_rate = self.agent.run(num_games=self.config["num_game_per_batch"])
            self.agent.rollout.cal_advantages(self.config["PPO"]["gamma"],self.config["PPO"]["gae_lambda"])
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
                    log_prob_new, entropy = self.dist.log_prob(pol_new,mini_batch["actions"].view(1,-1),mini_batch["action_mask"])

                    Kl = self.dist.kl_divergence(mini_batch["policy"],pol_new)
                    actor_loss, critic_loss, total_loss, entropy = self._truly_loss(
                        value        = mini_batch["values"].reshape(-1).detach(),
                        value_new    = val_new.squeeze(1),
                        entropy      = entropy,
                        log_prob     = mini_batch["probs"].reshape(-1).detach(),
                        log_prob_new = log_prob_new.squeeze(0),
                        Kl           = Kl,
                        advantage    = mini_batch["advantages"].reshape(-1).detach(),
                    )
                    with torch.autograd.set_detect_anomaly(self.config["set_detect_anomaly"]):
                        if not torch.isnan(total_loss).any():
                            self.optimizer.zero_grad()
                            total_loss.backward()
                            nn.utils.clip_grad_norm_(self.model.parameters(),max_norm=self.config["max_grad_norm"])
                            self.optimizer.step()
                            self._entropy_coef_schedule()
                    if write_data:  
                        with torch.no_grad():
                            self.writer.add(
                                    step        = self.data["step"],
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
                            self._save_log()
                        
            if (self.data["step"]%200==0): 
                self._save_model()
            self.agent.rollout.reset_data()

    def _save_model(self):
        """
        Overview:
            Saves the model and other data.
        """
        torch.save(self.model.state_dict(), f'{self.save_path}model.pt')
        with open(f"{self.save_path}stat.json","w") as f:
                json.dump(self.data,f)

    def _save_log(self):
        self.data["step"]+=1
        self.data["entropy_coef"] = self.entropy_coef

