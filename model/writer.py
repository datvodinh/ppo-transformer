from torch.utils.tensorboard import SummaryWriter
import os
class Writer:
    def __init__(self, path):
        """
        Overview: Initialize a Writer object to save statistics and plots using Tensorboard.
        
        Arguments:
            path (str): Path to the directory where Tensorboard logs will be saved.
        """
        self.writer = SummaryWriter(log_dir=path)

    def add(self, step, win_rate, reward, entropy, actor_loss, critic_loss, total_loss, kl_mean, kl_max, kl_min):
        """
        Overview: Add scalar values to Tensorboard for various training statistics.
        
        Arguments:
            step (int): Training step or iteration number.
            win_rate (float): Win rate value.
            reward (float): Reward value.
            entropy (float): Entropy value.
            actor_loss (float): Actor loss value.
            critic_loss (float): Critic loss value.
            total_loss (float): Total loss value.
            kl_mean (float): Mean value of KL divergence.
            kl_max (float): Maximum value of KL divergence.
            kl_min (float): Minimum value of KL divergence.
        """
        self.writer.add_scalar("A.Train/Win Rate", win_rate, step)
        self.writer.add_scalar("A.Train/Reward", reward, step)
        self.writer.add_scalar("B.Loss/Entropy", entropy, step)
        self.writer.add_scalar("B.Loss/ActorLoss", actor_loss, step)
        self.writer.add_scalar("B.Loss/CriticLoss", critic_loss, step)
        self.writer.add_scalar("B.Loss/TotalLoss", total_loss, step)
        self.writer.add_scalar("C.Kl_divergence/mean", kl_mean, step)
        self.writer.add_scalar("C.Kl_divergence/max", kl_max, step)
        self.writer.add_scalar("C.Kl_divergence/min", kl_min, step)

    def close(self):
        """
        Overview: Close the Writer object and flush any pending data to Tensorboard.
        """
        self.writer.close()
