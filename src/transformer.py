import torch
import torch.nn as nn
import torch.nn.functional as F
from src.rma import *
from src.gru import *

class SinusoidalPE(nn.Module):
    """Relative positional encoding"""
    def __init__(self, dim, min_timescale = 2., max_timescale = 1e4):
        super().__init__()
        freqs     = torch.arange(0, dim, min_timescale)
        inv_freqs = max_timescale ** (-freqs / dim)
        self.register_buffer('inv_freqs', inv_freqs)

    def forward(self, seq_len):
        seq            = torch.arange(seq_len - 1, -1, -1.)
        sinusoidal_inp = seq.view(-1,1) * self.inv_freqs.view(1,-1)
        pos_emb        = torch.cat((sinusoidal_inp.sin(), sinusoidal_inp.cos()), dim = -1)
        return pos_emb

class TransformerBlock(nn.Module):
    """Transformer Block"""
    def __init__(self,embed_dim,num_heads,config):
        super().__init__()
        self.attention   = RelativeMultiheadAttention(embed_dim,num_heads)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.gate1       = GRUGate(embed_dim,config['gru_bias'])
        self.gate2       = GRUGate(embed_dim,config['gru_bias'])

        self.fc = nn.Sequential(
            nn.Linear(embed_dim,embed_dim),
            nn.ReLU(),
        )

    def forward(self,query,key,value,mask=None):
        norm_key = self.layer_norm1(key)
        Y        = self.attention(self.layer_norm1(query),norm_key,norm_key,mask)
        out      = self.gate1(query,Y)
        E        = self.fc(self.layer_norm2(out))
        out      = self.gate2(out,E)

        return out




class GatedTransformerXL(nn.Module):
    """Gated Transformer XL model"""
    def __init__(self, config, input_dim, max_episode_steps=500) -> None:
        super().__init__()
        self.config            = config
        self.num_blocks        = config["num_blocks"]
        self.embed_dim         = config["embed_dim"]
        self.num_heads         = config["num_heads"]
        self.max_episode_steps = max_episode_steps
        self.activation        = nn.ReLU()

        # Input embedding layer
        self.linear_embedding  = nn.Linear(input_dim, self.embed_dim)
        nn.init.orthogonal_(self.linear_embedding.weight, np.sqrt(2))
        self.pos_embedding     = SinusoidalPE(dim = self.embed_dim)
        
        # Instantiate transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(self.embed_dim, self.num_heads, config) 
            for _ in range(self.num_blocks)])

    def forward(self, h, memories, mask, memory_indices):
        # Feed embedding layer and activate
        h = self.activation(self.linear_embedding(h))

        # Positional embedding
        pos_embedding = self.pos_embedding(self.max_episode_steps)[memory_indices]
        memories      = memories + pos_embedding.unsqueeze(2)

        # Forward transformer blocks
        out_memories = []
        for i, block in enumerate(self.transformer_blocks):
            out_memories.append(h.detach())
            out = block(out.unsqueeze(1), memories[:, :, i], memories[:, :, i],  mask) # args: query, value, key,  mask
            out = out.squeeze()
            if len(out.shape) == 1:
                out = out.unsqueeze(0)
        return out, torch.stack(out_memories, dim=1)
