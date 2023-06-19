import torch
import torch.nn as nn
import torch.nn.functional as F
from model_v8.rma import *
from model_v8.gru import *
from model_v8.memory import *

class SinusoidalPE(nn.Module):
    """Relative positional encoding"""
    def __init__(self, dim, min_timescale = 2., max_timescale = 1e4):
        super().__init__()
        freqs     = torch.arange(0, dim, min_timescale)
        inv_freqs = max_timescale ** (-freqs / dim)
        self.register_buffer('inv_freqs', inv_freqs)

    def forward(self, seq_len):
        """
        Overview:
            Compute positional embedding

        Arguments:
            - seq_len: (`int`): sequence length.

        Returns:
            - pos_embedding: (:obj:`torch.Tensor`): positional embedding. Shape (seq_len, 1, embedding_dim)
        """
        seq            = torch.arange(int(seq_len) - 1, -1, -1.)
        sinusoidal_inp = seq.view(-1,1) * self.inv_freqs.view(1,-1)
        pos_emb        = torch.cat((sinusoidal_inp.sin(), sinusoidal_inp.cos()), dim = -1)
        return pos_emb.unsqueeze(1)

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
            nn.Linear(embed_dim,4*embed_dim),
            nn.GELU(),
            nn.Linear(4*embed_dim,embed_dim),
            nn.GELU()
        )

    def forward(self,query,key,pos_embedding,mask=None):
        norm_key = self.layer_norm1(key)
        Y        = self.attention(self.layer_norm1(query),norm_key,norm_key,pos_embedding,mask)
        out      = self.gate1(query,nn.GELU()(Y))
        E        = self.fc(self.layer_norm2(out))
        out      = self.gate2(out,E)

        return out




class GatedTransformerXL(nn.Module):
    """Gated Transformer XL model"""
    def __init__(self, 
                 config:dict,
                 input_dim:int,
                 max_episode_steps=500) -> None:
        
        super().__init__()
        self.config            = config
        self.num_blocks        = config["num_blocks"]
        self.embed_dim         = config["embed_dim"]
        self.num_heads         = config["num_heads"]
        self.memory_length     = config["memory_length"]
        self.max_episode_steps = max_episode_steps
        self.activation        = nn.GELU()
        self.memory            = Memory(self.memory_length,config["batch_size"],self.embed_dim,self.num_blocks)

        # Input embedding layer
        self.linear_embedding  = nn.Linear(input_dim, self.embed_dim)
        nn.init.orthogonal_(self.linear_embedding.weight, np.sqrt(2))
        self.pos_embedding     = SinusoidalPE(dim = self.embed_dim)
        
        # Instantiate transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(self.embed_dim, self.num_heads, config) 
            for _ in range(self.num_blocks)])
        
        self.att_mask = {}  # create an attention mask for each different seq_len, in this way we don't need to create a
        # new one each time we call the forward method
        self.pos_embedding_dict = {}  # create a pos embedding for each different seq_len

    def reset_memory(self, 
                     batch_size: Optional[int] = None, 
                     state: Optional[torch.Tensor] = None):
        """
        Overview:
            Clear or set the memory of GTrXL.

        Arguments:
            - batch_size (:obj:`Optional[int]`): batch size
            - state (:obj:`Optional[torch.Tensor]`): input memory. Shape is (layer_num, memory_len, bs, embedding_dim).
        """
        self.memory = Memory(memory_len=self.memory_length, num_blocks=self.num_blocks, embedding_dim=self.embed_dim)
        if batch_size is not None:
            self.memory = Memory(self.memory_length, batch_size, self.embed_dim, self.num_blocks)
        elif state is not None:
            self.memory.init(state)

    def forward(self, h:torch.Tensor):
        """
        Overview:
            GTrXL forward pass.

        Arguments:
            - h (:obj:`torch.Tensor`): input tensor. Shape (seq_len, bs, input_size).

        Returns:
            - x (:obj:`torch.Tensor`): transformer output of shape (seq_len, bs, embedding_size).
        """

        h = torch.transpose(h,1,0) # (batch_size, cur_seq, input_dim) -> (cur_seq, batch_size, input_dim)
        # Feed embedding layer and activate
        cur_seq, bs = h.shape[:2]
        memory = None if self.memory is None else self.memory.get()

        if memory is None:
            self.reset_memory(bs)  # (layer_num+1) x memory_len x batch_size x embedding_dim
        elif memory.shape[-2] != bs or memory.shape[-1] != self.embed_dim:
            self.reset_memory(bs)

        h = self.activation(self.linear_embedding(h))
        
        memory = self.memory.get()
        # Positional embedding
        prev_seq = self.memory_length
        full_seq = cur_seq + prev_seq

        if cur_seq in self.att_mask.keys():
            attn_mask = self.att_mask[cur_seq]
        else:
            attn_mask = (
                torch.triu(
                    torch.ones((cur_seq, full_seq)),
                    diagonal=1 + prev_seq,  # fixed in train, eval, collect
                ).bool().unsqueeze(-1).to(h.device)
            )  # cur_seq x full_seq x 1
            self.att_mask[cur_seq] = attn_mask

        if cur_seq in self.pos_embedding_dict.keys():
            pos_embedding = self.pos_embedding_dict[cur_seq]
        else:
            pos_embedding = self.pos_embedding(full_seq)
            self.pos_embedding_dict[cur_seq] = pos_embedding

        hidden_state = [h]
        out = h
        for i in range(self.num_blocks):
            layer = self.transformer_blocks[i]
            out = layer(
                query=out,
                key=torch.cat([memory[i], out], dim=0),
                pos_embedding=pos_embedding,
                mask=attn_mask
            )  # cur_seq x bs x embedding_dim
            hidden_state.append(out.detach().clone())

        self.memory.update(hidden_state)  # (layer_num+1) x memory_len x batch_size x embedding_dim
        out = torch.transpose(out, 1, 0)  #  (cur_seq, batch_size, input_dim) ->  (batch_size, cur_seq, input_dim)
        
        return out