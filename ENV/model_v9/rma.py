import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class RelativeMultiheadAttention(nn.Module):
    """Calculate Relative Multihead Attention"""
    def __init__(self,
                 embed_dim:int,
                 num_heads:int,
                 dropout=None) -> None:
        """Overview:
            Init AttentionXL.
        Arguments:
            - embed_dim (:obj:`int`): dimension of embedding
            - num_heads (:obj:`int`): number of heads for multihead attention
        """

        super(RelativeMultiheadAttention,self).__init__()
        self.embed_dim  = embed_dim
        self.d          = np.sqrt(embed_dim)
        self.num_heads  = num_heads
        self.heads_dim  = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim % num_heads should be zero."
        
        self.Queries    = nn.Linear(embed_dim,embed_dim)
        self.Keys       = nn.Linear(embed_dim,embed_dim)
        self.Values     = nn.Linear(embed_dim,embed_dim)
        self.Pos        = nn.Linear(embed_dim,embed_dim,bias=False)

        self.U = nn.Parameter(torch.Tensor(self.num_heads,self.heads_dim))
        self.V = nn.Parameter(torch.Tensor(self.num_heads,self.heads_dim))
        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.V)

        self.out_projection = nn.Linear(embed_dim,embed_dim)

        if dropout is not None:
            self.dropout1 = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                pos_embedding: torch.Tensor,
                mask: None)->torch.Tensor:
        """Overview:
            Compute Relative Multi-head Attention.

        Arguments:
            - query (`torch.Tensor`): attention input of shape (cur_seq, bs, input_dim)
            - key (`torch.Tensor`): attention input of shape (full_seq, bs, input_dim)
            - value (`torch.Tensor`): attention input of shape (full_seq, bs, input_dim)
            - pos_embedding (`torch.Tensor`): positional embedding of shape (full_seq, 1, full_seq)
            - mask (`Optional[torch.Tensor|None]`): attention mask of shape (cur_seq, full_seq, 1)
            - full_seq = prev_seq + cur_seq

        Returns:
            - alpha (`torch.Tensor`): attention output of shape (cur_seq, bs, input_dim)
        """
        
        # key shape: (key_len,batch_size,embed_size)
        # query shape: (query_len,batch_size,embed_size)
        # value shape: (value_len,batch_size,embed_size)
        batch_size = query.shape[1]
        query_len,key_len,value_len = query.shape[0], key.shape[0], value.shape[0]
        
        queries = self.Queries(query).view(query_len,batch_size,self.num_heads,self.heads_dim)
        keys    = self.Keys(key).view(key_len,batch_size,self.num_heads,self.heads_dim)
        values  = self.Values(value).view(value_len,batch_size,self.num_heads,self.heads_dim)
        R = self.Pos(pos_embedding).view(-1,self.num_heads,self.heads_dim)

        content_score  = torch.einsum("qbhd,kbhd->bhqk",[queries+self.U,keys])
        position_score = torch.einsum("qbhd,khd->bhqk",[queries+self.V,R])
        position_score = self._rel_shift(position_score)
        attention_score = (content_score + position_score) / self.d # (batch_size,num_heads,query_len,key_len)
        if mask is not None:
            mask = mask.permute(2, 0, 1).unsqueeze(1)  # 1 x 1 x cur_seq x full_seq
            assert mask.shape[2:] == attention_score.shape[2:]  # check shape of mask
            attention_score = attention_score.masked_fill(mask,float("-1e20"))
            
        attention_score = torch.softmax(attention_score,dim=-1)
        attention_score = self.dropout1(attention_score)

        alpha = torch.einsum("bhqk,vbhd->qbhd",[attention_score,values]).view(-1,batch_size,self.embed_dim)
        # alpha shape: (query_len,batch_size,embed_dim)
        return self.dropout2(self.out_projection(alpha))
    

    def _rel_shift(self, x: torch.Tensor, zero_upper: bool = False):
        """
        Overview:
            Relatively shift the attention score matrix.

        Example:
            a00 a01 a02      0 a00 a01 a02       0  a00 a01      a02  0  a10     a02  0   0
            a10 a11 a12  =>  0 a10 a11 a12  =>  a02  0  a10  =>  a11 a12  0  =>  a11 a12  0
            a20 a21 a22      0 a20 a21 a22      a11 a12  0       a20 a21 a22     a20 a21 a22

        Arguments:
            - x (`torch.Tensor`): input tensor of shape (cur_seq, full_seq, bs, head_num).
            - zero_upper (`bool`): if True set the upper-right triangle to zero.
            
        Returns:
            - x (`torch.Tensor`): input after relative shift. Shape (cur_seq, full_seq, bs, head_num).
        """
        x_padded = F.pad(x, [1, 0])  # step 1
        x_padded = x_padded.view(x.size(0), x.size(1), x.size(3) + 1, x.size(2))  # step 2
        x = x_padded[:, :, 1:].view_as(x)  # step 3
        if zero_upper:
            ones = torch.ones((x.size(2), x.size(3))).unsqueeze(0).unsqueeze(0)
            x = x * torch.tril(ones.to(x.device), x.size(3) - x.size(2))  # step 4
        return x




