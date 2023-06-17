from typing import Optional, List
import torch

class Memory:
    def __init__(
            self,
            memory_len: int = 20,
            batch_size: int = 64,
            embedding_dim: int = 256,
            num_blocks: int = 2,
            memory: Optional[torch.Tensor] = None
    ) -> None:

        super(Memory, self).__init__()
        self.embedding_dim = embedding_dim
        self.bs = batch_size
        self.num_blocks = num_blocks
        self.memory_len = memory_len
        self.memory = None
        self.init_memory(memory)

    def init_memory(self, memory: Optional[torch.Tensor] = None):
        """
        Overview:
            Init memory with an input list of tensors or create it automatically given its dimensions.
        
        Arguments:
            - memory: (:obj:`Optional[torch.Tensor]`): memory input.
            Shape is (layer_num, memory_len, bs, embedding_dim).
            memory_len is length of memory, bs is batch size and embedding_dim is the dimension of embedding.
        """

        if memory is not None:
            self.memory = memory
            num_blocks_plus1, self.memory_len, self.bs, self.embedding_dim = memory.shape
            self.num_blocks = num_blocks_plus1 - 1
        else:
            self.memory = torch.zeros(
                self.num_blocks + 1, self.memory_len, self.bs, self.embedding_dim, dtype=torch.float
            )

    def update(self, hidden_state: List[torch.Tensor]):
        """
        Overview:
            Update the memory given a sequence of hidden states.
        Example for single layer:

            memory_len=3, hidden_size_len=2, bs=3

                m00 m01 m02      h00 h01 h02              m20 m21 m22
            m = m10 m11 m12  h = h10 h11 h12  => new_m =  h00 h01 h02
                m20 m21 m22                               h10 h11 h12

        Arguments:
            - hidden_state: (:obj:`List[torch.Tensor]`): hidden states to update the memory.
            - Shape is (cur_seq, bs, embedding_dim) for each layer. cur_seq is length of sequence.

        Returns:
            - memory: (:obj:`Optional[torch.Tensor]`): output memory.
            - Shape is (layer_num, memory_len, bs, embedding_dim).
        """

        if self.memory is None or hidden_state is None:
            raise ValueError('Failed to update memory! Memory would be None')
        sequence_len = hidden_state[0].shape[0]
        with torch.no_grad():
            new_memory = []
            end = self.memory_len + sequence_len
            start = max(0, end - self.memory_len)
            for i in range(self.num_blocks + 1):
                m = self.memory[i]
                h = hidden_state[i]
                cat = torch.cat([m, h], dim=0)
                new_memory.append(cat[start:end].detach())
        new_memory = torch.stack(new_memory, dim=0)
        self.memory = new_memory
        return new_memory

    def get(self):
        """
        Overview:
            Memory getter method.

        Returns:
            - memory: (:obj:`Optional[torch.Tensor]`): output memory.
            Shape is (layer_num, memory_len, bs, embedding_dim).
        """
        return self.memory