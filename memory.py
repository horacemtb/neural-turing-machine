import torch
import torch.nn.functional as F
from torch import nn

class Memory(nn.Module):

    def __init__(self, memory_rows, memory_cols):
        super(Memory, self).__init__()

        self.n = memory_rows  # Number of memory rows (units) in a Memory matrix
        self.m = memory_cols  # Number of memory columns (unit vector size) in a Memory matrix

        # Initialize the memory to be None; it will be set in reset()
        self.memory = None

    def read(self, weights):
        '''Returns a read vector using the attention weights
        Args:
            weights (tensor): attention weights (batch_size, N)
        Returns:
            (tensor): read vector (batch_size, M)
        '''
        assert self.memory is not None, "Memory must be initialized using reset() before read()"
        assert weights.size(1) == self.n, f"Expected weights of size (batch_size, {self.n}), got {weights.size()}"

        read_vec = torch.matmul(weights.unsqueeze(1), self.memory).squeeze(1)
        return read_vec

    def write(self, weights, erase_vec, add_vec):
        '''Erases and Writes a new memory matrix
        Args:
            weights (tensor): attention weights (batch_size, N)
            erase_vec (tensor): erase vector (batch_size, M)
            add_vec (tensor): add vector (batch_size, M)
        '''
        assert self.memory is not None, "Memory must be initialized using reset() before write()"
        assert weights.size(1) == self.n, f"Expected weights of size (batch_size, {self.n}), got {weights.size()}"
        assert erase_vec.size(1) == self.m, f"Expected erase_vec of size (batch_size, {self.m}), got {erase_vec.size()}"
        assert add_vec.size(1) == self.m, f"Expected add_vec of size (batch_size, {self.m}), got {add_vec.size()}"

        # Erase
        memory_erased = self.memory * (1 - weights.unsqueeze(2) * erase_vec.unsqueeze(1))
        # Add
        self.memory = memory_erased + (weights.unsqueeze(2) * add_vec.unsqueeze(1))

    def reset(self, batch_size=1):
        '''Reset/initialize the memory
        Args:
            batch_size (int): the batch size for the memory initialization
        '''
        self.memory = torch.full((batch_size, self.n, self.m), 1e-6)