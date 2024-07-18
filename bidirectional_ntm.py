import torch
import torch.nn as nn
import torch.nn.functional as F
from memory import Memory
from head import ReadHead, WriteHead
from bidirectional_lstm import StackedBidirectionalLSTMController

class NTM(nn.Module):
    """
    Neural Turing Machine (NTM) implementation.

    This class implements an NTM, a type of neural network architecture that combines a neural network controller
    with external memory, allowing the network to read from and write to the memory.
    """

    def __init__(self, input_dim, output_dim, ctrl_dim, memory_rows, memory_cols, num_heads, shift_dim, num_layers):
        """
        Initialize the NTM.

        Parameters:
        - input_dim (int): Dimensionality of the input data.
        - output_dim (int): Dimensionality of the output data.
        - ctrl_dim (int): Dimensionality of the controller's hidden state.
        - memory_rows (int): Number of rows in the memory matrix.
        - memory_cols (int): Number of columns in the memory matrix.
        - num_heads (int): Number of read/write heads.
        - shift_dim (int): Dimensionality of the shift vector.
        - num_layers (int): Number of stacked LSTM layers.
        """
        super(NTM, self).__init__()

        self.ctrl_dim = ctrl_dim

        # Create the LSTM-based controller
        self.controller = StackedBidirectionalLSTMController(input_dim + num_heads * memory_cols,
                                        ctrl_dim,
                                        output_dim,
                                        ctrl_dim + num_heads * memory_cols,
                                        num_layers)

        # Create the memory
        self.memory = Memory(memory_rows, memory_cols)
        self.memory_rows = memory_rows
        self.memory_cols = memory_cols

        # Create the read and write heads
        self.num_heads = num_heads
        self.heads = nn.ModuleList([])
        for head in range(num_heads):
            self.heads += [ReadHead(2*ctrl_dim, memory_cols, shift_dim),
                           WriteHead(2*ctrl_dim, memory_cols, shift_dim)]

        # Initialize previous head weights and read vectors
        self.prev_head_weights = []
        self.prev_reads = []

        # Layers to initialize the weights and read vectors
        self.head_weights_fc = nn.Linear(1, self.memory_rows)
        self.reads_fc = nn.Linear(1, self.memory_cols)

        self.reset()

    def forward(self, x):
        """
        Forward pass through the NTM.

        Parameters:
        - x (torch.Tensor): Input tensor.

        Returns:
        - output (torch.Tensor): Output tensor.
        """
        # Get controller hidden and cell states
        ctrl_hidden, ctrl_cell = self.controller(x, self.prev_reads)

        # Read and write operations
        reads = []
        head_weights = []

        for head, prev_head_weights in zip(self.heads, self.prev_head_weights):
            if head.is_read_head():
                weights, read_vec = head(ctrl_hidden, prev_head_weights, self.memory)
                reads.append(read_vec)
            else:
                weights = head(ctrl_hidden, prev_head_weights, self.memory)

            head_weights.append(weights)

        # Compute the output from the controller
        output = self.controller.output(reads)

        # Update previous head weights and reads
        self.prev_head_weights = head_weights
        self.prev_reads = reads

        return output

    def reset(self, batch_size=1):
        """
        Reset/initialize the NTM parameters.

        Parameters:
        - batch_size (int): Size of the batch. Default is 1.
        """
        # Reset memory and controller states
        self.memory.reset(batch_size)
        self.controller.reset(batch_size)

        # Initialize previous head weights (attention vectors)
        self.prev_head_weights = []
        for _ in range(len(self.heads)):
            prev_weight = F.softmax(self.head_weights_fc(torch.Tensor([[0.]])), dim=1)
            self.prev_head_weights.append(prev_weight)

        # Initialize previous read vectors
        self.prev_reads = []
        for _ in range(self.num_heads):
            prev_read = self.reads_fc(torch.Tensor([[0.]]))
            self.prev_reads.append(prev_read)