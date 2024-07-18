import torch
from torch import nn
import torch.nn.functional as F

class StackedBidirectionalLSTMController(nn.Module):
    """
    Stacked Bidirectional LSTM Controller for Neural Turing Machines (NTMs).

    This class implements a controller with multiple stacked bidirectional LSTM cells used in NTMs to process input data
    and generate hidden states, which are then used to read from and write to external memory.
    """

    def __init__(self, input_dim, ctrl_dim, output_dim, read_data_dim, num_layers):
        """
        Initialize the StackedBidirectionalLSTMController.

        Parameters:
        - input_dim (int): Dimensionality of the input data.
        - ctrl_dim (int): Dimensionality of the controller's hidden state.
        - output_dim (int): Dimensionality of the output data.
        - read_data_dim (int): Dimensionality of the read data.
        - num_layers (int): Number of stacked LSTM layers.
        """
        super(StackedBidirectionalLSTMController, self).__init__()

        self.input_dim = input_dim
        self.ctrl_dim = ctrl_dim
        self.output_dim = output_dim
        self.read_data_dim = read_data_dim
        self.num_layers = num_layers

        # Forward LSTM cells
        self.forward_net = nn.ModuleList([nn.LSTMCell(input_dim, ctrl_dim) if i == 0
                                        else nn.LSTMCell(read_data_dim, ctrl_dim) for i in range(num_layers)])
        
        # Backward LSTM cells
        self.backward_net = nn.ModuleList([nn.LSTMCell(input_dim, ctrl_dim) if i == 0
                                        else nn.LSTMCell(read_data_dim, ctrl_dim) for i in range(num_layers)])
        
        # Output neural network (Linear layer)
        self.out_net = nn.Linear(ctrl_dim + read_data_dim, output_dim)
        
        # Initialize the weights of the output network using Kaiming uniform initialization
        nn.init.kaiming_uniform_(self.out_net.weight)

        # Reset the controller states
        self.reset()

    def forward(self, x, prev_reads):
        """
        Forward pass through the StackedBidirectionalLSTMController.

        Parameters:
        - x (torch.Tensor): Input tensor.
        - prev_reads (list of torch.Tensor): List of previous read vectors.

        Returns:
        - h_states (torch.Tensor): Concatenated hidden states of the forward and backward LSTM layers.
        - c_states (torch.Tensor): Concatenated cell states of the forward and backward LSTM layers.
        """
        # Concatenate input and previous read vectors
        x_combined = torch.cat([x, *prev_reads], dim=1)

        # Forward pass through the LSTM cells
        h_states_f, c_states_f = self._process_lstm_cells(x_combined, prev_reads, self.forward_net, self.h_states_f, self.c_states_f)

        # Backward pass through the LSTM cells
        h_states_b, c_states_b = self._process_lstm_cells(x_combined, prev_reads, self.backward_net, self.h_states_b, self.c_states_b, backward = True)

        # Concatenate the hidden states from forward and backward passes
        self.h_states = [torch.cat([hf, hb], dim=1) for hf, hb in zip(h_states_f, h_states_b)]
        self.c_states = [torch.cat([cf, cb], dim=1) for cf, cb in zip(c_states_f, c_states_b)]

        # Compute the average of the concatenated hidden states
        self.avg_h_state = torch.mean(torch.stack(self.h_states), dim=0)
        self.avg_c_state = torch.mean(torch.stack(self.c_states), dim=0)

        return self.avg_h_state, self.avg_c_state

    def _process_lstm_cells(self, x, prev_reads, lstm_cells, h_states, c_states, backward = False):
        """
        Process the input through the stacked LSTM cells.

        Parameters:
        - x (torch.Tensor): Input tensor.
        - lstm_cells (nn.ModuleList): LSTM cells to process the input.
        - h_states (list of torch.Tensor): Hidden states of the LSTM layers.
        - c_states (list of torch.Tensor): Cell states of the LSTM layers.

        Returns:
        - h_states (list of torch.Tensor): Hidden states of the LSTM layers.
        - c_states (list of torch.Tensor): Cell states of the LSTM layers.
        """
        h_states_list = []
        c_states_list = []
        for i, lstm_cell in enumerate(lstm_cells):
            if backward:
                x = torch.flip(x, [1])
            h, c = lstm_cell(x, (h_states[i], c_states[i]))
            x = torch.cat([h, *prev_reads], dim=1)
            h_states_list.append(h)
            c_states_list.append(c)

        return h_states_list, c_states_list

    def output(self, reads):
        """
        Compute the external output from the read vectors.

        Parameters:
        - reads (list of torch.Tensor): List of read vectors.

        Returns:
        - output (torch.Tensor): External output tensor.
        """
        # Concatenate averaged hidden state and read vectors
        out_state = torch.cat([self.avg_h_state, *reads], dim=1)
        
        # Compute the output using the output network and apply sigmoid activation
        output = torch.sigmoid(self.out_net(out_state))

        return output

    def reset(self, batch_size=1):
        """
        Reset/initialize the controller states.

        Parameters:
        - batch_size (int): Size of the batch. Default is 1.
        """
        # Initialize hidden and cell states with zeros and repeat them for the batch size
        self.h_states_f = [torch.zeros(batch_size, self.ctrl_dim) for _ in range(self.num_layers)]
        self.c_states_f = [torch.zeros(batch_size, self.ctrl_dim) for _ in range(self.num_layers)]
        self.h_states_b = [torch.zeros(batch_size, self.ctrl_dim) for _ in range(self.num_layers)]
        self.c_states_b = [torch.zeros(batch_size, self.ctrl_dim) for _ in range(self.num_layers)]