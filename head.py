import torch
import torch.nn.functional as F
from torch import nn

class HeadBase(nn.Module):
    """
    Base class for Neural Turing Machine heads.

    Args:
        ctrl_dim (int): Dimension of the controller state.
        memory_cols (int): Number of columns in the memory matrix.
        shift_dim (int): Dimension of the shift vector.
    """
    def __init__(self, ctrl_dim, memory_cols, shift_dim):
        super(HeadBase, self).__init__()
        self.memory_cols = memory_cols
        self.shift_dim = shift_dim

        # Create a dictionary of linear layers for addressing parameters
        self.fc_layers = nn.ModuleDict({
            'key': nn.Linear(ctrl_dim, memory_cols),
            'key_strength': nn.Linear(ctrl_dim, 1),
            'interpolation_gate': nn.Linear(ctrl_dim, 1),
            'shift_weighting': nn.Linear(ctrl_dim, shift_dim),
            'sharpen_factor': nn.Linear(ctrl_dim, 1),
            'erase_weight': nn.Linear(ctrl_dim, memory_cols),
            'add_data': nn.Linear(ctrl_dim, memory_cols)
        })

        # Initialize the layers
        self._reset()

    def _reset(self):
        """
        Initializes the weights of the linear layers using Xavier uniform distribution
        and biases with a normal distribution.
        """
        for layer in self.fc_layers.values():
            nn.init.xavier_uniform_(layer.weight, gain=1.4)
            nn.init.normal_(layer.bias, std=0.01)

    def forward(self, ctrl_state, prev_weights, memory):
        """
        Forward pass for the head.

        Args:
            ctrl_state (torch.Tensor): Controller state.
            prev_weights (torch.Tensor): Previous weights.
            memory (torch.Tensor): Memory matrix.

        Returns:
            torch.Tensor: Updated weights.
            torch.Tensor: Erase vector.
            torch.Tensor: Add vector.
        """
        # Compute addressing parameters from controller state
        params = {name: layer(ctrl_state) for name, layer in self.fc_layers.items()}
        # Calculate weights based on parameters and memory
        weights = self._calc_weights(params, prev_weights, memory)
        return weights, params['erase_weight'], params['add_data']

    def _calc_weights(self, params, prev_weights, memory):
        """
        Calculates the addressing weights.

        Args:
            params (dict): Dictionary of parameters.
            prev_weights (torch.Tensor): Previous weights.
            memory (torch.Tensor): Memory matrix.

        Returns:
            torch.Tensor: Updated weights.
        """
        key = torch.tanh(params['key'])
        beta = F.softplus(params['key_strength'])
        gate = torch.sigmoid(params['interpolation_gate'])
        shift = F.softmax(params['shift_weighting'], dim=1)
        gamma = 1 + F.softplus(params['sharpen_factor'])

        # Content addressing
        content_weights = self._content_addressing(key, beta, memory.memory)
        # Gated interpolation with previous weights
        gated_weights = self._gated_interpolation(content_weights, prev_weights, gate)
        # Circular convolution shift
        shifted_weights = self._conv_shift(gated_weights, shift)
        # Sharpening
        weights = self._sharpen(shifted_weights, gamma)
        return weights

    def _content_addressing(self, key, beta, memory):
        """
        Performs content-based addressing.

        Args:
            key (torch.Tensor): Key vector.
            beta (torch.Tensor): Key strength.
            memory (torch.Tensor): Memory matrix.

        Returns:
            torch.Tensor: Content-based weights.
        """
        similarity_scores = F.cosine_similarity(key.unsqueeze(1), memory, dim=2)
        content_weights = F.softmax(beta * similarity_scores, dim=1)
        return content_weights

    def _gated_interpolation(self, w, prev_w, g):
        """
        Interpolates between the content weights and previous weights.

        Args:
            w (torch.Tensor): Content weights.
            prev_w (torch.Tensor): Previous weights.
            g (torch.Tensor): Gate scalar.

        Returns:
            torch.Tensor: Interpolated weights.
        """
        return g * w + (1 - g) * prev_w

    def _conv_shift(self, w, s):
        """
        Performs a circular convolution shift.

        Args:
            w (torch.Tensor): Weights to be shifted.
            s (torch.Tensor): Shift weighting.

        Returns:
            torch.Tensor: Shifted weights.
        """
        batch_size, shift_size = w.size(0), s.size(1)
        max_shift = (shift_size - 1) // 2
        unrolled = torch.cat([w[:, -max_shift:], w, w[:, :max_shift]], dim=1)
        conv_out = F.conv1d(unrolled.unsqueeze(1), s.unsqueeze(1))[range(batch_size), range(batch_size)]
        return conv_out

    def _sharpen(self, w, gamma):
        """
        Sharpens the weights.

        Args:
            w (torch.Tensor): Weights.
            gamma (torch.Tensor): Sharpening factor.

        Returns:
            torch.Tensor: Sharpened weights.
        """
        w = w.pow(gamma)
        return w / (w.sum(dim=1, keepdim=True) + 1e-16)

class ReadHead(HeadBase):
    """
    Read head for Neural Turing Machine.
    Inherits from HeadBase.

    Args:
        ctrl_dim (int): Dimension of the controller state.
        memory_cols (int): Number of columns in the memory matrix.
        shift_dim (int): Dimension of the shift vector.
    """
    def __init__(self, ctrl_dim, memory_cols, shift_dim):
        super(ReadHead, self).__init__(ctrl_dim, memory_cols, shift_dim)

    def forward(self, ctrl_state, prev_weights, memory):
        """
        Forward pass for the read head.

        Args:
            ctrl_state (torch.Tensor): Controller state.
            prev_weights (torch.Tensor): Previous weights.
            memory (torch.Tensor): Memory matrix.

        Returns:
            torch.Tensor: Updated weights.
            torch.Tensor: Read vector.
        """
        weights, _, _ = super(ReadHead, self).forward(ctrl_state, prev_weights, memory)
        read_vec = memory.read(weights)
        return weights, read_vec

    def is_read_head(self):
        return True

class WriteHead(HeadBase):
    """
    Write head for Neural Turing Machine.
    Inherits from HeadBase.

    Args:
        ctrl_dim (int): Dimension of the controller state.
        memory_cols (int): Number of columns in the memory matrix.
        shift_dim (int): Dimension of the shift vector.
    """
    def __init__(self, ctrl_dim, memory_cols, shift_dim):
        super(WriteHead, self).__init__(ctrl_dim, memory_cols, shift_dim)

    def forward(self, ctrl_state, prev_weights, memory):
        """
        Forward pass for the write head.

        Args:
            ctrl_state (torch.Tensor): Controller state.
            prev_weights (torch.Tensor): Previous weights.
            memory (torch.Tensor): Memory matrix.

        Returns:
            torch.Tensor: Updated weights.
        """
        weights, erase_vec, add_vec = super(WriteHead, self).forward(ctrl_state, prev_weights, memory)
        memory.write(weights, erase_vec, add_vec)
        return weights

    def is_read_head(self):
        return False