""" Isotropic deep sequence model backbone, in the style of ResNets / Transformers.

The SequenceModel class implements a generic (batch, length, d_input) -> (batch, length, d_output) transformation
"""

from functools import partial

import torch
import torch.nn as nn
from einops import rearrange

from src.utils.config import to_list, to_dict
from src.models.sequence.block import SequenceResidualBlock
from src.models.sequence.base import SequenceModule
from src.models.nn.components import Normalization, DropoutNd


class SequenceModel(SequenceModule):
    def __init__(
        self,
        d_model,           # Resize input (useful for deep models with residuals)
        n_layers=1,        # Number of layers
        transposed=False,  # Transpose inputs so each layer receives (batch, dim, length)
        dropout=0.0,       # Dropout parameter applied on every residual and every layer
        tie_dropout=False, # Tie dropout mask across sequence like nn.Dropout1d/nn.Dropout2d
        prenorm=True,      # Pre-norm vs. post-norm
        n_repeat=1,        # Each layer is repeated n times per stage before applying pooling
        layer=None,        # Layer config, must be specified
        residual=None,     # Residual config
        norm=None,         # Normalization config (e.g. layer vs batch)
        pool=None,         # Config for pooling layer per stage
        track_norms=True,  # Log norms of each layer output
        dropinp=0.0,       # Input dropout
    ):
        super().__init__()
        # Save arguments needed for forward pass
        self.d_model = d_model
        self.transposed = transposed
        self.track_norms = track_norms

        # Input dropout (not really used)
        dropout_fn = partial(DropoutNd, transposed=self.transposed) if tie_dropout else nn.Dropout
        self.drop = dropout_fn(dropinp) if dropinp > 0.0 else nn.Identity()

        layer = to_list(layer, recursive=False)

        # Some special arguments are passed into each layer
        for _layer in layer:
            # If layers don't specify dropout, add it
            if _layer.get('dropout', None) is None:
                _layer['dropout'] = dropout
            # Ensure all layers are shaped the same way
            _layer['transposed'] = transposed

        # Duplicate layers
        layers = layer * n_layers * n_repeat

        # Instantiate layers
        _layers = []
        d = d_model
        for l, layer in enumerate(layers):
            # Pool at the end of every n_repeat blocks
            pool_cfg = pool if (l+1) % n_repeat == 0 else None
            block = SequenceResidualBlock(d, l+1, prenorm=prenorm, dropout=dropout, tie_dropout=tie_dropout, transposed=transposed, layer=layer, residual=residual, norm=norm, pool=pool_cfg)
            _layers.append(block)
            d = block.d_output

        self.d_output = d
        self.layers = nn.ModuleList(_layers)
        if prenorm:
            if norm is None:
                self.norm = None
            elif isinstance(norm, str):
                self.norm = Normalization(self.d_output, transposed=self.transposed, _name_=norm)
            else:
                self.norm = Normalization(self.d_output, transposed=self.transposed, **norm)
        else:
            self.norm = nn.Identity()

    def forward(self, inputs, *args, state=None, **kwargs):
        """ Inputs assumed to be (batch, sequence, dim) """
        if self.transposed: inputs = rearrange(inputs, 'b ... d -> b d ...')
        inputs = self.drop(inputs)

        # Track norms
        if self.track_norms: output_norms = [torch.mean(inputs.detach() ** 2)]

        # Apply layers
        outputs = inputs
        prev_states = [None] * len(self.layers) if state is None else state
        next_states = []
        for layer, prev_state in zip(self.layers, prev_states):
            outputs, state = layer(outputs, *args, state=prev_state, **kwargs)
            next_states.append(state)
            if self.track_norms: output_norms.append(torch.mean(outputs.detach() ** 2))
        if self.norm is not None: outputs = self.norm(outputs)

        if self.transposed: outputs = rearrange(outputs, 'b d ... -> b ... d')

        if self.track_norms:
            metrics = to_dict(output_norms, recursive=False)
            self.metrics = {f'norm/{i}': v for i, v in metrics.items()}

        return outputs, next_states

    @property
    def d_state(self):
        d_states = [layer.d_state for layer in self.layers]
        return sum([d for d in d_states if d is not None])

    @property
    def state_to_tensor(self):
        # Slightly hacky way to implement this in a curried manner (so that the function can be extracted from an instance)
        # Somewhat more sound may be to turn this into a @staticmethod and grab subclasses using hydra.utils.get_class
        def fn(state):
            x = [_layer.state_to_tensor(_state) for (_layer, _state) in zip(self.layers, state)]
            x = [_x for _x in x if _x is not None]
            return torch.cat( x, dim=-1)
        return fn

    def default_state(self, *batch_shape, device=None):
        return [layer.default_state(*batch_shape, device=device) for layer in self.layers]

    def step(self, x, state, **kwargs):
        # Apply layers
        prev_states = [None] * len(self.layers) if state is None else state
        next_states = []
        for layer, prev_state in zip(self.layers, prev_states):
            x, state = layer.step(x, state=prev_state, **kwargs)
            next_states.append(state)

        x = self.norm(x)

        return x, next_states
