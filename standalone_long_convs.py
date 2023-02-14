'''
Standalone Long Conv class.

The `LongConvModel` class defined in this file provides a simple backbone to train models.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from opt_einsum import contract


class OurModule(nn.Module):
    """ Interface for Module that allows registering buffers/parameters with configurable optimizer hyperparameters """

    def register(self, name, tensor, lr=None, wd=0.0):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {}
            if lr is not None: optim["lr"] = lr
            if wd is not None: optim["weight_decay"] = wd
            setattr(getattr(self, name), "_optim", optim)

class LongConv(OurModule):
    def __init__(
            self,
            H,
            L,
            channels=2,
            dropout=0.1,
            kernel_learning_rate=None, 
            kernel_lam=0.1, 
            kernel_dropout=0,
    ):
        super().__init__()
        self.H = H
        self.L = L * 2 # for causal conv
        self.channels = channels
        self.dropout = nn.Dropout(p=dropout)
        self.kernel_learning_rate = kernel_learning_rate
        self.kernel_lam = kernel_lam
        self.kernel_drop = torch.nn.Dropout(p=kernel_dropout)

        self.D = nn.Parameter(torch.randn(channels, self.H))

        # Pointwise
        self.activation = nn.GELU()

        # output transform to mix features
        self.output_linear = nn.Sequential(
            nn.Linear(self.channels * self.H, 2 * self.H, bias=True),
            nn.GLU(dim=-1),
        )

        self.kernel = torch.nn.Parameter(torch.randn(self.channels, self.H, self.L) * 0.002) #(c,H,L) 

        self.register("kernel", self.kernel, kernel_learning_rate)

    def forward(self, u):
        L = u.size(-1)

        k = self.kernel

        # squash operator
        k = F.relu(torch.abs(k)-self.kernel_lam)*torch.sign(k)
        k = self.kernel_drop(k)
        
        # use FFT to compute convolution
        k_f = torch.fft.rfft(k, n=2*L)
        u_f = torch.fft.rfft(u, n=2*L)
        y_f = contract('bhl,chl->bchl', u_f, k_f)
        y = torch.fft.irfft(y_f, n=2*L)[..., :L]

        # Compute skip connection
        y = y + contract('bhl,ch->bchl', u, self.D)

        # Reshape to flatten channels
        y = rearrange(y, '... c h l -> ... (c h) l')

        y = self.dropout(self.activation(y))

        # Transpose for the linear
        y = y.transpose(-1, -2)
        y = self.output_linear(y)
        y = y.transpose(-1, -2)

        return y

class LongConvModel(nn.Module):

    def __init__(
        self,
        d_input,
        d_output=10,
        d_model=512,
        n_layers=6,
        dropout=0.1,
        prenorm=False,
        **conv_kwargs,
    ):
        super().__init__()

        self.prenorm = prenorm

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.encoder = nn.Linear(d_input, d_model)

        # Stack S4 layers as residual blocks
        self.conv_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.conv_layers.append(
                LongConv(d_model, L=1024, dropout=dropout, **conv_kwargs)
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(nn.Dropout1d(dropout))

        # Linear decoder
        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x):
        """
        Input x is shape (B, L, d_input)
        """
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for layer, norm, dropout in zip(self.conv_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)
            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Apply long conv block
            z = layer(z)

            # Dropout on the output of the conv block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)

        # Pooling: average pooling over the sequence length
        x = x.mean(dim=1)

        # Decode the outputs
        x = self.decoder(x)  # (B, d_model) -> (B, d_output)

        return x
