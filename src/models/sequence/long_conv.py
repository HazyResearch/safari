import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import opt_einsum as oe

optimized = True

if optimized:
    contract = oe.contract
else:
    contract = torch.einsum

from src.models.nn import LinearActivation, Activation, DropoutNd
from src.models.sequence.block_fft import BlockFFT
from src.models.sequence.long_conv_kernel import LongConvKernel

class LongConv(nn.Module):
    def __init__(
            self,
            d_model,
            l_max=1024,
            channels=1,
            bidirectional=False,
            # Arguments for position-wise feedforward components
            activation='gelu', # activation between conv and FF
            postact='glu', # activation after FF
            initializer=None, # initializer on FF
            weight_norm=False, # weight normalization on FF
            dropout=0.0, tie_dropout=False,
            transposed=True, # axis ordering (B, L, D) or (B, D, L)
            verbose=False,
            block_fft_conv=False, # replace the FFT conv with Monarch blocks
            block_fft_conv_args={},

            # SSM Kernel arguments
            **kernel_args,
        ):
        """
        d_state: the dimension of the state, also denoted by N
        l_max: the maximum kernel length, also denoted by L
        channels: can be interpreted as a number of "heads"; the SSM is a map from a 1-dim to C-dim sequence. It's not recommended to change this unless desperate for things to tune; instead, increase d_model for larger models
        bidirectional: if True, convolution kernel will be two-sided

        Position-wise feedforward components:
        --------------------
        activation: activation in between SS and FF
        postact: activation after FF ('id' for no activation, None to remove FF layer)
        initializer: initializer on FF
        weight_norm: weight normalization on FF
        dropout: standard dropout argument. tie_dropout=True ties the dropout mask across the sequence length, emulating nn.Dropout1d

        Other arguments:
        --------------------
        transposed: choose backbone axis ordering of (B, L, H) (if False) or (B, H, L) (if True) [B=batch size, L=sequence length, H=hidden dimension]
        """

        super().__init__()
        if verbose:
            import src.utils.train
            log = src.utils.train.get_logger(__name__)
            log.info(f"Constructing Long Conv (H, L) = ({d_model}, {l_max})")

        self.d_model = d_model
        self.H = d_model
        self.L = l_max
        self.bidirectional = bidirectional
        self.channels = channels
        self.transposed = transposed
        self.block_fft_conv = block_fft_conv
        self.block_fft_conv_args = block_fft_conv_args

        self.D = nn.Parameter(torch.randn(channels, self.H))

        if self.bidirectional:
            channels *= 2

        # SSM Kernel
        self.kernel = LongConvKernel(self.H, L=self.L, channels=channels, verbose=verbose, **kernel_args)

        if self.block_fft_conv:
            self.block_fft_u = BlockFFT(**self.block_fft_conv_args)
            self.block_fft_k = BlockFFT(**self.block_fft_conv_args)
            
        # Pointwise
        self.activation = Activation(activation)
        # dropout_fn = nn.Dropout2d if self.transposed else nn.Dropout # Broken in torch==1.11
        dropout_fn = DropoutNd if tie_dropout else nn.Dropout
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        if postact is None:
            self.output_linear = nn.Identity()
        else:
            self.output_linear = LinearActivation(
                self.d_model * self.channels,
                self.d_model,
                # self.H*self.channels,
                # self.d_model*(1 if self.gate is None else self.gate),
                transposed=self.transposed,
                initializer=initializer,
                activation=postact,
                activate=True,
                weight_norm=weight_norm,
            )



    def forward(self, u, state=None, rate=1.0, lengths=None, **kwargs): # absorbs return_output and transformer src mask
        """
        u: (B H L) if self.transposed else (B L H)
        state: (H N) never needed, remnant from state spaces repo

        Returns: same shape as u
        """
        if not self.transposed: u = u.transpose(-1, -2)
        L = u.size(-1)
        # Mask out padding tokens
        # TODO handle option for mask - instead of lengths, which assumes suffix padding
        if isinstance(lengths, int):
            if lengths != L:
                lengths = torch.tensor(lengths, dtype=torch.long, device=u.device)
            else:
                lengths = None
        if lengths is not None:
            assert isinstance(lengths, torch.Tensor) and lengths.ndim == 1 and lengths.size(0) in [1, u.size(0)]
            mask = torch.where(torch.arange(L, device=lengths.device) < lengths[:, None, None], 1., 0.)
            u = u * mask

        # Compute SS Kernel
        L_kernel = L if self.L is None else min(L, round(self.L / rate))
        k, _ =  self.kernel(L=L_kernel, rate=rate, state=state) # (C H L) (B C H L)

        # Convolution
        if self.bidirectional:
            k0, k1 = rearrange(k, '(s c) h l -> s c h l', s=2)
            k = F.pad(k0, (0, L)) \
                    + F.pad(k1.flip(-1), (L, 0))

        if self.block_fft_conv:
            k_f = self.block_fft_k(k.to(torch.complex64), N=L_kernel+L) # (C H L)
            u_f = self.block_fft_u(u.to(torch.complex64), N=L_kernel+L) # (B H L)
            y_f = contract('bhl,chl->bchl', u_f, k_f)
            if self.learn_ifft:
                y = self.block_fft_u(y_f, N=L_kernel+L,forward=False).real[..., :L]
            else:
                y = torch.fft.ifft(y_f, n=L_kernel+L, dim=-1).real[..., :L] # (B C H L)
        else:
            k_f = torch.fft.rfft(k, n=L_kernel+L) # (C H L)
            u_f = torch.fft.rfft(u, n=L_kernel+L) # (B H L)
            y_f = contract('bhl,chl->bchl', u_f, k_f)
            y = torch.fft.irfft(y_f, n=L_kernel+L)[..., :L] # (B C H L)

        # Compute skip connection
        y = y + contract('bhl,ch->bchl', u, self.D)

        # Reshape to flatten channels
        y = rearrange(y, '... c h l -> ... (c h) l')

        if not self.transposed: y = y.transpose(-1, -2)
        y = self.activation(y)
        y = self.dropout(y)
        y = self.output_linear(y)

        return y, None

    @property
    def d_state(self):
        return self.H

    @property
    def d_output(self):
        return self.d_model
