'''PyTorch version of the block FFT convolution as described in the H3 paper.'''

import torch
from einops import rearrange
import math
from torch import nn
from src.models.nn import Activation
from src.utils.train import OptimModule

def ref_dft_matrix(N, H=1):
    """Compute the DFT matrix of size N x N.
    
    This is where we could add extra compute for free."""
    # n = torch.arange(N)
    n = torch.arange(N).cuda()
    k = n.view(-1, 1)
    M = torch.exp(-2j * torch.pi * n * k / N)
    return torch.view_as_real(M.repeat(H, 1, 1))

def compute_twiddle_factors(n, m):
    """Compute the twiddle factors of size n x m"""
    # n_a = torch.arange(n).view(-1, 1)
    # m_a = torch.arange(m)
    n_a = torch.arange(n).cuda().view(-1, 1)
    m_a = torch.arange(m).cuda()
    N = n * m
    M = torch.exp(-2j * torch.pi * n_a * m_a / N)
    return torch.view_as_real(M)

def _cooley_tukey(
    k, n, m, 
    dft_matrix=ref_dft_matrix,
    max_m=16,
    activation=None,
):
    '''
    Compute the FFT using the general Cooley-Tukey algorithm:
        * Reshape to (m, n)
        * Do n m-length FFTs along the rows
        * Transpose to (n, m), multiply by twiddle factors
        * Do m n-length FFTs along the rows

    This function assumes that m <= 16 and recurses on n.
    The base case is n <= 16 (we are simulating tensor cores of 16x16 mm).
    The dft_matrix function is overwriteable
    so that we can replace it with learnable parameters in a model.
    '''
    assert m <= max_m

    if activation is not None:
        act_fn = Activation(activation)

    k = rearrange(k, '... (m n) -> ... m n', m=m, n=n) # (m, n)

    # do n m-length FFTs
    if activation is None:
        mat = torch.view_as_complex(dft_matrix(m))
        k_f = torch.einsum('... m o, ... o n -> ... m n', mat, k) # (..., m, n)
    else:
        mat = torch.view_as_complex(dft_matrix(m))
        k_f = torch.view_as_complex(act_fn(
            torch.view_as_real(torch.einsum('... m o, ... o n -> ... m n', mat, k))
        )) # (..., m, n)

    # multiply by twiddle factors
    twi = torch.view_as_complex(compute_twiddle_factors(n, m)) # (n, m)
    k_f = torch.einsum('n m, ... m n -> ... n m', twi, k_f) # (..., n, m)

    if n <= max_m:
        # do m n-length FFTs
        if activation is None:
            mat = torch.view_as_complex(dft_matrix(n))
            k_f = torch.einsum('... n o, ... o m -> ... n m', mat, k_f) # (.., n, m)
        else:
            mat = torch.view_as_complex(dft_matrix(n))
            k_f = torch.view_as_complex(act_fn(
                torch.view_as_real(torch.einsum('... n o, ... o m -> ... n m', mat, k_f))
            )) # (.., n, m)
    else:
        # recurse
        k_f = rearrange(k_f, '... h n m -> ... m h n')
        k_f = _cooley_tukey(k_f, n // max_m, max_m, dft_matrix, max_m, activation)
        k_f = rearrange(k_f, '... m h n -> ... h n m')

    # reshape for the output
    k_f = rearrange(k_f, '... n m -> ... (n m)') # (..., n*m)

    return k_f

def block_fft(
    k, N,
    dft_matrix=ref_dft_matrix,
    max_m=16,
    **kwargs,
):
    '''
    Compute the FFT of size N of the vector k, using _block_fft_recurse.
    
    The dft_matrix function is overwriteable
    so that we can replace it with learnable parameters in a model.
    '''
    if not math.log(N, 2).is_integer():
        N = int(2 ** math.ceil(math.log(N, 2)))
    # pad k with zeros if necessary (e.g. for causality)
    if k.shape[-1] != N:
        k = nn.ConstantPad1d((0, N - k.shape[-1]), 0)(k)
    
    if N <= max_m:
        mat = torch.view_as_complex(dft_matrix(m))
        return torch.einsum('... n o, ... o -> ... n', mat, k) # (.., n, m)
    n = N // max_m
    m = max_m
    return _cooley_tukey(k, n, m, dft_matrix, max_m, **kwargs)

class BlockFFT(OptimModule):
    '''
    Learnable Block FFT module.

    Args:
        learn_dft_matrix (bool): If True, learn a different DFT matrix for lengths 2, 4, 8, and 16. If False, this module computes a normal FFT.
    '''
    def __init__(self, learn_dft_matrices=True, H=1, max_m=16, dft_lr=0.001, dropout=0, learn_additive=False, **block_fft_args):
        super().__init__()
        self.learn_dft_matrices = learn_dft_matrices
        self.block_fft_args = block_fft_args
        self.max_m=max_m
        self.drop = torch.nn.Dropout(p=dropout)
        self.learn_additive=learn_additive
        # get the powers of 2 up to max_m
        assert math.log(max_m, 2).is_integer(), 'max_m must be a power of 2'

        self.powers = [ 2 ** (i + 1) for i in range(int(math.log(max_m, 2))) ]

        if learn_dft_matrices:
            assert dft_lr>0,"If learn_dft_matrices=True dft_lr must be positive"
            self.dft_matrices = nn.ParameterList()
            for n in self.powers:
                setattr(self,f"mat_{n}",nn.Parameter(
                    0.01 * torch.randn(H, n, n, 2) if self.learn_additive
                    else ref_dft_matrix(n, H=H),
                    requires_grad=True))
                self.register(f"mat_{n}",getattr(self,f"mat_{n}"),dft_lr)
                self.dft_matrices.append(getattr(self,"mat_{}".format(n)))

    def compute_dft_matrix(self, n):
        if not self.learn_dft_matrices:
            return ref_dft_matrix(n)
        else:
            assert n in self.powers
            if self.learn_additive:
                mat = ref_dft_matrix(n)
                return mat + self.drop(self.dft_matrices[int(math.log(n, 2) - 1)])
            else:
                return self.drop(self.dft_matrices[int(math.log(n, 2) - 1)])

    def forward(self, x, N,forward=True):
        '''Compute an FFT (forward=True) or iFFT (forward=False) of length N over x.'''
        if forward:
            return block_fft(x, N, dft_matrix=self.compute_dft_matrix, **self.block_fft_args)
        else:
            return (1/(N))*torch.conj(block_fft(torch.conj(x), N, dft_matrix=self.compute_dft_matrix, **self.block_fft_args))


if __name__ == "__main__":
    B = 128
    H = 29
    N = 8192
    n = 2
    m = 8
    k = torch.randn(B, H, N).to(torch.complex64)

    print(f'(B, H, N) = ({B}, {H}, {N})')    

    # test FFT    
    k_f = block_fft(k, N)
    k_f_ref = torch.fft.fft(k, N)
    print('L-inf error in FFT: ', torch.max(torch.abs(k_f - k_f_ref)).item())