import sys

from pathlib import Path
import torch

import torch.utils.benchmark as benchmark

from safari.models.sequence.hyena import HyenaOperator
from flash_attn.flash_attention import FlashMHA

def benchmark_forward(fn, *inputs, repeats = 10, desc='', verbose=True, **kwinputs):
    if verbose:
        print(desc, '- Forward pass')
    t = benchmark.Timer(
            stmt='fn(*inputs, **kwinputs)',
            globals={'fn': fn, 'inputs': inputs, 'kwinputs': kwinputs},
            num_threads=torch.get_num_threads(),
            )
    m = t.timeit(repeats)
    if verbose:
        print(m)
    return t, m

def benchmark_backward(fn, *inputs, grad=None, repeats=10, desc='', verbose=True, **kwinputs):
    if verbose:
        print(desc, '- Backward pass')
    y = fn(*inputs, **kwinputs)
    if not hasattr(y, 'shape'):
        y = y[0]
    if grad is None:
        grad = torch.randn_like(y)
    else:
        if grad.shape != y.shape:
            raise RuntimeError('Grad shape does not match output shape')
    t = benchmark.Timer(
            stmt='y.backward(grad, retain_graph=True)',
            globals={'y': y, 'grad': grad},
            num_threads=torch.get_num_threads(),
            )
    m = t.timeit(repeats)
    if verbose:
        print(m)
    return t, m

DIM = 768
torch.manual_seed(0)
batch_size = 1
dtype = torch.float16
device = torch.device(f"cuda")

runtime_mha, runtime_hyena = {}, {}
runtime_bwd_mha, runtime_bwd_hyena = {}, {}

for SEQ_LEN in [2048, 4096, 8192, 16384, 32768, 65536, 131072]:
    print(SEQ_LEN)
    mha = FlashMHA(embed_dim=DIM, num_heads=12, causal=True, device=device, dtype=dtype)
    hyena = HyenaOperator(d_model=DIM, l_max=SEQ_LEN, fused_fft_conv=False, groups=3*DIM, fused_bias_fc=True, modulate=False, emb_dim=33, d_state=64).to(device).to(dtype)

    x = torch.ones((batch_size, SEQ_LEN, DIM), dtype=dtype, device=device)

    m, t = benchmark_forward(mha, x, repeats=10, desc='', verbose=False)
    runtime_mha[SEQ_LEN] = t.mean
    m, t = benchmark_backward(mha, x, repeats=10, desc='', verbose=False)
    runtime_bwd_mha[SEQ_LEN] = t.mean
    
    m, t = benchmark_forward(hyena, x, repeats=10, desc='', verbose=False)
    runtime_hyena[SEQ_LEN] = t.mean
    m, t = benchmark_backward(hyena, x, repeats=10, desc='', verbose=False)
    runtime_bwd_hyena[SEQ_LEN] = t.mean
    

print('---')
print(runtime_mha)
print(runtime_bwd_mha)

print('---')
print(runtime_hyena)
print(runtime_bwd_hyena)


