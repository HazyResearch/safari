import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

from src.utils.train import OptimModule

class LongConvKernel(OptimModule):
    def __init__(
        self, 
        H, 
        L,
        channels=1, 
        learning_rate=None, 
        lam=0.1, 
        causal=True, 
        kernel_dropout=0,
        weight_init="random",
        use_ma_smoothing = False,
        ma_window_len = 7,
        smooth_freq = False,
        **kwargs
    ):
        super().__init__()
       
        self.drop = torch.nn.Dropout(p=kernel_dropout)
        self.H = H
        self.weight_init = weight_init
        self.causal = causal
        self.L = L*2 if not causal else L
        
        self.channels = channels
        self.lam = lam
        self.kernel = torch.nn.Parameter(self._parameter_initialization()) #(c,H,L) 

        self.register("kernel", self.kernel, learning_rate)
        
        self.use_ma_smoothing=use_ma_smoothing
        self.smooth_freq = smooth_freq
        self.ma_window_len = ma_window_len
        if self.use_ma_smoothing:
            if smooth_freq:
                weight = torch.arange(ma_window_len, dtype = self.kernel.dtype)
                weight = torch.exp(-0.5 * torch.abs(weight - ma_window_len // 2) ** 2)
                weight = repeat(weight, 'l -> h1 h2 l', h1 = self.H, h2 = 1)
                weight = weight.type(torch.fft.rfft(self.kernel).dtype)
                self.smooth_weight = weight
            else:
                self.ma_window_len = ma_window_len
                assert self.ma_window_len%2!=0, "window size must be odd"
                padding = (self.ma_window_len//2)
                self.smooth = torch.nn.AvgPool1d(kernel_size=self.ma_window_len,stride=1,padding=padding)

    def _parameter_initialization(self):
        if self.weight_init=="random":
            return torch.randn(self.channels, self.H, self.L) * 0.002
        elif self.weight_init=="double_exp":
            K = torch.randn(self.channels, self.H, self.L,dtype=torch.float32) * 0.02
            double_exp = torch.zeros((self.H,self.L),dtype=torch.float32)
            for i in range(self.H):
                for j in range(self.L):
                    double_exp[i,j] = torch.exp(-(j/self.L)*torch.pow(torch.tensor(int(self.H/2)),torch.tensor(i/self.H)))
            K = torch.einsum("c h l, h l -> c h l",K,double_exp)
            return K
        else: raise NotImplementedError(f"{self.weight_init} is not valid") 

    def forward(self, **kwargs):
        k = self.kernel
        if self.use_ma_smoothing:
            if self.smooth_freq:
                k_f = torch.fft.rfft(k, dim=-1)
                k_f = F.conv1d(k_f, self.smooth_weight.to(k_f.device), padding='same', groups=self.H)
                k = torch.fft.irfft(k_f, dim=-1)
            else:
                k = self.smooth(k)
        k = F.relu(torch.abs(k)-self.lam)*torch.sign(k)
        k = self.drop(k)
        return k, None

    @property
    def d_output(self):
        return self.H