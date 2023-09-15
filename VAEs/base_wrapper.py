import torch
import torch.nn as nn

class BaseVAE(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        
    def eval_logpxz(self, x, z, T, decoder_std):
        raise NotImplemented
    
    def latent_reg(self, x, beta, T, mode, sample_from_prior_after=None, dec_std=None):
        raise NotImplemented

    def encode(self, x):
        raise NotImplemented

