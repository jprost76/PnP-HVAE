import numpy as np
import torch
import torch.nn as nn

try:
    from .conv2d import Conv2d
except (ImportError, ValueError, ModuleNotFoundError):
    from model.conv2d import Conv2d

from hparams import HParams

hparams = HParams.get_hparams_by_name("patch_vdvae")


def _std_mode(x, prior_stats, softplus):
    mean, std = torch.chunk(x, chunks=2, dim=1)  # B, C, H, W
    std = softplus(std)

    if prior_stats is not None:
        mean = mean + prior_stats[0]
        std = std * prior_stats[1]

    stats = [mean, std]
    return mean, std, stats


def _logstd_mode(x, prior_stats):
    mean, logstd = torch.chunk(x, chunks=2, dim=1)

    if prior_stats is not None:
        mean = mean + prior_stats[0]
        logstd = logstd + prior_stats[1]

    std = torch.exp(hparams.model.gradient_smoothing_beta * logstd)
    stats = [mean, logstd]

    return mean, std, stats


class GaussianLatentLayer(nn.Module):
    def __init__(self, in_filters, num_variates, min_std=np.exp(-2)):
        super(GaussianLatentLayer, self).__init__()

        self.projection = Conv2d(
            in_channels=in_filters,
            out_channels=num_variates * 2,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding='same'
        )

        self.min_std = min_std
        self.softplus = torch.nn.Softplus(beta=hparams.model.gradient_smoothing_beta)

    def forward(self, x, temperature=None, prior_stats=None, return_sample=True):
        x = self.projection(x)

        if hparams.model.distribution_base == 'std':
            mean, std, stats = _std_mode(x, prior_stats, self.softplus)
        elif hparams.model.distribution_base == 'logstd':
            mean, std, stats = _logstd_mode(x, prior_stats)
        else:
            raise ValueError(f'distribution base {hparams.model.distribution_base} not known!!')

        if temperature is not None:
            std = std * temperature

        if return_sample:
            z, mean, std = calculate_z(mean, std)
            return z, stats
        return stats

class FirstGaussianLatentLayer(nn.Module):
    # z ~ N(z;0, I)
    def __init__(self, num_variates):
        super(FirstGaussianLatentLayer, self).__init__()
        self.num_variates = num_variates
        self.mean = nn.Parameter(torch.zeros(1, num_variates, 1, 1), requires_grad=False)
        self.std = nn.Parameter(torch.ones(1, num_variates, 1, 1), requires_grad=False)

    def forward(self, x, temperature=None, prior_stats=None, return_sample=True):
        # x : dummy input to get the wanted input shape
        B, _, H, W = x.shape
        mean = self.mean.expand(B, self.num_variates, H, W)
        std = self.std.expand(B, self.num_variates, H, W)
        stats = [mean, std]
        if temperature is not None:
            std = std * temperature

        if return_sample:
            z, mean, std = calculate_z(mean, std)
            return z, stats
        return stats

def interpolate(post_stats, prior_stats, lq, lp):
    #return z = arg min -lq . log q(z|x) -lp log p(z)
    softplus = torch.nn.Softplus(beta=hparams.model.gradient_smoothing_beta)
    if hparams.model.distribution_base == 'std':
        mp, stdp = prior_stats
        mq, stdq = post_stats
    elif hparams.model.distribution_base == 'logstd':
        mp, logstdp = prior_stats
        mq, logstdq = post_stats
        stdp = torch.exp(hparams.model.gradient_smoothing_beta * logstdp)
        stdq = torch.exp(hparams.model.gradient_smoothing_beta * logstdq)
    else:
        raise ValueError(f'distribution base {hparams.model.distribution_base} not known!!')
    a = lq / (stdq**2)
    b = lp / (stdp**2)
    z = (a*mq + b*mp) / (a+b) # TODO : jit?
    return z

    

    
@torch.jit.script
def calculate_z(mean, std):
    eps = torch.empty_like(mean, device=torch.device('cuda')).normal_(0., 1.)
    z = eps * std + mean
    return z, mean, std

#@torch.jit.script
#def gaussian_ll_logstd(mu, logsigma, z):
#    log2pi = torch.log(torch.tensor(2*torch.pi))
#    return - 0.5 * torch.sum(log2pi + logsigma + (z-mu)**2 / torch.exp(logsigma)) 

@torch.jit.script
def gaussian_ll_std(mu, sigma, z):
    log2pi = torch.log(torch.tensor(2*torch.pi))
    return - 0.5 * torch.sum(log2pi + 2*torch.log(sigma) + (z-mu)**2 / sigma**2) 

def gaussian_ll(stats, z, t=1):
    if hparams.model.distribution_base == 'std':
        mp, stdp = stats
    elif hparams.model.distribution_base == 'logstd':
        mp, logstdp = stats
        stdp = torch.exp(hparams.model.gradient_smoothing_beta * logstdp)
    else:
        raise ValueError(f'distribution base {hparams.model.distribution_base} not known!!')
    stdp = stdp / t
    ll = gaussian_ll_std(mp, stdp, z)
    return ll