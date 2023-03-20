import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.jit.script
def gaussian_analytical_kl(mu1, mu2, logsigma1, logsigma2):
    return -0.5 + logsigma2 - logsigma1 + 0.5 * (logsigma1.exp() ** 2 + (mu1 - mu2) ** 2) / (logsigma2.exp() ** 2)


@torch.jit.script
def draw_gaussian_diag_samples(mu, logsigma):
    eps = torch.empty_like(mu).normal_(0., 1.)
    return torch.exp(logsigma) * eps + mu

def gaussian_ll(mu, logsigma, z):
    return - 0.5 * torch.sum(np.log(2 * np.pi) + 2*logsigma + (z-mu)**2 / logsigma.exp()**2) 

def get_conv(in_dim, out_dim, kernel_size, stride, padding, zero_bias=True, zero_weights=False, groups=1, scaled=False):
    c = nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, groups=groups)
    if zero_bias:
        c.bias.data *= 0.0
    if zero_weights:
        c.weight.data *= 0.0
    return c


def get_3x3(in_dim, out_dim, zero_bias=True, zero_weights=False, groups=1, scaled=False):
    return get_conv(in_dim, out_dim, 3, 1, 1, zero_bias, zero_weights, groups=groups, scaled=scaled)


def get_1x1(in_dim, out_dim, zero_bias=True, zero_weights=False, groups=1, scaled=False):
    return get_conv(in_dim, out_dim, 1, 1, 0, zero_bias, zero_weights, groups=groups, scaled=scaled)


def log_prob_from_logits(x):
    """ numerically stable log_softmax implementation that prevents overflow """
    # x.shape : (B, 32, 32, 10) for nr_mix=10
    axis = len(x.shape) - 1
    m = x.max(dim=axis, keepdim=True)[0] # (32, 32, 1)
    return x - m - torch.log(torch.exp(x - m).sum(dim=axis, keepdim=True)) #  (B, 32, 32, 10) 


def const_max(t, constant):
    other = torch.ones_like(t) * constant
    return torch.max(t, other)


def const_min(t, constant):
    other = torch.ones_like(t) * constant
    return torch.min(t, other)


def discretized_mix_logistic_loss(x, l, low_bit=False):
    """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
    x = x.permute(0, 2, 3, 1) # (B,C,H,W) -> (B,H,W,C)
    # Adapted from https://github.com/openai/pixel-cnn/blob/master/pixel_cnn_pp/nn.py
    xs = [s for s in x.shape]  # true image (i.e. labels) to regress to, e.g. (B,32,32,3)
    ls = [s for s in l.shape]  # predicted distribution, e.g. (B,32,32,100)
    nr_mix = int(ls[-1] / 10)  # here and below: unpacking the params of the mixture of logistics
    logit_probs = l[:, :, :, :nr_mix] # (B, 32, 32, 10)
    l = torch.reshape(l[:, :, :, nr_mix:], xs + [nr_mix * 3]) # (B, 32, 32, 3, 30)
    means = l[:, :, :, :, :nr_mix] #(B, 32, 32, 3, 10)
    log_scales = const_max(l[:, :, :, :, nr_mix:2 * nr_mix], -7.) #(B, 32, 32, 3, 10)
    coeffs = torch.tanh(l[:, :, :, :, 2 * nr_mix:3 * nr_mix]) #(B, 32, 32, 3, 10)
    x = torch.reshape(x, xs + [1]) + torch.zeros(xs + [nr_mix]).to(x.device)  # here and below: getting the means and adjusting them based on preceding sub-pixels
    m2 = torch.reshape(means[:, :, :, 1, :] + coeffs[:, :, :, 0, :] * x[:, :, :, 0, :], [xs[0], xs[1], xs[2], 1, nr_mix])
    m3 = torch.reshape(means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] + coeffs[:, :, :, 2, :] * x[:, :, :, 1, :], [xs[0], xs[1], xs[2], 1, nr_mix])
    means = torch.cat([torch.reshape(means[:, :, :, 0, :], [xs[0], xs[1], xs[2], 1, nr_mix]), m2, m3], dim=3)#(B, 32, 32, 3, 10)
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    if low_bit:
        plus_in = inv_stdv * (centered_x + 1. / 31.)
        cdf_plus = torch.sigmoid(plus_in)
        min_in = inv_stdv * (centered_x - 1. / 31.)
    else:
        plus_in = inv_stdv * (centered_x + 1. / 255.)
        cdf_plus = torch.sigmoid(plus_in)
        min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = torch.sigmoid(min_in)
    log_cdf_plus = plus_in - F.softplus(plus_in)  # log probability for edge case of 0 (before scaling)
    log_one_minus_cdf_min = -F.softplus(min_in)  # log probability for edge case of 255 (before scaling)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * centered_x
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)  # log probability in the center of the bin, to be used in extreme cases (not actually used in our code)

    # now select the right output: left edge case, right edge case, normal case, extremely low prob case (doesn't actually happen for us)

    # this is what we are really doing, but using the robust version below for extreme cases in other applications and to avoid NaN issue with tf.select()
    # log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999, log_one_minus_cdf_min, tf.log(cdf_delta)))

    # robust version, that still works if probabilities are below 1e-5 (which never happens in our code)
    # tensorflow backpropagates through tf.select() by multiplying with zero instead of selecting: this requires use to use some ugly tricks to avoid potential NaNs
    # the 1e-12 in tf.maximum(cdf_delta, 1e-12) is never actually used as output, it's purely there to get around the tf.select() gradient issue
    # if the probability on a sub-pixel is below 1e-5, we use an approximation based on the assumption that the log-density is constant in the bin of the observed sub-pixel value
    if low_bit:
        log_probs = torch.where(x < -0.999,
                                log_cdf_plus,
                                torch.where(x > 0.999,
                                            log_one_minus_cdf_min,
                                            torch.where(cdf_delta > 1e-5,
                                                        torch.log(const_max(cdf_delta, 1e-12)),
                                                        log_pdf_mid - np.log(15.5))))
    else:
        log_probs = torch.where(x < -0.999,
                                log_cdf_plus,
                                torch.where(x > 0.999,
                                            log_one_minus_cdf_min,
                                            torch.where(cdf_delta > 1e-5,
                                                        torch.log(const_max(cdf_delta, 1e-12)),
                                                        log_pdf_mid - np.log(127.5))))
    log_probs = log_probs.sum(dim=3) + log_prob_from_logits(logit_probs)
    mixture_probs = torch.logsumexp(log_probs, -1)
    return -1. * mixture_probs.sum(dim=[1, 2]) / np.prod(xs[1:])


def sample_from_discretized_mix_logistic(l, nr_mix):
    # l : [B,H,W,100] #100 = nr_mix + 3 * 2 * nr_mix + 1* nr_mix + 2*nr_mix # shift, scale, means, scales, logit
    ls = [s for s in l.shape]
    xs = ls[:-1] + [3]
    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix] # [B, H, W, nr_mix]
    l = torch.reshape(l[:, :, :, nr_mix:], xs + [nr_mix * 3]) #[B, H, W, 3, 30] 
    # sample mixture indicator from softmax
    eps = torch.empty(logit_probs.shape, device=l.device).uniform_(1e-5, 1. - 1e-5)
    amax = torch.argmax(logit_probs - torch.log(-torch.log(eps)), dim=3)
    sel = F.one_hot(amax, num_classes=nr_mix).float()
    sel = torch.reshape(sel, xs[:-1] + [1, nr_mix])
    # select logistic parameters
    means = (l[:, :, :, :, :nr_mix] * sel).sum(dim=4)
    log_scales = const_max((l[:, :, :, :, nr_mix:nr_mix * 2] * sel).sum(dim=4), -7.)
    coeffs = (torch.tanh(l[:, :, :, :, nr_mix * 2:nr_mix * 3]) * sel).sum(dim=4)
    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    u = torch.empty(means.shape, device=means.device).uniform_(1e-5, 1. - 1e-5)
    x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1. - u))
    x0 = const_min(const_max(x[:, :, :, 0], -1.), 1.)
    x1 = const_min(const_max(x[:, :, :, 1] + coeffs[:, :, :, 0] * x0, -1.), 1.)
    x2 = const_min(const_max(x[:, :, :, 2] + coeffs[:, :, :, 1] * x0 + coeffs[:, :, :, 2] * x1, -1.), 1.)
    x = torch.cat([torch.reshape(x0, xs[:-1] + [1]), torch.reshape(x1, xs[:-1] + [1]), torch.reshape(x2, xs[:-1] + [1])], dim=3)
    return x.permute(0, 3, 1, 2)

def _get_dmol_stats(l, nr_mix):
    # return mean and variance of the dmol distribution
    ls = [s for s in l.shape]
    xs = ls[:-1] + [3]
    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix] # [B, H, W, nr_mix]
    log_probs_mix = log_prob_from_logits(logit_probs)
    probs_mix = torch.exp(log_probs_mix)
    probs_mix = torch.reshape(probs_mix, xs[:-1] + [1, nr_mix]) # [B, H, W, 1, nr_mix]
    l = torch.reshape(l[:, :, :, nr_mix:], xs + [nr_mix * 3]) #[B, H, W, 3, 9 * nr_mix] 
    mix_means_p = l[:, :, :, :, :nr_mix]# [B, H, W, 3, nr_mix]
    mix_means = torch.zeros_like(mix_means_p)
    mix_means[:, :, :, 0, :] =  mix_means_p[:, :, :, 0, :]
    mix_coeffs = (torch.tanh(l[:, :, :, :, nr_mix * 2:nr_mix * 3])) # [B, H, W, 3, nr_mix]
    mix_means[:, :, :, 1, :] = mix_means_p[:, :, :, 1, :] + mix_coeffs[:, :, :, 0, :] * mix_means_p[:, :, :, 0, :]
    mix_means[:, :, :, 2, :] = mix_means_p[:, :, :, 2, :] + mix_coeffs[:, :, :, 1, :] * mix_means_p[:, :, :, 0, :] + mix_coeffs[:, :, :, 2, :] * mix_means_p[:, :, :, 1, :]
    
    
    mix_means = const_min(const_max(mix_means, -1.), 1.)
    mean = (mix_means * probs_mix).sum(dim=4) #[B, H, W, 3] 
    mix_log_scales = const_max((l[:, :, :, :, nr_mix:nr_mix * 2]), -7.) #[B, H, W, 3, nr_mix]
    mix_var = (torch.pi * torch.exp(mix_log_scales))**2 / 3
    var = ((mix_var + mix_means**2)* probs_mix).sum(dim=4)
    var = var - mean**2
    mean = mean.permute(0, 3, 1, 2)
    var = var.permute(0, 3, 1, 2)
    return mean, var 
    


class HModule(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.H = H
        self.build()


class DmolNet(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.H = H
        self.width = H.width
        self.out_conv = get_conv(H.width, H.num_mixtures * 10, kernel_size=1, stride=1, padding=0)

    def nll(self, px_z, x):
        return discretized_mix_logistic_loss(x=x, l=self.forward(px_z), low_bit=self.H.dataset in ['ffhq_256'])

    def forward(self, px_z):
        xhat = self.out_conv(px_z)
        return xhat.permute(0, 2, 3, 1)

    def sample(self, px_z, quantize=True, detach=False):
        """
        quantize = True : return uint image [0,255]
        quantize = False : return float image [0, 1]
        """
        im = sample_from_discretized_mix_logistic(self.forward(px_z), self.H.num_mixtures)
        if quantize:    
            #xhat = xhat.detach().cpu().numpy()
            #xhat = np.minimum(np.maximum(0.0, xhat), 255.0).astype(np.uint8)
            xhat = (im + 1.0) * 127.5
            xhat = xhat.clamp(0, 255).type(torch.uint8)
        else:
            xhat = (im + 1) / 2 # -> [0, 1]
        if detach:
            xhat = xhat.detach()
        return xhat

    def get_dmol_stats(self, px_z):
        px_z = self.out_conv(px_z).permute(0, 2, 3, 1)
        mean_xz, var_xz = _get_dmol_stats(px_z, nr_mix=self.H.num_mixtures)
        # [-1, 1] -> [0, 1]
        mean_xz = (mean_xz + 1) / 2
        var_xz = var_xz / 4
        return mean_xz, var_xz

def info_tensor(t):
    N = len(t.flatten())
    print(t.shape)
    print('NaN : {}/{}'.format(torch.isnan(t).sum().item(), N))
    print('inf : {}/{}'.format(torch.isinf(t).sum().item(), N))
    print('min : {}'.format(t.min()))
    print('max : {}'.format(t.max()))
    print('mean : {}'.format(t.mean()))