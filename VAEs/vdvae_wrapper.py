import torch
import torch.nn as nn
import yaml
from .base_wrapper import BaseVAE
from .utils_vae import gaussian_log_likelihood


try:
    from .vdvae.vae import VAE as VDVAE
except (ImportError, ValueError):
    from VAEs.vdvae.vae import VAE as VDVAE


class Hyperparams(dict):
    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            return None

    def __setattr__(self, attr, value):
        self[attr] = value

def load_vdvae(conf_path, map_location=None):
    H = Hyperparams()
    with open(conf_path) as file:
        H.update(yaml.safe_load(file))
    vae = VDVAE(H)
    state_dict_path = H['checkpoint_path']
    if map_location:
        state_dict = torch.load(state_dict_path, map_location=map_location)
    else:
        state_dict = torch.load(state_dict_path)
    new_state_dict = vae.state_dict()
    new_state_dict.update(state_dict)
    vae.load_state_dict(new_state_dict)
    vae = vae.cuda()
    receptive_field = 256
    return vae, receptive_field

class VDVAEWrapper(BaseVAE):
    def __init__(self):
        super().__init__()
        vae, rf = load_vdvae(conf_path='VAEs/vdvae/saved_models/vdvae_ffhq256.yaml') # TODO: control conf path with hydra?
        self.vae = vae
        self.receptive_field = rf

    def eval_logpxz(self, x, z, T, decoder_std):
        xtarget = x
        log_pzl, log_pxz, mu_xz, var_xz = self.vae.eval_logpxz(xtarget, z, T)
        log_pz = sum(log_pzl)
        if decoder_std is not None:
            var_xz = torch.ones_like(mu_xz) * decoder_std**2
            #recompute log p(x|z) assuming it is gaussian with diagonal constant variance
            log_pxz = gaussian_log_likelihood(x, mu_xz, var_xz)
        return log_pz, log_pzl, log_pxz, mu_xz, var_xz

    def latent_reg(self, x, beta, T, mode, sample_from_prior_after=None, dec_std=None):
        # input x should be in range [0, 1]
        with torch.no_grad():
            x = torch.clamp(x*256, 0, 255)
            x, stats, mu_xz, var_xz = self.vae.forward_with_latent_reg(x, beta=beta, T=T, quantize=False, nmax=sample_from_prior_after, mode=mode) 
            x += 4/255 # correct bias due 5 bit quantization bug
            mu_xz += 4/255
        x = x.clamp(0, 1)
        mu_xz = mu_xz.clamp(0,1)
        var_xz = var_xz.clamp(0.004, 0.5)
        if dec_std is not None:
            var_xz = torch.ones_like(mu_xz) * dec_std**2
        log_pz = [block_stats['ll'] for block_stats in stats]
        gx = -1 * sum(log_pz)
        return x, gx, log_pz, mu_xz, var_xz

    def encode(self, x):
        # input x should be in range [0, 1]
        with torch.no_grad():
            x = torch.clamp(x*256, 0, 255)
            stats = self.vae.forward_get_latents(x)
            latents = [d['z'] for d in stats]
        return latents