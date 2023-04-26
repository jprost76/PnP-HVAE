import torch
from torch import nn
from torch.nn import functional as F
from collections import defaultdict
import numpy as np
import itertools

try:
    from VAEs.vdvae.vae_helpers import HModule, get_1x1, get_3x3, DmolNet, draw_gaussian_diag_samples, draw_gaussian_diag_samples_covariance, gaussian_analytical_kl, gaussian_ll, gaussian_product
except (ImportError, ValueError):
    from vae_helpers import HModule, get_1x1, get_3x3, DmolNet, draw_gaussian_diag_samples, draw_gaussian_diag_samples_covariance, gaussian_analytical_kl, gaussian_ll, gaussian_product

class Block(nn.Module):
    def __init__(self, in_width, middle_width, out_width, down_rate=None, residual=False, use_3x3=True, zero_last=False):
        super().__init__()
        self.down_rate = down_rate
        self.residual = residual
        self.c1 = get_1x1(in_width, middle_width)
        self.c2 = get_3x3(middle_width, middle_width) if use_3x3 else get_1x1(middle_width, middle_width)
        self.c3 = get_3x3(middle_width, middle_width) if use_3x3 else get_1x1(middle_width, middle_width)
        self.c4 = get_1x1(middle_width, out_width, zero_weights=zero_last)

    def forward(self, x):
        xhat = self.c1(F.gelu(x))
        xhat = self.c2(F.gelu(xhat))
        xhat = self.c3(F.gelu(xhat))
        xhat = self.c4(F.gelu(xhat))
        out = x + xhat if self.residual else xhat
        if self.down_rate is not None:
            out = F.avg_pool2d(out, kernel_size=self.down_rate, stride=self.down_rate)
        return out


def parse_layer_string(s):
    layers = []
    for ss in s.split(','):
        if 'x' in ss:
            res, num = ss.split('x')
            count = int(num)
            layers += [(int(res), None) for _ in range(count)]
        elif 'm' in ss:
            res, mixin = [int(a) for a in ss.split('m')]
            layers.append((res, mixin))
        elif 'd' in ss:
            res, down_rate = [int(a) for a in ss.split('d')]
            layers.append((res, down_rate))
        else:
            res = int(ss)
            layers.append((res, None))
    return layers


def pad_channels(t, width):
    d1, d2, d3, d4 = t.shape
    empty = torch.zeros(d1, width, d3, d4, device=t.device)
    empty[:, :d2, :, :] = t
    return empty


def get_width_settings(width, s):
    mapping = defaultdict(lambda: width)
    if s:
        s = s.split(',')
        for ss in s:
            k, v = ss.split(':')
            mapping[int(k)] = int(v)
    return mapping


class Encoder(HModule):
    def build(self):
        H = self.H
        self.in_conv = get_3x3(H.image_channels, H.width)
        self.widths = get_width_settings(H.width, H.custom_width_str)
        enc_blocks = []
        blockstr = parse_layer_string(H.enc_blocks)
        for res, down_rate in blockstr:
            use_3x3 = res > 2  # Don't use 3x3s for 1x1, 2x2 patches
            enc_blocks.append(Block(self.widths[res], int(self.widths[res] * H.bottleneck_multiple), self.widths[res], down_rate=down_rate, residual=True, use_3x3=use_3x3))
        n_blocks = len(blockstr)
        for b in enc_blocks:
            b.c4.weight.data *= np.sqrt(1 / n_blocks)
        self.enc_blocks = nn.ModuleList(enc_blocks)

    def forward(self, x):
        """
        input :
        x : tensor of shape (B,C,H,W)
        """
        x = self.in_conv(x)
        activations = {}
        activations[x.shape[2]] = x
        for block in self.enc_blocks:
            x = block(x)
            res = x.shape[2]
            x = x if x.shape[1] == self.widths[res] else pad_channels(x, self.widths[res])
            activations[res] = x
        return activations


class DecBlock(nn.Module):
    def __init__(self, H, res, mixin, n_blocks):
        super().__init__()
        self.base = res
        self.mixin = mixin
        self.H = H
        self.widths = get_width_settings(H.width, H.custom_width_str)
        width = self.widths[res]
        use_3x3 = res > 2
        cond_width = int(width * H.bottleneck_multiple)
        self.zdim = H.zdim
        self.enc = Block(width * 2, cond_width, H.zdim * 2, residual=False, use_3x3=use_3x3)
        self.prior = Block(width, cond_width, H.zdim * 2 + width, residual=False, use_3x3=use_3x3, zero_last=True)
        self.z_proj = get_1x1(H.zdim, width)
        self.z_proj.weight.data *= np.sqrt(1 / n_blocks)
        self.resnet = Block(width, cond_width, width, residual=True, use_3x3=use_3x3)
        self.resnet.c4.weight.data *= np.sqrt(1 / n_blocks)
        self.z_fn = lambda x: self.z_proj(x)

    def sample(self, x, acts, t=None):
        qm, qv = self.enc(torch.cat([x, acts], dim=1)).chunk(2, dim=1)
        feats = self.prior(x)
        pm, pv, xpp = feats[:, :self.zdim, ...], feats[:, self.zdim:self.zdim * 2, ...], feats[:, self.zdim * 2:, ...]
        x = x + xpp
        if t is not None:
            z = draw_gaussian_diag_samples(qm, qv + torch.ones_like(qv) * np.log(t))
        else:
            z = draw_gaussian_diag_samples(qm, qv)
        kl = gaussian_analytical_kl(qm, pm, qv, pv)
        ll = gaussian_ll(pm, pv, z)
        return z, x, kl, ll

    def sample_uncond(self, x, t=None, lvs=None, acts=None, compute_ll=False):
        """
        acts : activation from the bottom-up pass, must be specified to get the posterior distirbution
        """
        n, c, h, w = x.shape
        feats = self.prior(x)
        pm, pv, xpp = feats[:, :self.zdim, ...], feats[:, self.zdim:self.zdim * 2, ...], feats[:, self.zdim * 2:, ...]
        x = x + xpp
        if t is not None:
            pv = pv + torch.ones_like(pv) * np.log(t)
        if lvs is not None:
            z = lvs
        else:
            z = draw_gaussian_diag_samples(pm, pv)
        if compute_ll:
            ll = gaussian_ll(pm, pv, z)
        else:
            ll = None
        return z, x, ll

    def sample_reg(self, x, zx, acts, a, b, t_prior=1, t=None, mode='map'):
        # return arg min -a.log q(z|x) -b.log p(z)
        # qv : log(sigma)
        qm, qv = self.enc(torch.cat([x, acts], dim=1)).chunk(2, dim=1)
        feats = self.prior(x)
        pm, pv, xpp = feats[:, :self.zdim, ...], feats[:, self.zdim:self.zdim * 2, ...], feats[:, self.zdim * 2:, ...]
        pS = (torch.exp(pv) * t_prior) ** 2
        qS = torch.exp(qv) ** 2
        x = x + xpp
        if mode == 'map':
            #zreg = (a*qm*pS + b*pm*qS) / (a*pS + b*qS) 
            zreg= (a*qm/qS + b*pm/pS) / (a/qS + b/pS)
        elif mode == 'sample':
            m, S = gaussian_product(qm, qS/a, pm, pS/b) # S : covariance Cov
            if t is not None:
                zreg = draw_gaussian_diag_samples_covariance(m, t**2*S)
            else:
                # t=1
                zreg = draw_gaussian_diag_samples_covariance(m, S)
        else:
            raise ValueError(f"expecting mode to be either \'map\' or \'sample\', got {mode}")
        
        kl = gaussian_analytical_kl(qm, pm, qv, pv)
        ll = gaussian_ll(pm, pv, zreg)
        return zreg, x, kl, ll # return regularized latent log_likelyhood

    def sample_proj(self, x, zx, acts, p, t=None, mode='map'):# TODO: mode in 'sample' , 'map'
        # qv : log(sigma)
        qm, qv = self.enc(torch.cat([x, acts], dim=1)).chunk(2, dim=1)
        feats = self.prior(x)
        pm, pv, xpp = feats[:, :self.zdim, ...], feats[:, self.zdim:self.zdim * 2, ...], feats[:, self.zdim * 2:, ...]
        psigma = torch.exp(pv) 
        x = x + xpp
        if zx is None:
            if t is not None:
                zx = draw_gaussian_diag_samples(qm, qv + torch.ones_like(qv) * np.log(t))
            else:
                zx = draw_gaussian_diag_samples(qm, qv)
        # find w so that P(mu - w * sigma <= Z <= mu + w*sigma) = p
        normal = torch.distributions.Normal(0, 1)
        w = normal.icdf((torch.tensor(p) + 1) / 2)
        # projection
        zreg = torch.max(torch.min(zx, pm + w * psigma), pm - w * psigma)
        kl = gaussian_analytical_kl(qm, pm, qv, pv)
        ll = gaussian_ll(pm, pv, zreg)
        return zreg, x, kl, ll # return regularized latent log_likelyhood

    def get_inputs(self, xs, activations):
        acts = activations[self.base]
        try:
            x = xs[self.base]
        except KeyError:
            x = torch.zeros_like(acts)
        if acts.shape[0] != x.shape[0]:
            x = x.repeat(acts.shape[0], 1, 1, 1)
        return x, acts

    def forward(self, xs, activations, get_latents=False, t=None):
        x, acts = self.get_inputs(xs, activations)
        if self.mixin is not None:
            x = x + F.interpolate(xs[self.mixin][:, :x.shape[1], ...], scale_factor=self.base // self.mixin)
        z, x, kl, ll = self.sample(x, acts, t=t)
        x = x + self.z_fn(z)
        x = self.resnet(x)
        xs[self.base] = x
        stats = dict(kl=kl, ll=ll)
        if get_latents:
            #stats['z'] = z.detach() 
            stats['z'] = z
        return xs, stats

    def forward_uncond(self, xs, t=None, lvs=None, compute_ll=False):
        try:
            x = xs[self.base]
        except KeyError:
            ref = xs[list(xs.keys())[0]]
            x = torch.zeros(dtype=ref.dtype, size=(ref.shape[0], self.widths[self.base], self.base, self.base), device=ref.device)
        if self.mixin is not None:
            x = x + F.interpolate(xs[self.mixin][:, :x.shape[1], ...], scale_factor=self.base // self.mixin)
        z, x, ll = self.sample_uncond(x, t, lvs=lvs, compute_ll=compute_ll)
        x = x + self.z_fn(z)
        x = self.resnet(x)
        xs[self.base] = x
        return xs, ll

    def forward_uncond_get_latents(self, xs, t=None, lvs=None):
        try:
            x = xs[self.base]
        except KeyError:
            ref = xs[list(xs.keys())[0]]
            x = torch.zeros(dtype=ref.dtype, size=(ref.shape[0], self.widths[self.base], self.base, self.base), device=ref.device)
        if self.mixin is not None:
            x = x + F.interpolate(xs[self.mixin][:, :x.shape[1], ...], scale_factor=self.base // self.mixin)
        z, x = self.sample_uncond(x, t, lvs=lvs)
        x = x + self.z_fn(z)
        x = self.resnet(x)
        xs[self.base] = x
        return xs, z

    def forward_reg(self, xs, zx, activations, a, b, t_prior, get_latents=False, t=None, mode='map'):
        x, acts = self.get_inputs(xs, activations)
        if self.mixin is not None:
            x = x + F.interpolate(xs[self.mixin][:, :x.shape[1], ...], scale_factor=self.base // self.mixin)
        z, x, kl, ll = self.sample_reg(x, zx, acts, a, b, t_prior=t_prior, t=t, mode=mode)
        x = x + self.z_fn(z)
        x = self.resnet(x)
        xs[self.base] = x
        stats = dict(kl=kl, ll=ll)
        if get_latents:
            stats['z'] = z.detach() 
        return xs, stats

    def forward_proj(self, xs, zx, activations, p, get_latents=False, t=None):
        x, acts = self.get_inputs(xs, activations)
        if self.mixin is not None:
            x = x + F.interpolate(xs[self.mixin][:, :x.shape[1], ...], scale_factor=self.base // self.mixin)
        z, x, kl, ll = self.sample_proj(x, zx, acts, p, t=t)
        x = x + self.z_fn(z)
        x = self.resnet(x)
        xs[self.base] = x
        stats = dict(kl=kl, ll=ll)
        if get_latents:
            stats['z'] = z.detach() 
        return xs, stats




class Decoder(HModule):

    def build(self):
        H = self.H
        resos = set()
        dec_blocks = []
        self.widths = get_width_settings(H.width, H.custom_width_str)
        blocks = parse_layer_string(H.dec_blocks)
        for idx, (res, mixin) in enumerate(blocks):
            dec_blocks.append(DecBlock(H, res, mixin, n_blocks=len(blocks)))
            resos.add(res)
        self.resolutions = sorted(resos)
        self.dec_blocks = nn.ModuleList(dec_blocks)
        self.bias_xs = nn.ParameterList([nn.Parameter(torch.zeros(1, self.widths[res], res, res)) for res in self.resolutions if res <= H.no_bias_above])
        self.out_net = DmolNet(H)
        self.gain = nn.Parameter(torch.ones(1, H.width, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, H.width, 1, 1))
        self.final_fn = lambda x: x * self.gain + self.bias

    def forward(self, activations, get_latents=False):
        stats = []
        xs = {a.shape[2]: a for a in self.bias_xs}
        for block in self.dec_blocks:
            xs, block_stats = block(xs, activations, get_latents=get_latents)
            stats.append(block_stats)
        xs[self.H.image_size] = self.final_fn(xs[self.H.image_size])
        return xs[self.H.image_size], stats

    def forward_get_latents(self, activations, nmax=None, t=None):
        """
        nmax : number of latents to get (if nmax == None, get all the latents)
        """
        nmax = nmax if nmax else len(self.dec_blocks)
        stats = []
        xs = {a.shape[2]: a for a in self.bias_xs}
        for idx, block in enumerate(self.dec_blocks[:nmax]):
            try:
                temp = t[idx]
            except TypeError:
                temp = t
            xs, block_stats = block(xs, activations, get_latents=True, t=temp)
            stats.append(block_stats)
        return stats


    def forward_uncond(self, n, t=None):
        xs = {}
        for bias in self.bias_xs:
            xs[bias.shape[2]] = bias.repeat(n, 1, 1, 1)
        for idx, block in enumerate(self.dec_blocks):
            try:
                temp = t[idx]
            except TypeError:
                temp = t
            xs, ll = block.forward_uncond(xs, temp)
        xs[self.H.image_size] = self.final_fn(xs[self.H.image_size])
        return xs[self.H.image_size]

    def forward_uncond_get_latents(self, n, t=None):
        xs = {}
        latents = []
        for bias in self.bias_xs:
            xs[bias.shape[2]] = bias.repeat(n, 1, 1, 1)
        for idx, block in enumerate(self.dec_blocks):
            try:
                temp = t[idx]
            except TypeError:
                temp = t
            xs, z = block.forward_uncond_get_latents(xs, temp)
            latents.append(z)
        xs[self.H.image_size] = self.final_fn(xs[self.H.image_size])
        return xs[self.H.image_size], latents

    def forward_manual_latents(self, n, latents, t=None, compute_ll=False):
        """
        return :
        px_z
        post : [{'mean' : , 'cov' : }] 
        """
        xs = {}
        LL = []
        for bias in self.bias_xs:
            xs[bias.shape[2]] = bias.repeat(n, 1, 1, 1)
        idx = 0
        for block, lvs in itertools.zip_longest(self.dec_blocks, latents):
            try:
                temp = t[idx]
            except TypeError:
                temp = t
            xs, ll = block.forward_uncond(xs, temp, lvs=lvs, compute_ll=compute_ll)
            LL.append(ll)
            idx += 1
        xs[self.H.image_size] = self.final_fn(xs[self.H.image_size])
        return xs[self.H.image_size], LL

    def forward_reg(self, activations, latents_x, beta, T, get_latents=False, t=None, nmax=None, mode='map'):
        nmax = len(self.dec_blocks) if nmax is None else nmax
        stats = []
        xs = {a.shape[2]: a for a in self.bias_xs}
        for idx, (block, zl) in enumerate(zip(self.dec_blocks, latents_x)):
            try:
                temp = t[idx]
            except TypeError:
                temp = t
            if idx < nmax:
                xs, block_stats = block.forward_reg(xs, zl, activations, a=beta, b=1-beta*T[idx]**2, t_prior=T[idx], get_latents=get_latents, t=temp, mode=mode)
                stats.append(block_stats)
            else:
                #xs, block_stats = block.forward_uncond(xs, temp)
                xs, block_stats = block.forward_uncond(xs, 1) # TODO: control temp
        xs[self.H.image_size] = self.final_fn(xs[self.H.image_size])
        return xs[self.H.image_size], stats

    def forward_proj(self, activations, latents_x, p, get_latents=False, t=None, nmax=None):
        nmax = len(self.dec_blocks) if nmax is None else nmax
        stats = []
        xs = {a.shape[2]: a for a in self.bias_xs}
        for idx, (block, zl) in enumerate(zip(self.dec_blocks, latents_x)):
            try:
                temp = t[idx]
            except TypeError:
                temp = t
            if idx < nmax:
                xs, block_stats = block.forward_proj(xs, zl, activations, p, get_latents=get_latents, t=temp)
                stats.append(block_stats)
            else:
                xs = block.forward_uncond(xs, temp)
        xs[self.H.image_size] = self.final_fn(xs[self.H.image_size])
        return xs[self.H.image_size], stats


class VAE(HModule):
    def build(self):
        self.encoder = Encoder(self.H)
        self.decoder = Decoder(self.H)
        self.register_buffer('shift', torch.tensor([self.H.shift]).view(1, 1, 1))
        self.register_buffer('scale', torch.tensor([self.H.scale]).view(1, 1, 1))
    
    def normalize_input(self, x):
        #x.add_(self.shift).mul_(self.scale)
        x = (x + self.shift) * self.scale
        return x

    def forward(self, x, x_target):
        x = self.normalize_input(x)
        activations = self.encoder.forward(x)
        px_z, stats = self.decoder.forward(activations)
        distortion_per_pixel = self.decoder.out_net.nll(px_z, x_target)
        rate_per_pixel = torch.zeros_like(distortion_per_pixel)
        ndims = np.prod(x.shape[1:])
        for statdict in stats:
            rate_per_pixel += statdict['kl'].sum(dim=(1, 2, 3))
        rate_per_pixel /= ndims
        elbo = (distortion_per_pixel + rate_per_pixel).mean()
        return dict(elbo=elbo, distortion=distortion_per_pixel.mean(), rate=rate_per_pixel.mean())

    def forward_get_latents(self, x, nmax=None, t=None):
        """
        input :
        x : image tensor
        get_latents (bool) : return latent codes zi
        nmax (int): number of latents to get (if nmax==None, get all the latents)
        return :
        stats (list) : list of stats for each level [{'kl' : , 'z' : ' ('mean' : , 'cov' : )}]
        """
        x = self.normalize_input(x)
        activations = self.encoder.forward(x)
        stats = self.decoder.forward_get_latents(activations, nmax=nmax, t=t)
        return stats

    def forward_uncond_samples(self, n_batch, t=None):
        x = self.normalize_input(x)
        px_z = self.decoder.forward_uncond(n_batch, t=t)
        return self.decoder.out_net.sample(px_z)

    def forward_uncond_samples_get_latents(self, n_batch, t=None):
        px_z, latents = self.decoder.forward_uncond_get_latents(n_batch, t=t)
        samples = self.decoder.out_net.sample(px_z)
        return samples, latents
        
    def forward_samples_set_latents(self, n_batch, latents, t=None, compute_ll=False, detach=True):
        px_z, ll = self.decoder.forward_manual_latents(n_batch, latents, t=t, compute_ll=True)
        return self.decoder.out_net.sample(px_z, detach=detach), ll

    def forward_with_latent_reg(self, x, beta, T, get_latents=False, presample_z=False, t=None, nmax=None, quantize=False, mode='sample'):
        """_summary_
        compute p(x|z), where z = (z_l)_l is computed as (or sampled from)
        z_l = \arg\min_{u_l} -beta \log q(u_l|z<l, x) - (1/T[l]**2 - beta) \log p(u_l|z<l) for 0<=l<L
        """
        x = self.normalize_input(x)
        activations = self.encoder.forward(x)
        if presample_z:
            stats = self.decoder.forward_get_latents(activations)
            latents_x = [d['z'] for d in stats]
        else:
            latents_x = [None for _ in self.decoder.dec_blocks]
        px_z, stats = self.decoder.forward_reg(activations, latents_x, beta, T, get_latents=get_latents, t=t, nmax=nmax, mode=mode)
        mean_xz, var_xz = self.decoder.out_net.get_dmol_stats(px_z)
        samples = self.decoder.out_net.sample(px_z, quantize=quantize)
        return samples, stats, mean_xz, var_xz

    def forward_with_latent_proj(self, x, p, get_latents=False, presample_z=False, t=None, nmax=None):
        x = self.normalize_input(x)
        activations = self.encoder.forward(x)
        if presample_z:
            stats = self.decoder.forward_get_latents(activations)
            latents_x = [d['z'] for d in stats]
        else:
            latents_x = [None for _ in self.decoder.dec_blocks]
        px_z, stats = self.decoder.forward_proj(activations, latents_x, p, get_latents=get_latents, t=t, nmax=nmax)
        samples = self.decoder.out_net.sample(px_z)
        return samples, stats

    def eval_logpxz(self, x, latents, T):
        # x should be in range [0, 1]!
        px_z, log_pz = self.decoder.forward_manual_latents(1, latents, t=T, compute_ll=True)
        # log p(x|z)
        mean_xz, var_xz = self.decoder.out_net.get_dmol_stats(px_z)
        # assume a gaussian decoder
        log_pxz = - 0.5 * torch.sum(np.log(2 * np.pi) + torch.log(var_xz) + (x-mean_xz)**2 / var_xz) #TODO: compute in utils_vae.py
        return log_pz, log_pxz, mean_xz, var_xz
