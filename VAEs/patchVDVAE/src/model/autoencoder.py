import torch
import torch.nn as nn
import numpy as np
from hparams import HParams

try:
    from .layers import LevelBlockUp, LevelBlockDown, ResidualConvCell
    from ..utils.utils import one_hot, compute_latent_dimension, scale_pixels
    from .conv2d import Conv2d
    from .losses import get_dmol_mean_var_from_logits
except (ImportError, ValueError, ModuleNotFoundError):
    from model.layers import LevelBlockUp, LevelBlockDown, ResidualConvCell
    from utils.utils import one_hot, get_same_padding, compute_latent_dimension
    from model.conv2d import Conv2d
    from utils.utils import scale_pixels
    from losses import get_dmol_mean_var_from_logits

hparams = HParams.get_hparams_by_name("patch_vdvae")


class BottomUp(torch.nn.Module):
    def __init__(self):
        super(BottomUp, self).__init__()

        in_channels_up = [hparams.model.input_conv_filters] + hparams.model.up_filters[0:-1]

        self.levels_up = nn.ModuleList([])
        self.levels_up_downsample = nn.ModuleList([])

        for i, stride in enumerate(hparams.model.up_strides):
            elements = nn.ModuleList([])
            for j in range(hparams.model.up_n_blocks_per_res[i]):
                elements.extend([LevelBlockUp(n_blocks=hparams.model.up_n_blocks[i],
                                              n_layers=hparams.model.up_n_layers[i],
                                              in_filters=in_channels_up[i],
                                              filters=hparams.model.up_filters[i],
                                              bottleneck_ratio=hparams.model.up_mid_filters_ratio[i],
                                              kernel_size=hparams.model.up_kernel_size[i],
                                              strides=1,
                                              skip_filters=hparams.model.up_skip_filters[i],
                                              use_skip=False)])

            self.levels_up.extend([elements])

            self.levels_up_downsample.extend([LevelBlockUp(n_blocks=hparams.model.up_n_blocks[i],
                                                           n_layers=hparams.model.up_n_layers[i],
                                                           in_filters=in_channels_up[i],
                                                           filters=hparams.model.up_filters[i],
                                                           bottleneck_ratio=hparams.model.up_mid_filters_ratio[i],
                                                           kernel_size=hparams.model.up_kernel_size[i],
                                                           strides=stride,
                                                           skip_filters=hparams.model.up_skip_filters[i],
                                                           use_skip=True)])

        self.input_conv = Conv2d(in_channels=hparams.data.channels,
                                 out_channels=hparams.model.input_conv_filters,
                                 kernel_size=hparams.model.input_kernel_size,
                                 stride=(1, 1),
                                 padding='same')

    def forward(self, x):
        x = self.input_conv(x)

        skip_list = []

        for i, (level_up, level_up_downsample) in enumerate(zip(self.levels_up, self.levels_up_downsample)):
            for layer in level_up:
                x, _ = layer(x)

            x, skip_out = level_up_downsample(x)
            skip_list.append(skip_out)

        skip_list = skip_list[::-1]
        return skip_list


class TopDown(torch.nn.Module):
    def __init__(self):
        super(TopDown, self).__init__()
        self.min_pix_value = scale_pixels(0.)
        self.max_pix_value = scale_pixels(255.)


        in_channels_down = [hparams.model.down_filters[0]] + hparams.model.down_filters[0:-1]
        self.levels_down, self.levels_down_upsample = nn.ModuleList([]), nn.ModuleList([])

        for i, stride in enumerate(hparams.model.down_strides):
            self.levels_down_upsample.extend([LevelBlockDown(
                n_blocks=hparams.model.down_n_blocks[i],
                n_layers=hparams.model.down_n_layers[i],
                in_filters=in_channels_down[i], # TODO: 0 if i==0?
                filters=hparams.model.down_filters[i],
                bottleneck_ratio=hparams.model.down_mid_filters_ratio[i],
                kernel_size=hparams.model.down_kernel_size[i],
                strides=stride,
                skip_filters=hparams.model.up_skip_filters[::-1][i],
                latent_variates=hparams.model.down_latent_variates[i],
                first_block=i == 0,
                last_block=False
            )])

            self.levels_down.extend([nn.ModuleList(
                [LevelBlockDown(n_blocks=hparams.model.down_n_blocks[i],
                                n_layers=hparams.model.down_n_layers[i],
                                in_filters=hparams.model.down_filters[i],
                                filters=hparams.model.down_filters[i],
                                bottleneck_ratio=hparams.model.down_mid_filters_ratio[i],
                                kernel_size=hparams.model.down_kernel_size[i],
                                strides=1,
                                skip_filters=hparams.model.up_skip_filters[::-1][i],
                                latent_variates=hparams.model.down_latent_variates[i],
                                first_block=False,
                                last_block=i == len(hparams.model.down_strides) - 1 and j ==
                                           hparams.model.down_n_blocks_per_res[i] - 1)
                 for j in range(hparams.model.down_n_blocks_per_res[i])])])

        if hparams.data.dataset_source == 'binarized_mnist':
            output_channels = 1
        elif hparams.model.decoder == 'dmol':
            output_channels = hparams.model.num_output_mixtures * (3 * hparams.data.channels + 1)
        elif hparams.model.decoder == 'gaussian':
            output_channels = 2*hparams.data.channels
        else:
            raise ValueError(f'distribution base {hparams.model.decoder} not known!!')

        self.output_conv = Conv2d(in_channels=hparams.model.down_filters[-1],
                                  out_channels=output_channels,
                                  kernel_size=hparams.model.output_kernel_size,
                                  stride=(1, 1), padding='same')

    def sample(self, logits):
        if hparams.data.dataset_source == 'binarized_mnist':
            return self._sample_from_bernoulli(logits)
        elif hparams.model.decoder == 'dmol':
            return self._sample_from_mol(logits) 
        else: # gaussian decoder
            return self._sample_from_gaussian(logits)

    def _sample_from_bernoulli(self, logits):
        logits = logits[:, :, 2:30, 2:30]
        probs = torch.sigmoid(logits)
        return torch.Tensor(logits.size()).bernoulli_(probs)  # B, C, H, W

    def _compute_scales(self, logits):
        softplus = nn.Softplus(beta=hparams.model.output_gradient_smoothing_beta)
        if hparams.model.output_distribution_base == 'std':
            scales = torch.maximum(softplus(logits), torch.as_tensor(np.exp(hparams.loss.min_mol_logscale)))

        elif hparams.model.output_distribution_base == 'logstd':
            log_scales = torch.maximum(logits, torch.as_tensor(np.array(hparams.loss.min_mol_logscale)))
            scales = torch.exp(hparams.model.output_gradient_smoothing_beta * log_scales)

        else:
            raise ValueError(f'distribution base {hparams.model.output_distribution_base} not known!!')

        return scales

    def _sample_from_mol(self, logits):
        B, _, H, W = logits.size()  # B, M*(3*C+1), H, W,

        logit_probs = logits[:, :hparams.model.num_output_mixtures, :, :]  # B, M, H, W
        l = logits[:, hparams.model.num_output_mixtures:, :, :]  # B, M*C*3 ,H, W
        l = l.reshape(B, hparams.data.channels, 3 * hparams.model.num_output_mixtures, H, W)  # B, C, 3 * M, H, W

        model_means = l[:, :, :hparams.model.num_output_mixtures, :, :]  # B, C, M, H, W
        scales = self._compute_scales(
            l[:, :, hparams.model.num_output_mixtures: 2 * hparams.model.num_output_mixtures, :, :])  # B, C, M, H, W
        model_coeffs = torch.tanh(
            l[:, :, 2 * hparams.model.num_output_mixtures: 3 * hparams.model.num_output_mixtures, :,
            :])  # B, C, M, H, W

        # Gumbel-max to select the mixture component to use (per pixel)
        gumbel_noise = -torch.log(-torch.log(
            torch.Tensor(logit_probs.size()).uniform_(1e-5, 1. - 1e-5).cuda()))  # B, M, H, W
        logit_probs = logit_probs / hparams.synthesis.output_temperature + gumbel_noise
        lambda_ = one_hot(torch.argmax(logit_probs, dim=1), logit_probs.size()[1], dim=1)  # B, M, H, W

        lambda_ = lambda_.unsqueeze(1)  # B, 1, M, H, W

        # select logistic parameters
        means = torch.sum(model_means * lambda_, dim=2)  # B, C, H, W
        scales = torch.sum(scales * lambda_, dim=2)  # B, C, H, W
        coeffs = torch.sum(model_coeffs * lambda_, dim=2)  # B, C,  H, W

        # Samples from logistic & clip to interval
        # we don't actually round to the nearest 8bit value when sampling
        u = torch.Tensor(means.size()).uniform_(1e-5, 1. - 1e-5).cuda()
        x = means + scales * hparams.synthesis.output_temperature * (
                torch.log(u) - torch.log(1. - u))  # B, C,  H, W

        # Autoregressively predict RGB
        x0 = torch.clamp(x[:, 0:1, :, :], min=self.min_pix_value, max=self.max_pix_value)  # B, 1, H, W
        x1 = torch.clamp(x[:, 1:2, :, :] + coeffs[:, 0:1, :, :] * x0, min=self.min_pix_value,
                         max=self.max_pix_value)  # B, 1, H, W
        x2 = torch.clamp(x[:, 2:3, :, :] + coeffs[:, 1:2, :, :] * x0 + coeffs[:, 2:3, :, :] * x1,
                         min=self.min_pix_value,
                         max=self.max_pix_value)  # B, 1, H, W

        x = torch.cat([x0, x1, x2], dim=1)  # B, C, H, W
        return x

    def _sample_from_gaussian(self, decoder_stats):
        mean, std = torch.chunk(decoder_stats, chunks=2, dim=1)
        std = std * hparams.synthesis.output_temperature
        x = mean + std * torch.randn_like(mean)
        return x

    def _post_process_output(self, y):
        # for gaussian decoder only
        if hparams.model.decoder == 'gaussian':
            if hparams.model.distribution_base == 'std':
            # apply softmax to std to get positive std
                mean, std = torch.chunk(y, chunks=2, dim=1)
                std = torch.nn.functional.softplus(std, beta=hparams.model.gradient_smoothing_beta)
            elif hparams.model.distribution_base == 'logstd':
                mean, logstd = torch.chunk(y, chunks=2, dim=1)
                std = torch.exp(hparams.model.gradient_smoothing_beta * logstd)
            else:
                raise ValueError(f'distribution base {hparams.model.distribution_base} not known!!')
            y = torch.cat((mean, std), dim=1)
        return y

    def forward(self, skip_list, variate_masks, sample_from_prior_after=None):
        if sample_from_prior_after is None:
            sample_from_prior_after = len (self.levels_down_upsample) + sum([len(level) for level in self.levels_down])

        # input size
        #B, _, H, W = skip_list[0].size()
        #y = torch.tile(self.trainable_h, (skip_list[0].size()[0], 1, 1, 1))
        # fully convolutionnal -> adapt the constant input to the size of the encoded image
        #y = torch.tile(self.trainable_h, (B, 1, H, W)) 
        y = None # TODO: if constant_input ...
        posterior_dist_list = []
        prior_kl_dist_list = []

        layer_idx = 0
        for i, (level_down_upsample, level_down, skip_input) in enumerate(
                zip(self.levels_down_upsample, self.levels_down, skip_list)):
            #print("i :", i, " skip : ", skip_input.shape, ", y: ", y.shape if y is not None else "")
            if layer_idx < sample_from_prior_after:
                y, posterior_dist, prior_kl_dist, = level_down_upsample(skip_input, y,
                                                                    variate_mask=variate_masks[layer_idx])
            else:
                y, _, _ = level_down_upsample.sample_from_prior(y, temperature=0.7)# TODO: control temperature
                posterior_dist = None
                prior_kl_dist = None
            layer_idx += 1

            resolution_posterior_dist = [posterior_dist]
            resolution_prior_kl_dist = [prior_kl_dist]

            for j, layer in enumerate(level_down):
                #print("j :", j, " skip : ", skip_input.shape, ", y: ", y.shape)
                if layer_idx < sample_from_prior_after:
                    y, posterior_dist, prior_kl_dist = layer(skip_input, y, variate_mask=variate_masks[layer_idx])
                else:
                    y, _, _ = layer.sample_from_prior(y, temperature=0.1)
                    posterior_dist = None
                    prior_kl_dist = None
                layer_idx += 1

                resolution_posterior_dist.append(posterior_dist)
                resolution_prior_kl_dist.append(prior_kl_dist)

            posterior_dist_list += resolution_posterior_dist # TODO?
            prior_kl_dist_list += resolution_prior_kl_dist

        y = self.output_conv(y)
        y = self._post_process_output(y)

        return y, posterior_dist_list, prior_kl_dist_list,

    def forward_with_latent_reg(self, skip_list, beta, T, variate_masks=None, sample_from_prior_after=None):
        if sample_from_prior_after is None:
            sample_from_prior_after = len (self.levels_down_upsample) + sum([len(level) for level in self.levels_down])
    #return z = arg max lq . log q(z|x) + lp log p(z)
        B, _, H, W = skip_list[0].size()
        #y = torch.tile(self.trainable_h, (B, 1, H, W)) 
        y = None # TODO: if constant input
        posterior_dist_list = []
        prior_kl_dist_list = []
        log_pzk_list = []

        layer_idx = 0
        for i, (level_down_upsample, level_down, skip_input) in enumerate(
                zip(self.levels_down_upsample, self.levels_down, skip_list)):
            if layer_idx < sample_from_prior_after:
                lq = beta
                lp = 1/T[layer_idx]**2 - beta
                y, posterior_dist, prior_kl_dist, log_pzk = level_down_upsample.forward_with_latent_reg(skip_input, y,
                                                                     lq, lp, t=T[layer_idx], variate_mask=variate_masks[layer_idx])
            else:
                y, _, log_pzk = level_down_upsample.sample_from_prior(y, temperature=T[layer_idx])
                posterior_dist = None
                prior_kl_dist = None
            layer_idx += 1

            resolution_posterior_dist = [posterior_dist]
            resolution_prior_kl_dist = [prior_kl_dist]
            resolution_log_pzk = [log_pzk]

            for j, layer in enumerate(level_down):
                if layer_idx < sample_from_prior_after:
                    lq = beta
                    lp = 1/T[layer_idx]**2 - beta
                    y, posterior_dist, prior_kl_dist, log_pzk = layer.forward_with_latent_reg(skip_input, y, lq, lp, t=T[layer_idx], variate_mask=variate_masks[layer_idx])
                else:
                    y, _, log_pzk = layer.sample_from_prior(y, temperature=T[layer_idx])
                    posterior_dist = None
                    prior_kl_dist = None
                layer_idx += 1

                resolution_posterior_dist.append(posterior_dist)
                resolution_prior_kl_dist.append(prior_kl_dist)
                resolution_log_pzk.append(log_pzk)

            posterior_dist_list += resolution_posterior_dist
            prior_kl_dist_list += resolution_prior_kl_dist
            log_pzk_list += resolution_log_pzk

        y = self.output_conv(y)
        y = self._post_process_output(y)

        return y, posterior_dist_list, prior_kl_dist_list, log_pzk_list
    
    def _compute_input_size(self, size):
        # compute the size of the constance input according to the output size
        rf = hparams.model.receptive_field
        if size is None:
            h = 1
            w = 1
        else:
            h_out, w_out = size
            if (h_out % rf !=0) or ((w_out % rf) != 0):
                raise ValueError(f'incorrect output shape, output shape must be a multiple of {rf}')
            h = h_out // rf
            w = w_out // rf
        return h, w

    def sample_from_prior(self, batch_size, temperatures, output_size=None):
        h_in, w_in = self._compute_input_size(output_size)
        with torch.no_grad():
            #y = torch.tile(self.trainable_h, (batch_size, 1, h_in, w_in))
            y = torch.empty(batch_size, 1, h_in, w_in) #TODO: if trainable_h
            prior_zs = []
            for i, (level_down_upsample, level_down, temperature) in enumerate(
                    zip(self.levels_down_upsample, self.levels_down, temperatures)):
                y, z, _ = level_down_upsample.sample_from_prior(y, temperature=temperature)

                level_z = [z]
                for _, layer in enumerate(level_down):
                    y, z, _ = layer.sample_from_prior(y, temperature=temperature)
                    level_z.append(z)

                prior_zs += level_z  # n_layers * [batch_size,  n_variates H, W]
            y = self.output_conv(y)
            y = self._post_process_output(y)

        return y, prior_zs

    def get_mol_mean_and_var(self, logits):
        if hparams.data.dataset_source == 'binarized_mnist':
            raise NotImplementedError
        else:
            mean_xz, var_xz = get_dmol_mean_var_from_logits(logits)
            return mean_xz, var_xz

    def forward_get_latents(self, skip_list, variate_masks, sample_from_prior_after=None):
        if sample_from_prior_after is None:
            sample_from_prior_after = len (self.levels_down_upsample) + sum([len(level) for level in self.levels_down])

        y = None
        posterior_dist_list = []
        prior_kl_dist_list = []
        latents_list = []

        layer_idx = 0
        for i, (level_down_upsample, level_down, skip_input) in enumerate(
                zip(self.levels_down_upsample, self.levels_down, skip_list)):
            if layer_idx < sample_from_prior_after:
                y, posterior_dist, prior_kl_dist, z = level_down_upsample.forward_get_latents(skip_input, y,
                                                                    variate_mask=variate_masks[layer_idx])
            else:
                y, z, _ = level_down_upsample.sample_from_prior(y, temperature=0.7)# TODO: control temperature
                posterior_dist = None
                prior_kl_dist = None
            layer_idx += 1

            resolution_posterior_dist = [posterior_dist]
            resolution_prior_kl_dist = [prior_kl_dist]
            resolution_latents = [z]

            for j, layer in enumerate(level_down):
                if layer_idx < sample_from_prior_after:
                    y, posterior_dist, prior_kl_dist, z = layer.forward_get_latents(skip_input, y, variate_mask=variate_masks[layer_idx])
                else:
                    y, z, _ = layer.sample_from_prior(y, temperature=0.1)
                    posterior_dist = None
                    prior_kl_dist = None
                layer_idx += 1

                resolution_posterior_dist.append(posterior_dist)
                resolution_prior_kl_dist.append(prior_kl_dist)
                resolution_latents.append(z)

            posterior_dist_list += resolution_posterior_dist # TODO?
            prior_kl_dist_list += resolution_prior_kl_dist
            latents_list += resolution_latents

        y = self.output_conv(y)

        return y, posterior_dist_list, prior_kl_dist_list, latents_list

    def forward_manual_latents(self, latents, T, sample_from_prior_after=None):
        if sample_from_prior_after is None:
            sample_from_prior_after = len (self.levels_down_upsample) + sum([len(level) for level in self.levels_down])

        y = None
        log_pzk_list = []

        layer_idx = 0
        for i, (level_down_upsample, level_down) in enumerate(
                zip(self.levels_down_upsample, self.levels_down)):
            if layer_idx < sample_from_prior_after:
                y, log_pzk = level_down_upsample.forward_manual_latents(y,z=latents[layer_idx], T=T[layer_idx])
                log_pzk_list.append(log_pzk)
            else:
                y, _, log_pzk  = level_down_upsample.sample_from_prior(y, temperature=0.1)# TODO: control temperature
                log_pzk_list.append(log_pzk)
            layer_idx += 1

            for j, layer in enumerate(level_down):
                if layer_idx < sample_from_prior_after:
                    y, log_pzk = layer.forward_manual_latents(y, z=latents[layer_idx], T=T[layer_idx])
                    log_pzk_list.append(log_pzk)
                else:
                    y, _, log_pzk = layer.sample_from_prior(y, temperature=0.1)

                    log_pzk_list.append(log_pzk)
                layer_idx += 1

        y = self.output_conv(y)

        return y, log_pzk_list


