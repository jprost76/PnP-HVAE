from torch import nn
from hparams import HParams

#try:
#    from .autoencoder import TopDown, BottomUp
#except (ImportError, ValueError, ModuleNotFoundError):
#    from model.autoencoder import TopDown, BottomUp

from VAEs.patchVDVAE.src.model.autoencoder import TopDown, BottomUp

hparams = HParams.get_hparams_by_name("patch_vdvae")


class UniversalAutoEncoder(nn.Module):
    def __init__(self):
        super(UniversalAutoEncoder, self).__init__()

        self.bottom_up = BottomUp()
        self.top_down = TopDown()

    def forward(self, x, variate_masks=None, sample_from_prior_after=None):
        """
        x: (batch_size, time, H, W, C). In train, this is the shifted version of the target
        In slow synthesis, it would be the concatenated previous outputs
        """
        if variate_masks is None:
            variate_masks = [None] * (sum(hparams.model.down_n_blocks_per_res) + len(hparams.model.down_strides))
        else:
            assert len(variate_masks) == sum(hparams.model.down_n_blocks_per_res) + len(hparams.model.down_strides)

        skip_list = self.bottom_up(x)
        outputs, posterior_dist_list, prior_kl_dist_list = self.top_down(skip_list, variate_masks=variate_masks, sample_from_prior_after=sample_from_prior_after)

        return outputs, posterior_dist_list, prior_kl_dist_list

    def forward_get_latents(self, x, variate_masks=None):
        """
        x: (batch_size, time, H, W, C). In train, this is the shifted version of the target
        In slow synthesis, it would be the concatenated previous outputs
        """
        if variate_masks is None:
            variate_masks = [None] * (sum(hparams.model.down_n_blocks_per_res) + len(hparams.model.down_strides))
        else:
            assert len(variate_masks) == sum(hparams.model.down_n_blocks_per_res) + len(hparams.model.down_strides)

        skip_list = self.bottom_up(x)
        outputs, posterior_dist_list, prior_kl_dist_list, latent_list = self.top_down.forward_get_latents(skip_list, variate_masks=variate_masks)

        return outputs, posterior_dist_list, prior_kl_dist_list, latent_list
        
    def forward_with_latent_reg(self, x, beta, T, mode, variate_masks=None, sample_from_prior_after=None):
        """reconstruct x while regularizing the latent code toward the prior:
        z_l = \arg\min_{u_l} -beta \log q(u_l|z<l, x) - (1/T[l]**2 - beta) \log p(u_l|z<l) 
        """
        if variate_masks is None:
            variate_masks = [None] * (sum(hparams.model.down_n_blocks_per_res) + len(hparams.model.down_strides))
        else:
            assert len(variate_masks) == sum(hparams.model.down_n_blocks_per_res) + len(hparams.model.down_strides)

        skip_list = self.bottom_up(x)
        outputs, posterior_dist_list, prior_kl_dist_list, log_pzk_list = self.top_down.forward_with_latent_reg(skip_list, beta, T, mode=mode, variate_masks=variate_masks, sample_from_prior_after=sample_from_prior_after)

        return outputs, posterior_dist_list, prior_kl_dist_list, log_pzk_list

    def reconstruct(self, x, sample_from_prior_after=None, variate_masks=None):
        # nmax number of level to sample from the prior
        if variate_masks is None:
            variate_masks = [None] * (sum(hparams.model.down_n_blocks_per_res) + len(hparams.model.down_strides))
        else:
            assert len(variate_masks) == sum(hparams.model.down_n_blocks_per_res) + len(hparams.model.down_strides)

        skip_list = self.bottom_up(x)
        pxz, posterior_dist_list, prior_kl_dist_list = self.top_down(skip_list, variate_masks=variate_masks, sample_from_prior_after=sample_from_prior_after)
        xrec = self.top_down.sample(pxz)
        return xrec, posterior_dist_list, prior_kl_dist_list

    def forward_manual_latents(self, latents, T, sample_from_prior_after=None):
        pxz, log_pzk_list = self.top_down.forward_manual_latents(latents, T=T, sample_from_prior_after=sample_from_prior_after)
        return pxz, log_pzk_list