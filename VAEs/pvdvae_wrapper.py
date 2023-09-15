import torch
from .base_wrapper import BaseVAE
from .utils_vae import gaussian_log_likelihood


def load_patchVDVAE(hps_filename='hdn_bn'):
    from hparams import HParams
    try:
        hparams = HParams(project_path='VAEs/patchVDVAE/src/saved_models', hparams_filename=hps_filename, name="patch_vdvae")
    except ValueError:
        # unload previously loaded params
        HParams._loaded_hparams_objects = {}
        hparams = HParams(project_path='VAEs/patchVDVAE/src/saved_models', hparams_filename=hps_filename, name="patch_vdvae")
    from VAEs.patchVDVAE.src.utils.utils import create_checkpoint_manager_and_load_if_exists
    from VAEs.patchVDVAE.src.model.def_model import UniversalAutoEncoder
    print('ok')
    model = UniversalAutoEncoder().cuda()
    # this step is somehow necessary to activate unpool.scale_biais parameters
    with torch.no_grad():
        _ = model(torch.ones((2, hparams.data.channels, hparams.data.target_res, hparams.data.target_res)).cuda())
    checkpoint, checkpoint_path = create_checkpoint_manager_and_load_if_exists(model_directory='VAEs/patchVDVAE/src/saved_models', rank=0)
    print(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.load_state_dict(checkpoint['ema_model_state_dict'])
    receptive_field = hparams.model.receptive_field
    return model, receptive_field

class PatchVDVAEWrapper(BaseVAE):
    def __init__(self, confname) -> None:
        super().__init__()
        vae, rf = load_patchVDVAE(confname)
        self.vae = vae
        self.receptive_field = rf

    def eval_logpxz(self, x, z, T, decoder_std):
        pxz, log_pzk_list = self.vae.forward_manual_latents(z, T=T)
        mu_xz, var_xz = torch.chunk(pxz, chunks=2, dim=1)  # range[-1, 1]
        mu_xz = torch.clamp((mu_xz + 1) / 2, 0, 1) # [-1, 1] -> [0, 1]
        mu_xz[torch.isnan(mu_xz)] = x[torch.isnan(mu_xz)] 
        if decoder_std is not None:
            var_xz = torch.ones_like(mu_xz) * decoder_std**2
        else:
            var_xz = var_xz / 4
            var_xz = var_xz.clamp(min=(1/255)**2)
            var_xz[torch.isnan(var_xz)] = 0.01
        log_pz = sum(log_pzk_list)
        log_pxz = gaussian_log_likelihood(x, mu_xz, var_xz)
        return log_pz, log_pzk_list, log_pxz, mu_xz, var_xz

    def encode(self, x):
        # [0, 1] -> [-1, 1]
        x = torch.clamp(x * 2 - 1, -1, 1)
        with torch.no_grad():
            _, _, _, latents = self.vae.forward_get_latents(x)
        return latents

    def latent_reg(self, x, beta, T, mode, sample_from_prior_after=None, dec_std=None):
        # [0, 1] -> [-1, 1]
        xinput = torch.clamp(x * 2 - 1, -1, 1)
        with torch.no_grad():
            pxz, _, _, log_pzk_list = self.vae.forward_with_latent_reg(xinput, beta=beta, T=T, mode=mode, sample_from_prior_after=sample_from_prior_after)
        xrec = self.vae.top_down.sample(pxz)
        xrec = xrec.clamp(0, 1)
        mu_xz, var_xz = torch.chunk(pxz, chunks=2, dim=1)  # range[-1, 1]
        mu_xz = torch.clamp((mu_xz + 1) / 2, 0, 1) # [-1, 1] -> [0, 1]
        mu_xz[torch.isnan(mu_xz)] = x[torch.isnan(mu_xz)] 
        if dec_std is not None:
            var_xz = torch.ones_like(mu_xz) * dec_std**2
        else:
            var_xz = var_xz = var_xz / 4 
            var_xz[torch.isnan(var_xz)] = 0.5
        gx = -1 * sum(log_pzk_list)
        return xrec, gx, log_pzk_list, mu_xz, var_xz

       