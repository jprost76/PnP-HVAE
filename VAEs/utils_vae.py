
#ffhq_shift = -112.8666757481
#
#
#ffhq_scale = 1. / 69.8478027
#    return (x + ffhq_shift) * ffhq_scale
#def normalize_ffhq_input(x):
import yaml
import torch
import numpy as np

try:
    from .vdvae.vae import VAE as VDVAE
#except (ImportError, ValueError):
#    from vdvae.vae import VAE as VDVAE
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

def gaussian_log_likelihood(x, mean, var):
    return - 0.5 * torch.sum(np.log(2 * np.pi) + torch.log(var) + (x-mean)**2 / var) 

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

def load_efficient_vdvae():
    from hparams import HParams
    hparams = HParams(project_path='VAEs/efficient_vdvae', hparams_filename="hparams-ffhq256_8bits_baseline", name="efficient_vdvae")
    try:
        from .efficient_vdvae.utils.utils import create_checkpoint_manager_and_load_if_exists
        from .efficient_vdvae.model.def_model import UniversalAutoEncoder
    except (ImportError, ValueError):
        from efficient_vdvae.utils.utils import create_checkpoint_manager_and_load_if_exists
        from efficient_vdvae.model.def_model import UniversalAutoEncoder
    model = UniversalAutoEncoder().cuda()
    # this step is somehow necessary to activate unpool.scale_biais parameters
    with torch.no_grad():
        _ = model(torch.ones((1, hparams.data.channels, hparams.data.target_res, hparams.data.target_res)).cuda())
    checkpoint, checkpoint_path = create_checkpoint_manager_and_load_if_exists(model_directory='VAEs/efficient_vdvae/saved_models', rank=0)
    print(checkpoint_path)
    #model.load_state_dict(checkpoint['model_state_dict'])
    model.load_state_dict(checkpoint['ema_model_state_dict'])
    receptive_field = 256
    return model, receptive_field

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

# TODO: remove below:
# def init_vae_and_regularization_fn(model_name):
#     if model_name == "vdvae":
#         model, receptive_field = load_vdvae(conf_path='VAEs/vdvae/saved_models/vdvae_ffhq256.yaml')
#         def latent_reg(x, vae, lq, lp, sample_from_prior_after=None, dec_std=None):
#             # input x should be in range [0, 1]
#             with torch.no_grad():
#                 x = torch.clamp(x*256, 0, 255)
#                 x, stats, mu_xz, var_xz = vae.forward_with_latent_reg(x, a=lq, b=lp, quantize=False, nmax=sample_from_prior_after) 
#                 x += 4/255 # correct bias due 5 bit quantization bug
#                 mu_xz += 4/255
#             x = x.clamp(0, 1)
#             mu_xz = mu_xz.clamp(0,1)
#             var_xz = var_xz.clamp(0.004, 0.5)
#             if dec_std is not None:
#                 var_xz = torch.ones_like(mu_xz) * dec_std**2
#             log_pz = [block_stats['ll'] for block_stats in stats]
#             gx = -1 * sum(log_pz)
#             return x, gx, log_pz, mu_xz, var_xz
#         def eval_logpxz(vae, x, z, dec_std=None):
#             # input x should be in range [0, 1]
#             xtarget = x
#             log_pzl, log_pxz, mu_xz, var_xz = vae.eval_logpxz(xtarget, z)
#             log_pz = sum(log_pzl)
#             if dec_std is not None:
#                 var_xz = torch.ones_like(mu_xz) * dec_std**2
#                 #recompute log p(x|z) assuming it is gaussian with diagonal constant variance
#                 log_pxz = gaussian_log_likelihood(x, mu_xz, var_xz)
#             return log_pz, log_pzl, log_pxz, mu_xz, var_xz
#         def encode(vae, x):
#             # input x should be in range [0, 1]
#             with torch.no_grad():
#                 x = torch.clamp(x*256, 0, 255)
#                 stats = vae.forward_get_latents(x)
#                 latents = [d['z'] for d in stats]
#                 return latents
#     elif model_name == "efficient_vdvae":
#         model, receptive_field = load_efficient_vdvae()
#         def latent_reg(x, vae, lq, lp, sample_from_prior_after=None, dec_std=None):
#             # [0, 1] -> [-1, 1]
#             xinput = torch.clamp(x * 2 - 1, -1, 1)
#             with torch.no_grad():
#                 pxz, _, _, log_pzk_list = vae.forward_with_latent_reg(xinput, lq=lq, lp=lp, sample_from_prior_after=sample_from_prior_after)
#             xrec = vae.top_down.sample(pxz) # TODO: remove sampling step
#             xrec = xrec.clamp(0, 1)
#             mu_xz, var_xz = vae.top_down.get_mol_mean_and_var(pxz) # range[-1, 1]
#             mu_xz = torch.clamp((mu_xz + 1) / 2, 0, 1) # [-1, 1] -> [0, 1]
#             # avoid bug due to (rare) NaN output
#             mu_xz[torch.isnan(mu_xz)] = x[torch.isnan(mu_xz)] 
#             if dec_std is not None:
#                 var_xz = torch.ones_like(mu_xz) * dec_std**2
#             else:
#                 var_xz = var_xz = var_xz / 4 
#                 var_xz = var_xz.clamp(0, 0.5)
#                 var_xz[torch.isnan(var_xz)] = 0.01
#             gx = -1 * sum(log_pzk_list)
#             return xrec, gx, log_pzk_list, mu_xz, var_xz
#         def eval_logpxz(vae, x, z, dec_std=None): # input in [0, 1]!
#             pxz, log_pzk_list = vae.forward_manual_latents(z)
#             mu_xz, var_xz = vae.top_down.get_mol_mean_and_var(pxz) # range[-1, 1]
#             mu_xz = torch.clamp((mu_xz + 1) / 2, 0, 1) # [-1, 1] -> [0, 1]
#             mu_xz[torch.isnan(mu_xz)] = x[torch.isnan(mu_xz)] 
#             if dec_std is not None:
#                 var_xz = torch.ones_like(mu_xz) * dec_std**2
#             else:
#                 var_xz = var_xz = var_xz / 4 
#                 var_xz = var_xz.clamp(0, 0.5)
#                 var_xz[torch.isnan(var_xz)] = 0.01
#             log_pz = sum(log_pzk_list)
#             log_pxz = gaussian_log_likelihood(x, mu_xz, var_xz)
#             return log_pz, log_pzk_list, log_pxz, mu_xz, var_xz
#         def encode(vae, x): # input in [0, 1]!
#             # [0, 1] -> [-1, 1]
#             x = torch.clamp(x * 2 - 1, -1, 1)
#             with torch.no_grad():
#                 _, _, _, latents = vae.forward_get_latents(x)
#             return latents
#     else:
#         model, receptive_field = load_patchVDVAE(model_name)
#         # TODO: FileNotFound?
#         def latent_reg(x, vae, lq, lp, sample_from_prior_after=None, dec_std=None):
#             # [0, 1] -> [-1, 1]
#             xinput = torch.clamp(x * 2 - 1, -1, 1)
#             with torch.no_grad():
#                 pxz, _, _, log_pzk_list = vae.forward_with_latent_reg(xinput, lq=lq, lp=lp, sample_from_prior_after=sample_from_prior_after)
#             xrec = vae.top_down.sample(pxz)
#             xrec = xrec.clamp(0, 1)
#             mu_xz, var_xz = torch.chunk(pxz, chunks=2, dim=1)  # range[-1, 1]
#             mu_xz = torch.clamp((mu_xz + 1) / 2, 0, 1) # [-1, 1] -> [0, 1]
#             mu_xz[torch.isnan(mu_xz)] = x[torch.isnan(mu_xz)] 
#             if dec_std is not None:
#                 var_xz = torch.ones_like(mu_xz) * dec_std**2
#             else:
#                 var_xz = var_xz = var_xz / 4 
#                 var_xz[torch.isnan(var_xz)] = 0.5
#             gx = -1 * sum(log_pzk_list)
#             return xrec, gx, log_pzk_list, mu_xz, var_xz
#         def eval_logpxz(vae, x, z, dec_std=None): # input in [0, 1]!
#             pxz, log_pzk_list = vae.forward_manual_latents(z)
#             mu_xz, var_xz = torch.chunk(pxz, chunks=2, dim=1)  # range[-1, 1]
#             mu_xz = torch.clamp((mu_xz + 1) / 2, 0, 1) # [-1, 1] -> [0, 1]
#             mu_xz[torch.isnan(mu_xz)] = x[torch.isnan(mu_xz)] 
#             if dec_std is not None:
#                 var_xz = torch.ones_like(mu_xz) * dec_std**2
#             else:
#                 var_xz = var_xz / 4
#                 var_xz = var_xz.clamp(min=(1/255)**2)
#                 var_xz[torch.isnan(var_xz)] = 0.01
#             log_pz = sum(log_pzk_list)
#             log_pxz = gaussian_log_likelihood(x, mu_xz, var_xz)
#             return log_pz, log_pzk_list, log_pxz, mu_xz, var_xz
#         def encode(vae, x): # input in [0, 1]!
#             # [0, 1] -> [-1, 1]
#             x = torch.clamp(x * 2 - 1, -1, 1)
#             with torch.no_grad():
#                 _, _, _, latents = vae.forward_get_latents(x)
#             return latents
#     #else:
#     #    raise ValueError('model_name should be either \'vdvae\' or \'efficient_vdvae\', got {}'.format(model_name))
#     return model, latent_reg, eval_logpxz, encode, receptive_field