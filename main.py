import os
from natsort import os_sorted
import hydra
from omegaconf import DictConfig, OmegaConf
import lpips
import torchvision
import numpy as np
import torch

from VAEs import vdvae_wrapper, pvdvae_wrapper
from misc import load_image_tensor, pt2np, crop, get_fname
from utils.forward_operators import SRDegradationOperator, BlurringOperator, InpaintingOperator
from pnphvae import PnPHVAE, AdamBaseline
from utils import loggers
from utils import utils_scheduler as scheduler

from utils import utils_image as utils

def get_log_path(cfg):
    # TODO : log adam
    path = os.path.join(cfg.logdir, cfg.exp.name, cfg.mode, cfg.exp.type)
    if cfg.exp.type == 'sr':
        path = os.path.join(path, f'sf_{cfg.exp.sf}')
    path = os.path.join(path, f'std_{cfg.exp.noise_lvl}')
    if cfg.exp.type in ('sr', 'deblurring'):
        kname = get_fname(cfg.exp.kernel)
        path = os.path.join(path, f'kernel_{kname}')
    os.makedirs(path, exist_ok=True)
    return path

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    # load VAE
    if cfg.exp.vae.lower() == 'vdvae':
        vae = vdvae_wrapper.VDVAEWrapper()
    elif cfg.exp.vae.lower() == 'patchvdvae':
        vae = pvdvae_wrapper.PatchVDVAEWrapper(cfg.exp.vae_conf)
    else:
        raise ValueError
    vae.eval()

    lpips_loss_fn = lpips.LPIPS(net='alex')

    # get images path
    if os.path.isdir(cfg.exp.input_path):
        input_paths = os_sorted([os.path.join(cfg.exp.input_path,p) for p in os.listdir(cfg.exp.input_path)])
    else: 
        input_paths = [cfg.exp.input_path]

    # iterate over images
    for path in input_paths:
        print(path)
        xpt = load_image_tensor(path, size_multiple=vae.receptive_field)

        # init forward model
        
        if cfg.exp.type == 'sr':
            data_term = SRDegradationOperator(sf=cfg.exp.sf, noise_std=cfg.exp.noise_lvl/255, kernel_path=cfg.exp.kernel, xshape=xpt.shape, backprop=cfg.backprop) 
        elif cfg.exp.type == 'deblurring':
            data_term = BlurringOperator(noise_std=cfg.exp.noise_lvl/255, kernel_path=cfg.exp.kernel, xshape=xpt.shape, backprop=cfg.backprop)
        elif cfg.exp.type == 'inpainting':
            mask = np.load(cfg.exp.mask)
            mask = torch.tensor(mask).unsqueeze(0).type_as(xpt)
            mask = crop(mask, size_multiple=vae.receptive_field)
            #mask_name = get_fname(cfg.exp.mask)
            data_term = InpaintingOperator(mask, noise_std=cfg.exp.noise_lvl/255, backprop=cfg.backprop)
        else:
            raise NotImplementedError(f'expecting cfg.exp.type in inpainting, sr, or deblurring, got {cfg.exp.type}')
        ypt = data_term.compute_y(xpt.cuda()/255)

        # starting point
        x0 = data_term.get_x0().cuda() 

        # logger 
        logdir = get_log_path(cfg)
        imname = os.path.splitext(path.split('/')[-1])[0]
        vaename = 'vdvae' if cfg.exp.vae == 'vdvae' else cfg.exp.vae_conf
        xpname = f'{imname}_{vaename}_T_{cfg.exp.temperature.name}_sdec_{cfg.exp.decoder_std}'
        logger = loggers.JSONlogger(cfg, logdir, xpname) 

        # state fonction
        state_fn = scheduler.State_fn(cfg) 
        
        # run 
        if not cfg.backprop:
            solver = PnPHVAE(cfg, vae, data_term, logger, state_fn)
            xk1, muk1, metrics = solver.solve(x0)
            xk1 = muk1.cpu()
        else:
            solver = AdamBaseline(cfg, vae, data_term, logger, state_fn)
            xk1, metrics = solver.solve(x0)
            xk1 = xk1.cpu()

        # lpips (input must be in range [-1, 1])
        llpips = lpips_loss_fn((xk1 - 0.5) * 2, (xpt - 127.5)/127.5).item()
        
        #PSNR / SSIM
        xsolnp = pt2np(xk1.clamp(0, 1)* 255)
        xgtnp = pt2np(xpt)
        psnr = utils.calculate_psnr(xsolnp, xgtnp, border=0)
        ssim = utils.calculate_ssim(xsolnp, xgtnp, border=0)

        # save metrics
        logpath = os.path.join(logdir, 'logs.csv')
        with open(logpath, 'a+') as f:
            f.write(f'{imname};{psnr:.3f};{ssim:.3f};{llpips:.3f};')
            f.write('\n')
        print(f'{imname}; PSNR:{psnr:.3f}; SSIM:{ssim:.3f};LPIPS:{llpips:.3f};')
        # save images

        ypath = os.path.join(logdir, f'{imname}_y.png')
        xpath = os.path.join(logdir, f'{xpname}_sol.png')
        torchvision.utils.save_image(xk1, xpath)
        torchvision.utils.save_image(ypt.detach().cpu(), ypath)

if __name__ == "__main__":
    main()