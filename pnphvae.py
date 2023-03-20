import math
import torch
import numpy as np

def compute_psnr(img1, img2):
    # img1 and img2 have range [0, 1]
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    mse = torch.mean((img1 - img2)**2).item()
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1 / math.sqrt(mse))

class PnPHVAE():
    def __init__(self, cfg, vae, data_term, logger, state_fn):
        self.cfg = cfg
        self.vae = vae
        self.data_term = data_term
        self.logger = logger
        self.state_fn = state_fn
        self.T = cfg.exp.T

    def solve(self, xinit, xgt=None):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        k = 0
        cont = True
        beta = self.state_fn.beta_min
        N = len(xinit.flatten())
        xk = xinit.clone().cuda()
        zk = self.vae.encode(xk)
        # compute loss
        nlog_pyx = self.data_term.neg_log_likelihood(xk, reduce='mean')
        log_pz, log_pzl, log_pxz, mu_xzk, var_xzk = self.vae.eval_logpxz(xk, zk, self.cfg.exp.decoder_std)
        log_pz = log_pz / N
        log_pxz = log_pxz / N
        Cxk = nlog_pyx - log_pxz - log_pz/self.T**2
        Cbxk = nlog_pyx - beta*log_pxz - log_pz/self.T**2 
        if self.logger is not None:
            metrics = {
                '-log p(y|x)': nlog_pyx.item(),
                '-log p(z)': -log_pz.item(), 
                '-log p(x|z)': -log_pxz.item(),
                'C(xk,zk)': Cxk.item(),
                'Cb(xk,zk)': Cbxk.item(),
                'beta': beta
            }
            if xgt is not None:
                psnr_x = compute_psnr(xgt, xk)
                self.logger.log_metrics_on_same_curve('psnr', {'xk' : psnr_x}, k)
            self.logger.log_metrics(metrics, k, prefix="loss/")
            if self.cfg.log_logpzl:
                dlogpzl = {str(j): l for j, l in enumerate(log_pzl)}
                self.logger.log_metrics(dlogpzl, k, prefix="logpzl/")
            if self.cfg.log_images:
                images = {
                    'xk': xk.clone()
                }
                self.logger.log_images(images, k)
        rk_old = None
        while cont and k < self.cfg.exp.maxiter:
            k += 1
            xk_old = xk.clone()
            xk, info = self.data_term.data_solution(D=1/var_xzk, u=mu_xzk, alpha=beta, x0=mu_xzk, tol=N*0.001, maxiter=1000 )
            _, nlog_pz, log_pzl_list, mu_xzk, var_xzk = self.vae.latent_reg(xk, lq=beta, lp=1/self.T**2 - beta, dec_std=self.cfg.exp.decoder_std, sample_from_prior_after=self.cfg.exp.sample_from_prior_after)
            nlog_pz = nlog_pz / N
            nlog_pyx = self.data_term.neg_log_likelihood(xk, reduce='mean')
            nlogpxz = 0.5 * torch.mean(np.log(2 * np.pi) + torch.log(var_xzk) + (xk-mu_xzk)**2 / var_xzk)
            Cxk = nlog_pyx + nlogpxz + nlog_pz/self.T**2
            Cbxk = nlog_pyx + beta*nlogpxz + nlog_pz/self.T**2
            rk = torch.mean((xk-xk_old)**2)
            if rk_old is not None:
                Lk = rk/rk_old
                Lk = Lk.item()
            else:
                Lk = torch.nan
            rk_old = rk
            if self.logger is not None:
                metrics = {
                    '-log p(y|x)': nlog_pyx.item(),
                    '-log p(z)': nlog_pz.item(), 
                    '-log p(x|z)': nlogpxz.item(),
                    'C(xk,zk)': Cxk.item(),
                    'Cb(xk,zk)': Cbxk.item(),
                    'rk': rk.item(),
                    'Lk': Lk,
                    'beta': beta
                }
                self.logger.log_metrics(metrics, k, prefix="loss/")
                if xgt is not None:
                    psnr_xk = compute_psnr(xk, xgt)
                    psnr_muk = compute_psnr(mu_xzk, xgt)
                    self.logger.log_metrics_on_same_curve('psnr', {'xk': psnr_xk, 'muk': psnr_muk}, k)
                if self.cfg.log_logpzl:
                    logpzl = {str(j): l for j, l in enumerate(log_pzl_list)}
                    self.logger.log_metrics(logpzl, k, prefix="logpzl/")
                if self.cfg.log_images:
                    images = {
                        'mu_xz': mu_xzk,
                        'xk': xk
                    }
                    self.logger.log_images(images, k)
            beta, cont = self.state_fn.step(loss=Cbxk.item(), residual=rk.item())
        #metrics = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in metrics.items()}
        metrics['niter'] = k
        end.record()
        torch.cuda.synchronize()
        runtime = start.elapsed_time(end)
        metrics['runtime'] = runtime
        return xk, mu_xzk, metrics 

def adam_baseline(vae, data_term, xinit, lr, state_fn, logger, T, log_images=False, maxiter=np.inf, log_logpzl=False, xgt=None, decoder_std=None):
    # TODO: remove? cfg? class?
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    k = 0
    cont = True
    beta = state_fn.beta_min
    N = len(xinit.flatten())
    xk = xinit.clone().cuda()
    xk.requires_grad = True
    zk = vae.encode(vae, xk)
    for j in range(len(zk)):
        zk[j].requires_grad = True
    # init optimizer
    optim = torch.optim.Adam([xk]+zk, lr=lr)
    # compute loss
    nlog_pyx = data_term.neg_log_likelihood(xk, reduce='mean')
    log_pz, log_pzl, log_pxz, _, _ = vae.eval_logpxz(vae, xk, zk, dec_std=decoder_std)
    log_pz = log_pz / N
    log_pxz = log_pxz / N
    Cxk = nlog_pyx - log_pxz - log_pz/T**2
    Cbxk = nlog_pyx - beta*log_pxz - log_pz/T**2
    if logger is not None:
        metrics = {
            '-log p(y|x)': nlog_pyx.item(),
            '-log p(z)': -log_pz.item(), 
            '-log p(x|z)': -log_pxz.item(),
            'C(xk,zk)': Cxk.item(),
            'Cb(xk,zk)': Cbxk.item(),
            'beta': beta
        }
        if xgt is not None:
            with torch.no_grad():
                psnr_xk = compute_psnr(xk, xgt)
            logger.log_metrics({'xk' : psnr_xk}, k, prefix='psnr/')
        logger.log_metrics(metrics, k, prefix="loss/")
        if log_logpzl:
            dlogpzl = {str(j): l/N for j, l in enumerate(log_pzl)}
            logger.log_metrics(dlogpzl, k, prefix="logpzl/")
        if log_images:
            images = {
                'xk': xk.clone()
            }
            logger.log_images(images, k)
    # start iteration
    rk_old = None
    while cont and k < maxiter:
        k += 1
        xk_old = xk.clone()
        optim.zero_grad()
        nlog_pyx = data_term.neg_log_likelihood(xk, reduce='mean')
        log_pz, log_pzl, log_pxz, _, _ = vae.eval_logpxz(vae, xk, zk, dec_std=decoder_std)
        log_pz = log_pz / N
        log_pxz = log_pxz / N
        Cxk = nlog_pyx - log_pxz - log_pz/T**2
        Cbxk = nlog_pyx - beta*log_pxz - log_pz/T**2
        Cbxk.backward()
        optim.step()
        rk = torch.mean((xk-xk_old)**2)
        if rk_old is not None:
            Lk = rk/rk_old
            Lk = Lk.item()
        else:
            Lk = torch.nan
        rk_old = rk
        if logger is not None:
            metrics = {
                '-log p(y|x)': nlog_pyx.item(),
                '-log p(z)': -log_pz.item(), 
                '-log p(x|z)': -log_pxz.item(),
                'C(xk,zk)': Cxk.item(),
                'Cb(xk,zk)': Cbxk.item(),
                'beta': beta,
                'rk': rk.item(),
                'Lk': Lk
            }
            logger.log_metrics(metrics, k, prefix="loss/")
            if xgt is not None:
                with torch.no_grad():
                    psnr_xk = compute_psnr(xk, xgt)
                logger.log_metrics({'xk' : psnr_xk}, k, prefix='psnr/')
            if log_logpzl:
                dlogpzl = {str(j): l for j, l in enumerate(log_pzl)}
                logger.log_metrics(dlogpzl, k, prefix="logpzl/")
            if log_images:
                images = {
                    'xk': xk
                }
                logger.log_images(images, k)
        beta, cont = state_fn.step(loss=Cbxk.item(), residual=rk.item())
    #metrics = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in metrics.items()}
    metrics['niter'] = k    
    end.record()
    torch.cuda.synchronize()
    runtime = start.elapsed_time(end)
    metrics['runtime'] = runtime
    return xk, metrics 