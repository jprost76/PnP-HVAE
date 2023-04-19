from PIL import Image
import cupy as cp
from cupyx.scipy.sparse import linalg
from cupyx.scipy import ndimage

import torch
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack

import numpy as np
import cv2
from misc import load_image_cp
from .utils_sisr import shift_pixel

def np_kernel_to_conv(k, channels=3, stride=1, pad=False):
    """
    :param k: 2d numpy kernel of dimension (H,W), assume H==W:=p
           pad: If true, add replicate padding of len p//2
    :return: torch.nn.Conv2d module with spatial weight equal to k
    """
    H, W = k.shape
    assert H==W, 'numpy kernel shape (H,W) must be equals, got H={} and W={}'.format(H, W)
    # convert kernel to torch
    k = torch.tensor(k)
    k = k.view(1, 1, H, H)
    k = k.repeat(channels, 1, 1, 1)

    if pad:
        filter = torch.nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=H, stride=stride, groups=channels,
                                padding=H//2, padding_mode='circular', bias=False)
    else:
        filter = torch.nn.Conv2d(in_channels=channels, out_channels=channels,
                                 kernel_size=H, stride=stride, groups=channels,
                                 padding=0, bias=False)
    filter.weight.data = k
    filter.weight.requires_grad = False

    return filter

def _cp_to_tensor(cparray):
    if isinstance(cparray, cp.ndarray):
        tensor = from_dlpack(cparray.toDlpack())
    else:
        tensor = cparray
    return tensor

def _tensor_to_cp(tensor):
    if isinstance(tensor, torch.Tensor):
        cx = cp.from_dlpack(to_dlpack(tensor))
    else:
        cx = tensor
    return cx

def upsample_cp(x, sf=3):
    '''s-fold upsampler
    Upsampling the spatial size by filling the new entries with zeros
    x: tensor image, NxCxWxH
    '''
    st = 0
    z = cp.zeros((x.shape[0], x.shape[1]*sf, x.shape[2]*sf)).astype(x.dtype)
    z[..., st::sf, st::sf] = cp.copy(x)
    return z


def downsample_cp(x, sf=3):
    '''s-fold downsampler
    Keeping the upper-left pixel for each distinct sfxsf patch and discarding the others
    x: tensor image, NxCxWxH
    '''
    st = 0
    return x[..., st::sf, st::sf]

def downsample_pt(x, sf):
    st = 0
    return x[..., st::sf, st::sf]

def upsample_pt(x, sf):
    '''s-fold upsampler
    Upsampling the spatial size by filling the new entries with zeros
    x: tensor image, NxCxWxH
    '''
    st = 0
    z = torch.zeros((x.shape[0], x.shape[1]*sf, x.shape[2]*sf)).astype(x.dtype)
    z[..., st::sf, st::sf] = x 
    return z


def _init_cp_sr_linop(cp_filter, xshape, sf):
    """
    load SR degradation operator y = SHx, with H convolution operator with kernel cp_filter, 
    and S decimation operator with scale factor sf
    Args:
        cp_filter (cp.array): _description_
        xshape (tupple): _description_
        sf (int): _description_

    Returns:
        (cupyx.scipy.sparse.linalg.LinearOperator): 
    """
    if len(xshape) > 3:
        xshape = xshape[-3:]
    C, H, W = xshape
    if H % sf != 0 or W % sf != 0:
        raise ValueError('HR images spatial dimensions ({H}, {W}) should be a multiple of SR scale factor {sf}')
    # TODO: check valid dimension?
    Cy, Hy, Wy = (C, H//sf, W//sf)
    yshape = (C, H//sf, W//sf)
    Nx = C*H*W
    Ny = Cy*Hy*Wy 
    kcp = cp_filter
    kcp_C = kcp[::-1, ::-1].copy()
    # linear operator work with flat vector
    SH = linalg.LinearOperator(
        shape=(Ny, Nx),
        matvec = lambda x: downsample_cp(ndimage.convolve(x.reshape(xshape), cp.expand_dims(kcp, axis=0), mode='wrap'), sf=sf).flatten(),
        rmatvec = lambda y: ndimage.convolve(upsample_cp(y.reshape(yshape), sf=sf), cp.expand_dims(kcp_C, axis=0), mode='wrap').flatten()
    )
    return SH

def _init_cp_blur_linop(cp_filter, xshape):
    """
    load blur operator y = Hx, with H convolution operator with kernel cp_filter
    Args:
        cp_filter (cp.array): _description_
        xshape (tupple): _description_

    Returns:
        (cupyx.scipy.sparse.linalg.LinearOperator): 
    """
    if len(xshape) > 3:
        xshape = xshape[-3:]
    C, H, W = xshape
    Nx = C*H*W
    kcp = cp_filter
    kcp_C = kcp[::-1, ::-1].copy()
    # linear operator work with flat vector
    H = linalg.LinearOperator(
        shape=(Nx, Nx),
        matvec = lambda x: ndimage.convolve(x.reshape(xshape), cp.expand_dims(kcp, axis=0), mode='wrap').flatten(),
        rmatvec = lambda y: ndimage.convolve(y.reshape(xshape), cp.expand_dims(kcp_C, axis=0), mode='wrap').flatten()
    )
    return H

class SRDegradationOperator():
    def __init__(self, sf, noise_std, kernel_path, xshape, backprop=False):
        self._backprop = backprop
        self.noise_std = noise_std
        self.sf = sf
        self.xshape = xshape
        self.yshape = (xshape[-3], xshape[-2]//sf, xshape[-1]//sf)
        if backprop:
            fnp = np.load(kernel_path)
            fnp = fnp/fnp.sum()
            self.conv_pt = np_kernel_to_conv(fnp, pad=True).cuda()
        else:
            #self.filter = load_image_cp(kernel_path, format="CHW")[0,:,:]
            self.filter = cp.asarray(np.load(kernel_path))
            self.filter = self.filter / self.filter.sum()
            self.SH = _init_cp_sr_linop(self.filter, xshape, sf)

    def forward_linear(self, xpt):
        if self._backprop:
            ypt = self.conv_pt(xpt)
            ypt = downsample_pt(ypt, sf=self.sf)
        else:
            xcp = _tensor_to_cp(xpt).flatten()
            ycp = self.SH(xcp)
            ypt = torch.as_tensor(ycp, device='cuda').reshape(self.yshape)
        return ypt

    def compute_y(self, xpt):
        if self._backprop:
            ypt = self.forward_linear(xpt)
            ypt += self.noise_std*torch.randn_like(ypt)
            self.ypt = ypt
        else:
            xcp = _tensor_to_cp(xpt).flatten()
            ycp = self.SH.dot(xcp)
            ycp += cp.random.normal(0, self.noise_std, ycp.shape)
            self.ycp = ycp
            # precompute Aty for data solution
            self.Atycp = self.SH.rmatvec(self.ycp)
            ypt = torch.as_tensor(ycp).reshape(self.yshape) #.permute()
        return ypt

    def pseudo_inverse(self, y):
        pass

    def data_solution(self, D, u, alpha, x0=None, tol=None, maxiter=None):
        """return arg min_x ||Ax-y||^2/ (2 sigma^2) + alpha * (x-u)t.D.(x-u)/2

        Args: 
            u (torch.tensor or cp.array): _description_
            D (torch.tensor or cp.array): diagonal of D (diagonal operator)
            alpha (float): 
        """
        if self._backprop:
            raise NotImplemented('non implemented when backprop=True')
        else:
            ucp = _tensor_to_cp(u).flatten()
            Dcp = _tensor_to_cp(D).flatten()
            x0cp = _tensor_to_cp(x0).flatten()
            A = self.SH
            N = len(ucp)
            # init DtD = D^2 : linear operator
            Dlo = linalg.LinearOperator(
                shape=(N, N),
                matvec = lambda x: x * Dcp,
                rmatvec = lambda x: x * Dcp
            )
            # solve
            asig2 = alpha * self.noise_std**2
            b = self.Atycp + asig2*Dlo.dot(ucp)
            x, info = linalg.cg(A.T.dot(A)+asig2*Dlo, b=b, x0=x0cp, tol=tol, maxiter=maxiter)
            xpt = _cp_to_tensor(x).type(u.dtype).reshape(self.xshape)
        return xpt, info

    def sample(self, mu_xz, sigma_xz):
        """
        sample x ~ p(x|z, y), where p(x|z) ~ N(x; mu_xz, sigma_xz**2) and p(y|x) ~ N(y: Ax, sigma^2 I)
        """
        raise NotImplemented #TODO

    def get_x0(self):
        C, H, W = self.yshape
        ynp = np.asarray(self.ycp.get()).reshape(self.yshape).transpose(1, 2, 0)
        xnp = cv2.resize(ynp, (W*self.sf, H*self.sf), interpolation=cv2.INTER_CUBIC) # TODO : resize on cp?
        xnp = shift_pixel(xnp, self.sf)
        xpt = torch.as_tensor(xnp).permute(2, 0, 1).unsqueeze(0)
        return xpt


    def neg_log_likelihood(self, xpt, reduce='mean'):
        if self._backprop:
            Hx = self.forward_linear(xpt)
            if reduce == 'mean':
                ll = torch.mean((Hx - self.ypt)**2) / (2 * self.noise_std**2)
            elif reduce == 'sum':
                ll = torch.sum((Hx - self.ypt)**2) / (2 * self.noise_std**2)
            else:
                raise ValueError('got reduce = {}, expecting \"mean\" or \"sum\"'.format(reduce))
        else:
            xcp = _tensor_to_cp(xpt).flatten()
            SHxcp = self.SH.dot(xcp)
            if reduce == 'mean':
                llcp = cp.mean((SHxcp - self.ycp)**2) / (2 * self.noise_std**2)
            elif reduce == 'sum':
                llcp = cp.sum((SHxcp - self.ycp)**2) / (2 * self.noise_std**2)
            else:
                raise ValueError('reduce = {}, expecting \"mean\" or \"sum\"'.format(reduce))
            ll = _cp_to_tensor(llcp)
        return ll

    def set_y(self, y):
        if self._backprop:
            self.ypt = y
        else:
            self.ycp = _tensor_to_cp(y).flatten()
            # precompute Aty for data solution
            self.Atycp = self.SH.rmatvec(self.ycp)

class InpaintingOperator():
    def __init__(self, Mask, noise_std):
        self.M = Mask.cuda()
        self.noise_std = noise_std
        
    def forward_linear(self, xpt):
        #self.M = self.M.to(xpt.device)
        y = self.M * xpt
        return y

    def compute_y(self, xpt):
        #self.M = self.M.to(xpt.device)
        y = self.M * xpt
        y += torch.empty_like(y).normal_(mean=0, std=self.noise_std)
        self.y = y
        self.ypt = y
        return y

    def data_solution(self, D, u, alpha, x0=None, tol=None, maxiter=None):
        """return arg min_x ||Mx-y||^2 / (2 sigma^2) + alpha * (x-u)t.D.(x-u)

        Args: 
            u (torch.tensor): _description_
            D (torch.tensor): diagonal of D (diagonal operator)
            alpha (float): 
        """
        xk = (self.M*self.y + alpha * self.noise_std**2 * D * u) / (self.M + alpha * self.noise_std**2 * D)
        return xk, 0
    
    def sample(self, mu_xz, var_xz):
        """
        sample x ~ p(x|z, y), where p(x|z) ~ N(x; mu_xz, var_xz) and p(y|x) ~ N(y: Ax, sigma^2 I)
        """
        # cov =  1 / (self.M / (self.noise_std**2) + torch.ones_like(self.M) / var_xz)
        # mean = cov * (self.M*self.y / self.noise_std**2 + mu_xz / var_xz)
        # r = torch.randn_like(self.ypt)
        # x = mean + r*cov**0.5
        r = self.noise_std**2 / var_xz
        mean = (self.M*self.y + r*mu_xz) / (self.M + r)
        std = (self.M/self.noise_std**2 + 1/var_xz)**(-0.5)
        x = mean + torch.randn_like(mean) * std
        return x, 0
    
    def neg_log_likelihood(self, xpt, reduce='mean'):
        Mx = self.M * xpt
        if reduce == 'mean':
            ll = torch.mean((Mx - self.y)**2) / (2 * self.noise_std**2)
        elif reduce == 'sum':
            ll = torch.sum((Mx - self.y)**2) / (2 * self.noise_std**2)
        else:
            raise ValueError('reduce = {}, expecting \"mean\" or \"sum\"'.format(reduce))
        return ll

    def set_y(self, y):
        self.y = y.cuda()

    def get_x0(self):
        x0 = self.ypt.clone()
        x0 = x0 * self.M + (1-self.M) * 0.5
        return self.ypt

class BlurringOperator():
    def __init__(self, noise_std, kernel_path, xshape, backprop=False):
        self.noise_std = noise_std
        self.xshape = xshape
        self._backprop = backprop
        if backprop:
            fnp = np.load(kernel_path)
            fnp = fnp/fnp.sum()
            self.conv_pt = np_kernel_to_conv(fnp, pad=True).cuda()
        else:
            #self.filter = load_image_cp(kernel_path, format="CHW")[0,:,:]
            self.filter = cp.asarray(np.load(kernel_path))
            self.filter = self.filter / self.filter.sum()
            self.H = _init_cp_blur_linop(self.filter, xshape)

    def forward_linear(self, xpt):
        if self._backprop:
            ypt = self.conv_pt(xpt)
        else:
            xcp = _tensor_to_cp(xpt).flatten()
            ycp = self.H(xcp)
            ypt = _cp_to_tensor(ycp).reshape(self.xshape)
        return ypt

    def compute_y(self, xpt):
        if self._backprop:
            ypt = self.forward_linear(xpt)
            ypt += self.noise_std*torch.randn_like(ypt)
            self.ypt = ypt
        else:
            xcp = _tensor_to_cp(xpt).flatten()
            ycp = self.H.dot(xcp)
            ycp += cp.random.normal(0, self.noise_std, ycp.shape)
            self.ycp = ycp
            # precompute Aty for data solution
            self.Atycp = self.H.rmatvec(self.ycp)
            ypt = torch.as_tensor(ycp).reshape(self.xshape)
            self.ypt = ypt
        return ypt

    def pseudo_inverse(self, y):
        pass #TODO

    def data_solution(self, D, u, alpha, x0=None, tol=None, maxiter=None):
        """return arg min_x ||Ax-y||^2/ (2 sigma^2) + alpha * (x-u)t.D.(x-u)/2

        Args: 
            u (torch.tensor or cp.array): _description_
            D (torch.tensor or cp.array): diagonal of D (diagonal operator)
            alpha (float): 
        """
        if self._backprop:
            raise NotImplemented('non implemented when backprop=True')
        else:
            ucp = _tensor_to_cp(u).flatten()
            Dcp = _tensor_to_cp(D).flatten()
            x0cp = _tensor_to_cp(x0).flatten()
            A = self.H
            N = len(ucp)
            # init DtD = D^2 : linear operator
            Dlo = linalg.LinearOperator(
                shape=(N, N),
                matvec = lambda x: x * Dcp,
                rmatvec = lambda x: x * Dcp
            )
            # solve
            asig2 = alpha * self.noise_std**2
            b = self.Atycp + asig2*Dlo.dot(ucp)
            x, info = linalg.cg(A.T.dot(A)+asig2*Dlo, b=b, x0=x0cp, tol=tol, maxiter=maxiter)
            xpt = _cp_to_tensor(x).type(u.dtype).reshape(self.xshape)
        return xpt, info

    def sample(self, mu_xz, sigma_xz):
        """
        sample x ~ p(x|z, y), where p(x|z) ~ N(x; mu_xz, sigma_xz**2) and p(y|x) ~ N(y: Ax, sigma^2 I)
        """
        raise NotImplemented #TODO

    def get_x0(self):
        return self.ypt

    def neg_log_likelihood(self, xpt, reduce='mean'): # TODO -> rename neg_log_likelihood
        if self._backprop:
            Hx = self.forward_linear(xpt)
            if reduce == 'mean':
                ll = torch.mean((Hx - self.ypt)**2) / (2 * self.noise_std**2)
            elif reduce == 'sum':
                ll = torch.sum((Hx - self.ypt)**2) / (2 * self.noise_std**2)
            else:
                raise ValueError('got reduce = {}, expecting \"mean\" or \"sum\"'.format(reduce))
        else:
            xcp = _tensor_to_cp(xpt).flatten()
            Hxcp = self.H.dot(xcp)
            if reduce == 'mean':
                llcp = cp.mean((Hxcp - self.ycp)**2) / (2 * self.noise_std**2)
            elif reduce == 'sum':
                llcp = cp.sum((Hxcp - self.ycp)**2) / (2 * self.noise_std**2)
            else:
                raise ValueError('got reduce = {}, expecting \"mean\" or \"sum\"'.format(reduce))
            ll = _cp_to_tensor(llcp)
        return ll

    def set_y(self, y):
        if self._backprop:
            self.ypt = y
        else:
            self.ycp = _tensor_to_cp(y).flatten()
            # precompute Aty for data solution
            self.Atycp = self.H.rmatvec(self.ycp)