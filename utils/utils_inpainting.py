import torch
import torch.fft
import numpy as np


def degrad_inpainting_np(x, size=20, margin=50, seed=None):
	if seed is not None:
		np.random.seed(seed)
	H, W, C = x.shape[-3:]
	# get random mask coordinate
	Mh = torch.randint(low=margin, high=H-size-margin, size=(1, 1)).item()
	Mw = torch.randint(low=margin, high=W-size-margin, size=(1, 1)).item()
	Mask = np.ones(shape=(H, W, 1))
	Mask[Mh:Mh+size, Mw:Mw+size, :] = 0
	Masked = Mask * x
	return Masked, Mask 

def degrad_inpainting(x, size=20, margin=50, seed=None):
	if seed is not None:
		torch.manual_seed(seed)
	H, W = x.shape[-2:]
	# get random mask coordinate
	Mh = torch.randint(low=margin, high=H-size-margin, size=(1, 1)).item()
	Mw = torch.randint(low=margin, high=W-size-margin, size=(1, 1)).item()
	Mask = torch.ones(H, W)
	Mask[Mh:Mh+size, Mw:Mw+size] = 0
	Masked = Mask * x
	return Masked, Mask 

#def prox_denoising(x, y, sigma, beta):
#	"""
#	return arg min_u 1/(2sigma²)||u-y||² + (beta/2) ||u-x||²
#	"""
#	return (y + beta * sigma**2 * x) / (1 + beta * sigma**2)

def prox_inpainting(x, y, M, sigma, beta):
	"""
        return arg min_u 1/(2sigma²)||AtAu-y||² + (beta/2) ||u-x||²
        """
	return (M*y + beta * sigma**2 * x) / (M + beta * sigma**2)

def centered_mask(xshape, size):
	M = torch.ones(xshape)
	H, W = xshape[-2:]
	mh = H//2
	mw = W//2
	ms = size // 2
	M[..., mh-ms:mh+ms, mw-ms:mw+ms] = 0
	return M

#def pad_mask(M, x):
#	lx, Lx = x.shape[-2:]
#	lm, Lm = M.shape[-2:]
#	if lx < lm:
#		# crop M
#		a = lm//2 - lx // 2
#		b = M - 1
#		M = M[..., a:b, :]
#	elif lx > lm:
#		# pad M


	


