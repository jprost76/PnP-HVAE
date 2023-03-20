import os
import itertools
import torch
import numpy as np
import cupy as cp
import yaml
from PIL import Image
from torchvision.transforms.functional import to_tensor

ffhq_shift = -112.8666757481
ffhq_scale = 1. / 69.8478027

def normalize_ffhq_input(x):
    return (x + ffhq_shift) * ffhq_scale


def filter_state_dict(state_dict, pattern):
    """
    filter_state_dict({'encoder.block_enc1' : ...,'encoder.block_enc2' : ..., 'decoder.block_dec'}, 'encoder') = {'block_enc1' : ..., 'block_enc2' : ..., }
    """
    # remove pattern + the next character '.' to the key
    return {k[len(pattern)+1:] : v for k, v in state_dict.items() if k.startswith(pattern)}


def save_image_tensor(t, path):
    """
    save a uint8 torch image tensor to path
    """
    im = Image.fromarray(t.squeeze(0).permute(1, 2, 0).cpu().detach().numpy())
    im.save(path)

def load_image_tensor(path, size_multiple=1):
    """
    load a image as a torch tensor of type uint8 and shape (1,C,H,W)
    the image is cropped so that its spatial dimension are a multipe of size_multiple
    """
    img = Image.open(path)
    t = torch.tensor(np.asarray(img)).permute(2, 0, 1).unsqueeze(0)
    H, W = t.shape[-2:]
    H_cropped = H - (H%size_multiple)
    W_cropped = W - (W%size_multiple)
    t_cropped = t[..., :H_cropped, :W_cropped]
    return t_cropped

def crop(xt, size_multiple=1):
    H, W = xt.shape[-2:]
    H_cropped = H - (H%size_multiple)
    W_cropped = W - (W%size_multiple)
    xt_cropped = xt[..., :H_cropped, :W_cropped]
    return xt_cropped
    
def load_image_np(path, format="HWC", dtype=np.float32):
    assert format in ("CHW", "HWC"), 'format should be in (\"CHW\", \"HWC\"), got {}'.format(format)
    img = Image.open(path)
    img = np.asarray(img, dtype=dtype)
    if format == "CHW":
        img = img.transpose(2, 0, 1)
    return np.asarray(img)

def load_image_cp(path, format="CHW"):
    assert format in ("CHW", "HWC"), 'format should be in (\"CHW\", \"HWC\"), got {}'.format(format)
    img = Image.open(path)
    img = cp.asarray(img)
    if format == "CHW":
        img = img.transpose(2, 0, 1)
    return img

def pt2np(t):
    return t.detach().cpu().squeeze().permute(1, 2, 0).numpy()

def np2pt(a):
    return torch.tensor(a).permute(2, 0, 1).unsqueeze(0).float()

def find_avail_dirname(dir):
    i = 0
    while os.path.exists(os.path.join(dir, 'run_{}'.format(i))):
        i+= 1
    return 'run_{:03d}'.format(i)

def product_dict(**kwargs):
    # assert that all value are iterable (e)
    for k, val in kwargs.items():
        if isinstance(val, str):
            kwargs[k] = [val]
        else:
            try:
                iterator = iter(val)
            except TypeError:
                kwargs[k] = [val]
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


def get_fname(filepath):
    fname = os.path.split(filepath)[-1]
    fname = os.path.splitext(fname)[0]
    return fname

#def product_dict(**kwargs):
#    for k, val in kwargs:
#        try:
#            iterator = iter(val)
#        except TypeError:
#            kwargs[k] = [val]
#    print(kwargs)
#    keys = kwargs.keys()
#    vals = kwargs.values()
#    out = []
#    for instance in itertools.product(*vals):
#        out.append(dict(zip(keys, instance)))
#    return out