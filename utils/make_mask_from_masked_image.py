import os
import numpy as np
from PIL import Image
np.random.seed(1)
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("output")
args = parser.parse_args()

img = Image.open(args.input)
img = np.asarray(img)
#print(img.shape)
s = np.sum(img, axis=2)
#print(s.shape)
mask = np.ones(s.shape)
mask[s==0] = 0
np.save(args.output, mask)