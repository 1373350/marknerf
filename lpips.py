import torch
import lpips
from fengxi import payloadimage,dpimage
# from IPython import embed
import os

use_gpu = False  # Whether to use GPU
spatial = True  # Return a spatial map of perceptual distance.

# Linearly calibrated models (LPIPS)
loss_fn = lpips.LPIPS(net='alex', spatial=spatial)  # Can also set net = 'squeeze' or 'vgg'
# loss_fn = lpips.LPIPS(net='alex', spatial=spatial, lpips=False) # Can also set net = 'squeeze' or 'vgg'

if (use_gpu):
    loss_fn.cuda()

## Example usage with dummy tensors
#rood_path = r'D:\Project\results\faces'
#im0_path_list = []
#im1_path_list = []
#for root, _, fnames in sorted(os.walk(rood_path, followlinks=True)):
    #for fname in fnames:
        #path = os.path.join(root, fname)
        #if '_generated' in fname:
            #im0_path_list.append(path)
        #elif '_real' in fname:
            #im1_path_list.append(path)

dist_ = []

dummy_im0 = lpips.im2tensor(lpips.load_image(payloadimage))
dummy_im1 = lpips.im2tensor(lpips.load_image(dpimage))
if (use_gpu):
    dummy_im0 = dummy_im0.cuda()
    dummy_im1 = dummy_im1.cuda()
    dist = loss_fn.forward(dummy_im0, dummy_im1)
    dist_.append(dist.mean().item())
print('Avarage Distances: %.3f' % (sum(dist_) /1))
