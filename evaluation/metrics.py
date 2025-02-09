import math
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim, ms_ssim
from torchvision import transforms
from einops import rearrange, repeat, reduce, pack, unpack
transform_rev = transforms.Normalize([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225], [1. / 0.229, 1. / 0.224, 1. / 0.225])


class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2, dim=[1, 2, 3]):

        mse = torch.mean((img1 - img2) ** 2, dim=dim)
        return 20 * torch.log10(255.0 / torch.sqrt(mse))


def get_psnr(x_input, x_recon, zero_mean=False, is_video=False):
    if zero_mean:
        x_input_0_255 = ((x_input + 1) * 127.5)
        x_recon_0_255 = ((x_recon + 1) * 127.5)
    else:
        x_input_0_255 = x_input * 255
        x_recon_0_255 = x_recon * 255
    if is_video:
        psnr = PSNR()(x_input_0_255, x_recon_0_255, dim=[1,2,3,4])
    else:
        psnr = PSNR()(x_input_0_255, x_recon_0_255)
    return psnr


def get_ssim(x_input, x_recon, zero_mean=False, is_video=False):
    if zero_mean:
        x_input_0_255 = ((x_input + 1) * 127.5)
        x_recon_0_255 = ((x_recon + 1) * 127.5)
    else:
        x_input_0_255 = x_input * 255
        x_recon_0_255 = x_recon * 255
    if is_video:
        ssim_val = [
            ssim(x_input_0_255[:, :, t, :, :], x_recon_0_255[:, :, t, :, :], data_range=255, size_average=False)
            for t in range(x_input_0_255.shape[2])
        ]
        ssim_val = torch.stack(ssim_val).mean(0)
    else:
        ssim_val = ssim(x_input_0_255, x_recon_0_255, data_range=255, size_average=False)
    return ssim_val


def get_ssim_and_msssim(x_input, x_recon, zero_mean=False, is_video=False):
    if x_input.shape[2 + is_video] < 256 or x_input.shape[3 + is_video] < 256:
        ssim_val = get_ssim(x_input, x_recon, zero_mean, is_video)
        return ssim_val, torch.ones_like(ssim_val) * torch.nan
    if zero_mean:
        x_input_0_255 = ((x_input + 1) * 127.5)
        x_recon_0_255 = ((x_recon + 1) * 127.5)
    else:
        x_input_0_255 = x_input * 255
        x_recon_0_255 = x_recon * 255
    if is_video:
        ssim_val = [
            ssim(x_input_0_255[:, :, t, :, :], x_recon_0_255[:, :, t, :, :], data_range=255, size_average=False)
            for t in range(x_input_0_255.shape[2])
        ]
        ms_ssim_val = [
            ms_ssim(x_input_0_255[:, :, t, :, :], x_recon_0_255[:, :, t, :, :], data_range=255, size_average=False)
            for t in range(x_input_0_255.shape[2])
        ]
        ssim_val = torch.stack(ssim_val).mean(0)
        ms_ssim_val = torch.stack(ms_ssim_val).mean(0)
    else:
        ssim_val = ssim(x_input_0_255, x_recon_0_255, data_range=255, size_average=False)
        ms_ssim_val = ms_ssim(x_input_0_255, x_recon_0_255, data_range=255, size_average=False)
    return ssim_val, ms_ssim_val


def get_revd_perceptual(inputs, recons, perceptual_model):
    if inputs.ndim == 5:
        # flattened frames
        inputs = rearrange(inputs, 'b c f h w -> (b f) c h w')
        recons = rearrange(recons, 'b c f h w -> (b f) c h w')
    return torch.mean(perceptual_model(transform_rev(inputs.contiguous()), transform_rev(recons.contiguous())))
