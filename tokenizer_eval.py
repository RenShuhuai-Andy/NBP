import os
import tqdm
import json
import torch
import torch.nn.functional as F
import argparse
import numpy as np
import torch.nn as nn
from torchvision.models.inception import inception_v3
from einops import rearrange
from tqdm import tqdm
from PIL import Image
from evaluation.metrics import get_psnr, get_ssim_and_msssim, get_revd_perceptual
from evaluation.my_lpips import LPIPS
from nbp import VideoData
from nbp.utils import save_video_grid
from nbp.utils import shift_dim
from nbp.fvd.fvd import get_fvd_logits, frechet_distance, load_fvd_model
from nbp.download import load_magvit2
import random
import torch.backends.cudnn as cudnn
import ddp_utils as utils

def calculate_batch_codebook_usage_percentage(batch_encoding_indices,n_codes):
    # Flatten the batch of encoding indices into a single 1D tensor
    all_indices = batch_encoding_indices.flatten()

    # Initialize a tensor to store the percentage usage of each code
    codebook_usage = torch.zeros(n_codes, dtype=torch.long)
    
    # Count the number of occurrences of each index and get their frequency as percentages
    unique_indices, counts = torch.unique(all_indices, return_counts=True)
    
    # Populate the corresponding percentages in the codebook_usage_percentage tensor
    codebook_usage[unique_indices.long()] = counts
    
    return codebook_usage


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

parser = argparse.ArgumentParser()
parser = VideoData.add_data_specific_args(parser)
parser.add_argument('--tokenizer', type=str, default="magvit2")
parser.add_argument('--ckpt', type=str, default=None)
parser.add_argument('--train', action="store_true")
parser.add_argument('--infer_downsample', type=int, default=None)
parser.add_argument('--replacewithgt', type=int, default=None)
parser.add_argument('--save', type=str, default='./results/tats')
parser.add_argument('--dataset', type=str, default='ucf101')
parser.add_argument('--save_videos', action='store_true')
parser.add_argument("--seed", default=42, type=int)
args = parser.parse_args()

n_row = 1 # int(np.sqrt(args.batch_size))

device = torch.device('cuda')

# fix the seed for reproducibility
seed = args.seed + utils.get_rank()
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
cudnn.benchmark = True

if args.tokenizer == "magvit2":
    tokenizer = load_magvit2(args.ckpt, args.resolution, device="cuda")
    tokenizer.eval()
    num_codes = 64000
else:
    raise ValueError()

save_dir = '%s/%s'%(args.save, args.dataset)
print('generating and saving video to %s...'%save_dir)
os.makedirs(save_dir, exist_ok=True)

data = VideoData(args)

if args.train:
    loader = data.train_dataloader()[0]
else:
    loader = data.val_dataloader()


i3d = load_fvd_model(device)
# load perceptual model
perceptual_model = LPIPS().eval()
perceptual_model.cuda(device)

os.makedirs(os.path.join(save_dir, "gt"), exist_ok=True)
os.makedirs(os.path.join(save_dir, "recons"), exist_ok=True)

real_embeddings = []
fake_embeddings = []
psnr_all = []
ssim_all = []
msssim_all = []
get_l1_loss = torch.nn.L1Loss()
total_l1_loss = 0
total_per_loss = 0

total_usage = torch.zeros(num_codes).to(device)

print('computing fvd embeddings for real/fake videos')
i = 0
for batch in tqdm(loader):
    with torch.no_grad():
        input_ = batch['video'] # B C T H W
        B = input_.shape[0]
        if args.tokenizer == "magvit2":
            x_recons, _, _ = tokenizer(input_.to(device), return_id=True)
            x_recons = x_recons.cpu()
        else:
            raise ValueError()

        if args.infer_downsample is not None:
            if args.tokenizer == "magvit2":
                real_videos = torch.clamp(batch['video'], 0, 1)
                fake_videos = torch.clamp(x_recons, 0, 1)
            else:
                raise ValueError()
            B, C, T, H, W = real_videos.shape
            real_videos = rearrange(real_videos, "b c t h w -> (b t) c h w")
            fake_videos = rearrange(fake_videos, "b c t h w -> (b t) c h w")
            real_videos = F.interpolate(
                real_videos, scale_factor=1/args.infer_downsample, mode="bilinear", align_corners=False
            )
            fake_videos = F.interpolate(
                fake_videos, scale_factor=1/args.infer_downsample, mode="bilinear", align_corners=False
            )

            real_videos = rearrange(real_videos, "(b t) c h w -> b c t h w", b=B)
            fake_videos = rearrange(fake_videos, "(b t) c h w -> b c t h w", b=B)
            
        else:
            if args.tokenizer == "magvit2":
                real_videos = torch.clamp(batch['video'], 0, 1)
                fake_videos = torch.clamp(x_recons, 0, 1)
            else:
                raise ValueError()

        
        if args.replacewithgt is not None:
            # B C T H W
            fake_videos = torch.cat((real_videos[:, :, :args.replacewithgt], fake_videos[:, :, args.replacewithgt:]), dim=2)
            assert fake_videos.shape[2] == args.sequence_length

        real_embeddings.append(get_fvd_logits(shift_dim(real_videos * 255, 1, -1).byte().data.numpy(), i3d=i3d, device=device))
        fake_embeddings.append(get_fvd_logits(shift_dim(fake_videos * 255, 1, -1).byte().data.numpy(), i3d=i3d, device=device))

        real_videos = real_videos.cuda()
        fake_videos = fake_videos.cuda()
        # compute L1 loss and perceptual loss
        perceptual_loss = get_revd_perceptual(real_videos[:, :, 5], fake_videos[:, :, 5], perceptual_model)
        l1loss = get_l1_loss(real_videos[:, :, 5], fake_videos[:, :, 5])
        total_l1_loss += l1loss.cpu().item()
        total_per_loss += perceptual_loss.cpu().item()
        # PSNR
        pred_psnr = get_psnr(real_videos[:, :, :-1], fake_videos[:, :, :-1], zero_mean=False, is_video=True).cpu().tolist()
        psnr_all.extend(pred_psnr)
        # SSIM
        pred_ssim, pred_msssim = get_ssim_and_msssim(real_videos[:, :, :-1], fake_videos[:, :, :-1], zero_mean=False, is_video=True)
        pred_ssim = pred_ssim.cpu().tolist()
        pred_msssim = pred_msssim.cpu().tolist()
        ssim_all.extend(pred_ssim)
        msssim_all.extend(pred_msssim)        

    if args.save_videos:
        # save_video_grid(fake_videos, os.path.join(save_dir, "recons", f'{args.dataset}_{i}.gif'), n_row)
        # save_video_grid(real_videos, os.path.join(save_dir, "gt", f'{args.dataset}_{i}.gif'), n_row)

        paths = batch["path"]
        for in_video, sample, path in zip(real_videos, fake_videos, paths):
            video_class = os.path.basename(os.path.dirname(path))
            video_name = os.path.basename(path)
            os.makedirs(os.path.join(save_dir, "recons"), exist_ok=True)
            os.makedirs(os.path.join(save_dir, "gt"), exist_ok=True)
            save_video_grid(sample.unsqueeze(0), os.path.join(save_dir, "recons", video_class + "_" + video_name), 1)
            save_video_grid(in_video.unsqueeze(0), os.path.join(save_dir, "gt", video_class + "_" + video_name), 1)
    
    i += 1
    
print('caoncat fvd embeddings for real videos')
real_embeddings = torch.cat(real_embeddings, 0)
print('caoncat fvd embeddings for fake videos')
fake_embeddings = torch.cat(fake_embeddings, 0)

print('FVD = %.2f'%(frechet_distance(fake_embeddings, real_embeddings)))
print('Usage = %.2f'%((total_usage > 0.).sum() / num_codes))

print(f"have {i} batches")
print('l1loss:', total_l1_loss / i)
print('precep_loss:', total_per_loss / i)
print(f"PSNR: {np.mean(psnr_all):.4f} (±{np.std(psnr_all):.4f})")
print(f"SSIM: {np.mean(ssim_all):.4f} (±{np.std(ssim_all):.4f})")
print(f"MS-SSIM: {np.mean(msssim_all):.4f} (±{np.std(msssim_all):.4f})")
