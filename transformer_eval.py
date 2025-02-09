import os
import random
import argparse
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch.distributed as dist
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from nbp.download import load_magvit2
from nbp import load_transformer
from nbp import DecordVideoDataset, ImageDataset
from nbp.utils import save_video_grid
from nbp.modules.gpt import sample_with_past, hybrid_sample_with_past, hybrid_sample_with_past_cfg
import ddp_utils as utils
from einops import rearrange, repeat
from evaluation.common_metrics_on_video_quality.fvd.styleganv.fvd import get_fvd_feats, frechet_distance, load_i3d_pretrained, preprocess_single
from evaluation.metrics import get_psnr, get_ssim_and_msssim, get_revd_perceptual
from evaluation.my_lpips import LPIPS
import itertools
from torchmetrics.image.fid import FrechetInceptionDistance
get_l1_loss = torch.nn.L1Loss()


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def encode(tokenizer_name, tokenizer, raw_video):
    if tokenizer_name == "magvit2":
        _, index = tokenizer.encode(raw_video, quantize=True)
    else:
        raise NotImplementedError("Tokenizer not implemented")
    return index


def decode(tokenizer_name, tokenizer, index):
    if tokenizer_name == "magvit2":
        pred_video = tokenizer.decode_from_code_indices(index)
        pred_video = torch.clamp(pred_video, 0, 1)
    else:
        raise NotImplementedError("Tokenizer not implemented")
    return pred_video


class Metrics:
    def __init__(self, i3d_model, perceptual_model, is_image):
        self.i3d_model = i3d_model
        self.perceptual_model = perceptual_model
        self.fid = FrechetInceptionDistance(feature=2048, normalize=True).to(torch.cuda.current_device())
        self.total_l1_loss = 0
        self.total_per_loss = 0
        self.feats_real_all = []
        self.feats_fake_all = []
        self.psnr_all = []
        self.ssim_all = []
        self.msssim_all = []
        self.is_image = is_image

    def cal_metrics(self, raw_video, pred_video):
        '''
        raw_video: [B, C, T, H, W], range in [0, 1]
        pred_video: [B, C, T, H, W], range in [0, 1]
        '''
        if self.is_image:
            # compute L1 loss and perceptual loss
            perceptual_loss = get_revd_perceptual(raw_video[:, :, 0], pred_video[:, :, 0], self.perceptual_model)
            l1loss = get_l1_loss(raw_video[:, :, 0], pred_video[:, :, 0])
            # for FID
            self.fid.update(raw_video[:, :, 0], real=True)
            self.fid.update(pred_video[:, :, 0], real=False)
            return perceptual_loss, l1loss
        else:
            # compute L1 loss and perceptual loss
            perceptual_loss = get_revd_perceptual(raw_video[:, :, 5], pred_video[:, :, 5], self.perceptual_model)
            l1loss = get_l1_loss(raw_video[:, :, 5], pred_video[:, :, 5])
            # for FVD
            feats_real = get_fvd_feats(raw_video[:, :, :16], i3d=self.i3d_model, device=torch.cuda.current_device())
            feats_fake = get_fvd_feats(pred_video[:, :, :16], i3d=self.i3d_model, device=torch.cuda.current_device())
            # PSNR
            pred_psnr = get_psnr(raw_video[:, :, :16], pred_video[:, :, :16], zero_mean=False, is_video=True).cpu().tolist()
            # SSIM
            pred_ssim, pred_msssim = get_ssim_and_msssim(raw_video[:, :, :16], pred_video[:, :, :16], zero_mean=False, is_video=True)
            pred_ssim = pred_ssim.cpu().tolist()
            pred_msssim = pred_msssim.cpu().tolist()
            return perceptual_loss, l1loss, feats_real, feats_fake, pred_psnr, pred_ssim, pred_msssim

    def update_metrics(self, raw_video, pred_video):
        if self.is_image:
            perceptual_loss, l1loss = self.cal_metrics(raw_video, pred_video)
        else:
            perceptual_loss, l1loss, feats_real, feats_fake, pred_psnr, pred_ssim, pred_msssim = self.cal_metrics(raw_video, pred_video)
            self.feats_real_all.append(feats_real)
            self.feats_fake_all.append(feats_fake)
            self.psnr_all.extend(pred_psnr)
            self.ssim_all.extend(pred_ssim)
            self.msssim_all.extend(pred_msssim)

        batch_size = raw_video.size(0)
        self.total_l1_loss += l1loss.cpu().item() * batch_size
        self.total_per_loss += perceptual_loss.cpu().item() * batch_size


    def merge_metrics(self, distributed, n_sample):
        if self.is_image:
            print(f"fid real_features_sum: {self.fid.real_features_num_samples}, fake_features_sum: {self.fid.fake_features_num_samples}")
            print('FID score', self.fid.compute().item())
        else:
            self.feats_real_all = np.concatenate(self.feats_real_all, axis=0)
            self.feats_fake_all = np.concatenate(self.feats_fake_all, axis=0)
            if distributed:
                world_size = dist.get_world_size()
                feats_real_all_gathered = [None for _ in range(world_size)]
                feats_fake_all_gathered = [None for _ in range(world_size)]
                psnr_all_gathered = [None for _ in range(world_size)]
                ssim_all_gathered = [None for _ in range(world_size)]
                msssim_all_gathered = [None for _ in range(world_size)]
                total_l1_loss_gathered = [None for _ in range(world_size)]
                total_per_loss_gathered = [None for _ in range(world_size)]

                # Gather metrics from all processes
                dist.all_gather_object(feats_real_all_gathered, self.feats_real_all)
                dist.all_gather_object(feats_fake_all_gathered, self.feats_fake_all)
                dist.all_gather_object(psnr_all_gathered, self.psnr_all)
                dist.all_gather_object(ssim_all_gathered, self.ssim_all)
                dist.all_gather_object(msssim_all_gathered, self.msssim_all)
                dist.all_gather_object(total_l1_loss_gathered, self.total_l1_loss)
                dist.all_gather_object(total_per_loss_gathered, self.total_per_loss)

                if dist.get_rank() == 0:
                    # Concatenate gathered metrics
                    feats_real_all = np.concatenate(feats_real_all_gathered, axis=0)
                    feats_fake_all = np.concatenate(feats_fake_all_gathered, axis=0)
                    print(f"fvd feats_real_all shape: {feats_real_all.shape}, feats_fake_all shape: {feats_fake_all.shape}")
                    psnr_all = np.concatenate(psnr_all_gathered, axis=0)
                    ssim_all = np.concatenate(ssim_all_gathered, axis=0)
                    msssim_all = np.concatenate(msssim_all_gathered, axis=0)
                    total_l1_loss = sum(total_l1_loss_gathered)
                    total_per_loss = sum(total_per_loss_gathered)

                    # Calculate FVD score
                    fvd_score = frechet_distance(feats_fake_all, feats_real_all)
                    print(f"FVD score: {fvd_score}")
                    print('l1loss:', total_l1_loss / n_sample)
                    print('perceptual_loss:', total_per_loss / n_sample)
                    print(f"PSNR: {np.mean(psnr_all):.4f} (±{np.std(psnr_all):.4f})")
                    print(f"SSIM: {np.mean(ssim_all):.4f} (±{np.std(ssim_all):.4f})")
                    print(f"MS-SSIM: {np.mean(msssim_all):.4f} (±{np.std(msssim_all):.4f})")
            else:
                print(f"fvd feats_real_all shape: {self.feats_real_all.shape}, feats_fake_all shape: {self.feats_fake_all.shape}")
                fvd_score = frechet_distance(self.feats_fake_all, self.feats_real_all)
                print(f"FVD score: {fvd_score}")
                print('l1loss:', self.total_l1_loss / n_sample)
                print('perceptual_loss:', self.total_per_loss / n_sample)
                print(f"PSNR: {np.mean(self.psnr_all):.4f} (±{np.std(self.psnr_all):.4f})")
                print(f"SSIM: {np.mean(self.ssim_all):.4f} (±{np.std(self.ssim_all):.4f})")
                print(f"MS-SSIM: {np.mean(self.msssim_all):.4f} (±{np.std(self.msssim_all):.4f})")


@torch.no_grad()
def class_condition_generation(loader, gpt, tokenizer, args, save_dir, metrics, temperature=None):
    scale_cfg = not args.no_scale_cfg
    n_spe_tokens = args.token_number_per_step  # special tokens, include one begin_of_video, and (args.token_number_per_step-1) middle_of_video
    n_shift = gpt.cond_stage_vocab_size + n_spe_tokens  # because of the cond_stage_vocab and special tokens, the idx of video tokens shift by n_shift

    if args.inference_type == "image":
        latent_shape = [args.resolution // 8, args.resolution // 8]

    else:
        latent_shape = [
            (args.sequence_length - 1) // 4 + 1, args.resolution // 8, args.resolution // 8
        ]

    steps = np.prod(latent_shape) // args.token_number_per_step
    hybrid_steps = steps + n_spe_tokens - 1  # use ntp for the first args.token_number_per_step tokens, then use nbp

    loader = iter(loader)
    num_batches = args.n_sample // (utils.get_world_size() * args.batch_size)
    if args.n_sample % (utils.get_world_size() * args.batch_size) != 0:
        num_batches += 1

    raw_video_batch = []
    pred_video_batch = []

    for batch_id in tqdm(range(num_batches)):
        batch = next(loader)
        raw_video = batch["video"].to(tokenizer.device)
        c = batch[gpt.cond_stage_key]
        _, c_indices = gpt.encode_to_c(c)
        c_indices = c_indices.long().to(gpt.device)

        if args.cfg_ratio is None:  # do not use cfg
            _, bov = gpt.bov_provider.encode(c_indices)
            c_indices += n_spe_tokens  # shift cond token ids because of special tokens
            x = torch.cat((c_indices, bov), dim=1)
            seq_len = x.shape[-1]
            attn_mask = gpt.get_attn_mask(c_indices, seq_len)

            index_sample = hybrid_sample_with_past(x, attn_mask, gpt.transformer, steps=hybrid_steps,
                                            sample_logits=True, top_k=args.top_k, callback=None,
                                            temperature=temperature, top_p=args.top_p, pred_token_num_per_step=args.token_number_per_step)
        else:
            index_sample = hybrid_sample_with_past_cfg(x, gpt, steps=hybrid_steps,
                                                    sample_logits=True, top_k=args.top_k, callback=None,
                                                    temperature=temperature, top_p=args.top_p, cfg_ratio=args.cfg_ratio, scale_cfg=scale_cfg
            )

        index = torch.clamp(index_sample-n_shift, min=0, max=gpt.first_stage_vocab_size-1)  # re-shift token ids
        pred_video = decode(args.tokenizer, tokenizer, index)

        if args.save_video:
            if args.inference_type == "video":
                for i, (in_video, sample, class_label) in enumerate(zip(raw_video, pred_video, c)):
                    video_id = batch_id * args.batch_size + i
                    os.makedirs(os.path.join(save_dir, "gen"), exist_ok=True)
                    os.makedirs(os.path.join(save_dir, "input"), exist_ok=True)
                    if isinstance(class_label, str):
                        class_label = class_label.replace(' ', '_')
                    save_video_grid(sample.unsqueeze(0), os.path.join(save_dir, "gen", '%s_%d.mp4' % (class_label, video_id)), 1)
                    save_video_grid(in_video.unsqueeze(0), os.path.join(save_dir, "input", '%s_%d.mp4' % (class_label, video_id)), 1)
            else:
                images_batch = torch.clamp(pred_video, 0, 1)
                if images_batch.ndim == 5:
                    images_batch = images_batch.squeeze(2)  # squeeze T dimension
                for i, (img, class_label) in enumerate(zip(images_batch, c)):  # todo
                    img = (img.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
                    img = Image.fromarray(img).convert("RGB")
                    image_id = batch_id * args.batch_size + i
                    if isinstance(class_label, str):
                        class_label = class_label.strip().replace('-', '_').replace(',', '').replace(' ', '_').replace('|', '--').replace('/', 'or').strip('-_')[:128]
                    save_path = os.path.join(save_dir, '%s_%d.png'%(class_label, image_id))
                    img.save(save_path)

        raw_video_batch.append(raw_video)
        pred_video_batch.append(pred_video)

        if len(raw_video_batch) == 16:
            raw_video = torch.cat(raw_video_batch, dim=0)
            pred_video = torch.cat(pred_video_batch, dim=0)

            metrics.update_metrics(raw_video, pred_video)

            # reset
            raw_video_batch = []
            pred_video_batch = []

    if len(raw_video_batch) > 0:  # process the remaining videos
        raw_video = torch.cat(raw_video_batch, dim=0)
        pred_video = torch.cat(pred_video_batch, dim=0)

        metrics.update_metrics(raw_video, pred_video)

    metrics.merge_metrics(args.distributed, args.n_sample)


@torch.no_grad()
def frame_prediction(loader, gpt, tokenizer, args, save_dir, metrics):
    if args.inference_type == "image":
        latent_shape = [args.resolution // 8, args.resolution // 8]

    else:
        latent_shape = [
            (args.sequence_length - 1) // 4 + 1, args.resolution // 8, args.resolution // 8
        ]

    latent_L = (args.resolution // 8) ** 2
    steps = int((latent_shape[0] - 2) * (latent_L // args.token_number_per_step))

    sample_logits = True
    loader = iter(loader)
    num_batches = args.n_sample // (utils.get_world_size() * args.batch_size)
    if args.n_sample % (utils.get_world_size() * args.batch_size) != 0:
        num_batches += 1

    raw_video_batch = []
    pred_video_batch = []

    for _ in tqdm(range(num_batches)):
        batch = next(loader)
        raw_video = batch["video"].to(tokenizer.device)
        prefix_encodings = encode(args.tokenizer, tokenizer, raw_video)
        prefix_encodings = prefix_encodings[:, :2]  # take the first 5 frames as condition (latent_T=2)
        B, _, H, W = prefix_encodings.shape
        prefix_encodings = prefix_encodings.view(B, -1)

        c = batch[gpt.cond_stage_key]
        _, c_indices = gpt.encode_to_c(c)
        c_indices = c_indices.long().to(gpt.device)
        x = torch.cat([c_indices, prefix_encodings], dim=-1)
        seq_len = x.shape[1]
        attn_mask = gpt.get_attn_mask(c_indices, seq_len)

        index_sample = sample_with_past(x, attn_mask, gpt.transformer, steps=steps, sample_logits=sample_logits, top_k=args.top_k, temperature=1.0, top_p=args.top_p, pred_token_num_per_step=args.token_number_per_step)

        index = torch.clamp(index_sample, min=0, max=gpt.first_stage_vocab_size-1)
        index = torch.cat((prefix_encodings, index), dim=1)
        index = rearrange(index, "b (t h w) -> b t h w", h=H, w=W)
        pred_video = decode(args.tokenizer, tokenizer, index)

        if args.save_video:
            paths = batch["path"]
            for in_video, sample, path in zip(raw_video, pred_video, paths):
                video_class = os.path.basename(os.path.dirname(path))
                video_name = os.path.basename(path)
                os.makedirs(os.path.join(save_dir, "gen"), exist_ok=True)
                os.makedirs(os.path.join(save_dir, "input"), exist_ok=True)
                save_video_grid(sample.unsqueeze(0), os.path.join(save_dir, "gen", video_class + "_" + video_name), 1)
                save_video_grid(in_video.unsqueeze(0), os.path.join(save_dir, "input", video_class + "_" + video_name), 1)

        if args.eval_resolution != args.resolution:
            raw_video = torch.stack([preprocess_single(video, resolution=args.eval_resolution, normalize=False) for video in raw_video])
            pred_video = torch.stack([preprocess_single(video, resolution=args.eval_resolution, normalize=False) for video in pred_video])

        raw_video_batch.append(raw_video)
        pred_video_batch.append(pred_video)

        if len(raw_video_batch) == 16:
            raw_video = torch.cat(raw_video_batch, dim=0)
            pred_video = torch.cat(pred_video_batch, dim=0)

            metrics.update_metrics(raw_video, pred_video)

            # reset
            raw_video_batch = []
            pred_video_batch = []

    if len(raw_video_batch) > 0:  # process the remaining videos
        raw_video = torch.cat(raw_video_batch, dim=0)
        pred_video = torch.cat(pred_video_batch, dim=0)

        metrics.update_metrics(raw_video, pred_video)

    metrics.merge_metrics(args.distributed, args.n_sample)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer', type=str, default='magvit2')
    parser.add_argument('--gpt_ckpt', type=str, default='')
    parser.add_argument('--tokenizer_ckpt', type=str, default='')
    parser.add_argument('--llama_tokenizer', type=str, default='llama-2-7b-chat-hf')
    parser.add_argument('--inference_type', type=str, default='image', choices=["image", "video"])
    parser.add_argument('--save', type=str, default='./results/tats')
    parser.add_argument('--top_k', type=int, default=2048)
    parser.add_argument('--top_p', type=float, default=0.92)
    parser.add_argument('--n_sample', type=int, default=1000*50)
    parser.add_argument('--data_dir', type=str, default='ucf101')
    parser.add_argument("--data_list", type=str)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument('--class_cond', action='store_true')
    parser.add_argument('--cfg_ratio', type=float, default=None)
    parser.add_argument('--no_scale_cfg', action="store_true")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument('--semi_ar', action='store_true')
    parser.add_argument('--token_number_per_step', type=int, default=16)
    parser.add_argument('--resolution', type=int, default=128)
    parser.add_argument('--eval_resolution', type=int, default=64)  # use 64x64 for k600, see magvit
    parser.add_argument('--save_video', action='store_true')
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--frm_sampling_strategy', type=str, default='rand')
    parser.add_argument('--hybrid', action="store_true")
    parser.add_argument('--sequence_length', type=int, default=17)
    parser.add_argument(
        "--world_size",
        default=1,
        type=int,
        help="number of distributed processes",
    )
    parser.add_argument(
        "--dist_url",
        default="env://",
        help="url used to set up distributed training",
    )
    parser.add_argument("--distributed", default=False, type=bool)
    args = parser.parse_args()

    utils.init_distributed_mode(args)

    gpt, epoch, global_step = load_transformer(args)
    gpt = gpt.cuda().eval()
    if args.tokenizer == "magvit2":
        tokenizer = load_magvit2(args.tokenizer_ckpt, args.resolution, device="cuda")
        tokenizer.eval()
    else:
        raise ValueError('tokenizer %s is not implementated' % args.tokenizer)
    i3d_model = load_i3d_pretrained(device="cuda")
    # load perceptual model
    perceptual_model = LPIPS().eval()
    perceptual_model.cuda(torch.cuda.current_device())
    # Initialize the Metrics class
    metrics = Metrics(i3d_model, perceptual_model, is_image=args.inference_type == "image")

    device = torch.device("cuda")

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    args.save = f"{args.save}_ep{epoch}_step{global_step}"
    cfg_ratio = -1 if args.cfg_ratio is None else args.cfg_ratio
    save_dir = '%s/topp%.2f_topk%d_cfg%.1f' % (args.save, args.top_p, args.top_k, cfg_ratio)
    if args.top_p == -1:
        args.top_p = None
    if args.top_k == -1:
        args.top_k = None

    if utils.get_rank() == 0:
        print('generating and saving video to %s...'%save_dir)
        os.makedirs(save_dir, exist_ok=True)

    # load models
    if args.distributed:
        gpt = torch.nn.parallel.DistributedDataParallel(
            gpt, device_ids=[args.gpu], find_unused_parameters=True
        )

        tokenizer = torch.nn.parallel.DistributedDataParallel(
            tokenizer, device_ids=[args.gpu], find_unused_parameters=True
        )
        i3d_model = torch.nn.parallel.DistributedDataParallel(
            i3d_model, device_ids=[args.gpu], find_unused_parameters=True
        )
        gpt_potential_module = gpt.module
        tokenizer_potential_module = tokenizer.module
        i3d_model_potential_module = i3d_model.module
    else:
        gpt_potential_module = gpt
        tokenizer_potential_module = tokenizer
        i3d_model_potential_module = i3d_model

    # prepare data
    if args.inference_type == 'image':
        dataset = ImageDataset(
            args.data_dir, args.data_list, train=False, resolution=args.resolution, tokenizer=args.tokenizer
        )
    else:
        dataset = DecordVideoDataset(
            args.data_dir, args.data_list, fps=args.fps, sequence_length=args.sequence_length, train=False, resolution=args.resolution, tokenizer=args.tokenizer, frm_sampling_strategy=args.frm_sampling_strategy
        )

    if args.distributed:
        sampler = data.distributed.DistributedSampler(
            dataset, num_replicas=utils.get_world_size(), rank=utils.get_rank()
        )
    else:
        sampler = None

    dataloader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=False,
        sampler=sampler,
        shuffle=False
    )
    dataloader = itertools.cycle(dataloader)

    # evaluation
    if not args.class_cond:
        frame_prediction(
            dataloader, gpt_potential_module, tokenizer_potential_module, args, save_dir, metrics
        )
    else:
        class_condition_generation(
            dataloader, gpt_potential_module, tokenizer_potential_module, args, save_dir, metrics, temperature=1.0
        )
