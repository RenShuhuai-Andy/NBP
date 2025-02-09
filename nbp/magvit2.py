import copy
from pathlib import Path
from math import log2, ceil, sqrt
from functools import wraps, partial

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList
from torch.autograd import grad as torch_grad

import torchvision
from torchvision.models import VGG16_Weights

from collections import namedtuple

# from vector_quantize_pytorch import LFQ, FSQ
from .quantizers import VectorQuantizeEMA, FSQ, LFQ
from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange

from beartype import beartype
from beartype.typing import Union, Tuple, Optional, List

from .quantizers.attend import Attend
from .quantizers.version import __version__

import pickle


# helper

def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


def safe_get_index(it, ind, default=None):
    if ind < len(it):
        return it[ind]
    return default


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def identity(t, *args, **kwargs):
    return t


def divisible_by(num, den):
    return (num % den) == 0


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


def append_dims(t, ndims: int):
    return t.reshape(*t.shape, *((1,) * ndims))


def is_odd(n):
    return not divisible_by(n, 2)


def maybe_del_attr_(o, attr):
    if hasattr(o, attr):
        delattr(o, attr)


def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) else ((t,) * length)


# tensor helpers

def l2norm(t):
    return F.normalize(t, dim=-1, p=2)


def pad_at_dim(t, pad, dim=-1, value=0.):
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value=value)


def pick_video_frame(video, frame_indices):
    batch, device = video.shape[0], video.device
    video = rearrange(video, 'b c f ... -> b f c ...')
    batch_indices = torch.arange(batch, device=device)
    batch_indices = rearrange(batch_indices, 'b -> b 1')
    images = video[batch_indices, frame_indices]
    images = rearrange(images, 'b 1 c ... -> b c ...')
    return images


class Swish(nn.Module):
    def __init__(self, num_features):
        """
            num_features: int, the number of input feature dimensions.
        """
        super(Swish, self).__init__()
        shape = (1, num_features) + (1, ) * 3
        self.beta = nn.Parameter(torch.Tensor(*shape))
        self.reset_parameters()

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

    def reset_parameters(self):
        nn.init.ones_(self.beta)


class GroupNorm3D(nn.GroupNorm):
    def forward(self, x):
        """
        Args:
        - x (torch.Tensor): Input tensor of shape [b, c, t, h, w].
        """
        b = x.shape[0]
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = super().forward(x)
        x = rearrange(x, "(b t) c h w -> b c t h w", b=b)
        return x

# gan related


def gradient_penalty(images, output):
    batch_size = images.shape[0]

    gradients = torch_grad(
        outputs=output,
        inputs=images,
        grad_outputs=torch.ones(output.size(), device=images.device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = rearrange(gradients, 'b ... -> b (...)')
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()


def leaky_relu(p=0.1):
    return nn.LeakyReLU(p)


def hinge_discr_loss(fake, real):
    return (F.relu(1 + fake) + F.relu(1 - real)).mean()


def hinge_gen_loss(fake):
    return -fake.mean()


@autocast(enabled=False)
@beartype
def grad_layer_wrt_loss(
        loss: Tensor,
        layer: nn.Parameter
):
    return torch_grad(
        outputs=loss,
        inputs=layer,
        grad_outputs=torch.ones_like(loss),
        retain_graph=True
    )[0].detach()


# helper decorators

def remove_vgg(fn):
    @wraps(fn)
    def inner(self, *args, **kwargs):
        has_vgg = hasattr(self, 'vgg')
        if has_vgg:
            vgg = self.vgg
            delattr(self, 'vgg')

        out = fn(self, *args, **kwargs)

        if has_vgg:
            self.vgg = vgg

        return out

    return inner


# helper classes

def Sequential(*modules):
    modules = [*filter(exists, modules)]

    if len(modules) == 0:
        return nn.Identity()

    return nn.Sequential(*modules)


class Residual(Module):
    @beartype
    def __init__(self, fn: Module, residual_fn=None):
        super().__init__()
        self.fn = fn
        self.residual_fn = default(residual_fn, nn.Identity())

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + self.residual_fn(x)




class Depth2Space3D(Module):
    # https://discuss.pytorch.org/t/is-there-any-layer-like-tensorflows-space-to-depth-function/3487/15
    def __init__(self, block_size):
        """
        Initialize DepthToSpace module.
        
        Args:
        - block_size (tuple): Block size for depth to space operation in the format (time_block_size, height_block_size, width_block_size).
        """
        super().__init__()
        self.block_size = block_size

    def forward(self, x):
        """
        Perform depth to space operation on 3D video data.
        
        Args:
        - x (torch.Tensor): Input tensor of shape [b, c, t, h, w].
        
        Returns:
        - output_tensor (torch.Tensor): Output tensor after depth to space operation.
        """
        b, c, t, h, w = x.shape
        t_bs, h_bs, w_bs = self.block_size

        assert c % (t_bs * h_bs * w_bs) == 0, "Channels must be divisible by block_size"

        c_out = c // (t_bs * h_bs * w_bs)
        t_out = t * t_bs
        h_out = h * h_bs
        w_out = w * w_bs

        x = x.view(b, t_bs, h_bs, w_bs, c_out, t, h, w)
        x = x.permute(0, 4, 5, 1, 6, 2, 7, 3).contiguous()  # [b, c_out, t, t_bs, h, h_bs, w, w_bs]
        x = x.view(b, c_out, t_out, h_out, w_out)

        if t_bs > 1:  # causal time upsample, the first frame is not upsampled
            x = x[:, :, t_bs - 1:, :, :]

        return x



class CausalConv3d(Module):
    @beartype
    def __init__(
            self,
            chan_in,
            chan_out,
            kernel_size: Union[int, Tuple[int, int, int]],
            stride: Union[int, Tuple[int, int, int]],
            pad_mode='replicate',  # todo
            **kwargs
    ):
        super().__init__()
        kernel_size = cast_tuple(kernel_size, 3)
        stride = cast_tuple(stride, 3)

        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size
        time_stride, height_stride, width_stride = stride
        assert is_odd(height_kernel_size) and is_odd(width_kernel_size)

        self.pad_mode = pad_mode
        time_pad = time_kernel_size // 2
        height_pad = height_kernel_size // 2
        width_pad = width_kernel_size // 2

        self.time_pad = time_pad
        self.time_causal_padding = (width_pad, width_pad, height_pad, height_pad, time_pad*2, 0)

        self.conv = nn.Conv3d(chan_in, chan_out, kernel_size, stride=stride, **kwargs)

    def forward(self, x):
        pad_mode = self.pad_mode if self.time_pad < x.shape[2] else 'replicate'  # todo

        x = F.pad(x, self.time_causal_padding, mode=pad_mode)
        return self.conv(x)


@beartype
def ResidualUnit(
        dim_in,
        dim_out,
        kernel_size: Union[int, Tuple[int, int, int]],
        pad_mode: str = 'replicate',
):
    net = Sequential(
        GroupNorm3D(32, dim_in),
        Swish(dim_in),
        CausalConv3d(dim_in, dim_out, kernel_size, stride=(1, 1, 1), pad_mode=pad_mode),
        GroupNorm3D(32, dim_out),
        Swish(dim_out),
        CausalConv3d(dim_out, dim_out, kernel_size, stride=(1, 1, 1), pad_mode=pad_mode),
    )

    if dim_in != dim_out:
        residual_net = Sequential(
            CausalConv3d(dim_in, dim_out, (1, 1, 1), stride=(1, 1, 1), pad_mode=pad_mode),
        )
    else:
        residual_net = None

    return Residual(net, residual_net)


class VideoTokenizer(Module):
    @beartype
    def __init__(
            self,
            *,
            image_size,
            layers: Tuple[Union[str, Tuple[str, int]], ...] = (
                    'residual',
                    'residual',
                    'residual'
            ),
            residual_conv_kernel_size=(3, 3, 3),
            num_codebooks=1,
            codebook_size: Optional[int] = None,
            channels=3,
            embed_dim=5,
            dim_cond=None,
            dim_cond_expansion_factor=4.,
            input_conv_kernel_size: Tuple[int, int, int] = (3, 3, 3),
            output_conv_kernel_size: Tuple[int, int, int] = (3, 3, 3),
            pad_mode: str = 'replicate',
            lfq_entropy_loss_weight=0.1,
            lfq_commitment_loss_weight=0.25,
            lfq_diversity_gamma=2.5,
            quantizer_aux_loss_weight=1.,
            lfq_activation=nn.Identity(),
            use_fsq=False,
            fsq_levels: Optional[List[int]] = None,
            attn_dim_head=32,
            attn_heads=8,
            attn_dropout=0.,
            linear_attn_dim_head=8,
            linear_attn_heads=16,
            flash_attn=True,
    ):
        super().__init__()

        # for autosaving the config

        _locals = locals()
        _locals.pop('self', None)
        _locals.pop('__class__', None)
        self._configs = pickle.dumps(_locals)

        # image size

        self.channels = channels
        self.image_size = image_size

        # encoder and decoder layers

        self.encoder_layers = ModuleList([
            CausalConv3d(channels, 128, (3, 3, 3), stride=(1, 1, 1), pad_mode=pad_mode),
            *[ResidualUnit(128, 128, residual_conv_kernel_size) for _ in range(4)],
            CausalConv3d(128, 128, (3, 3, 3), stride=(1, 2, 2), pad_mode=pad_mode),
            ResidualUnit(128, 256, residual_conv_kernel_size),
            *[ResidualUnit(256, 256, residual_conv_kernel_size) for _ in range(3)],
            CausalConv3d(256, 256, (3, 3, 3), stride=(2, 2, 2), pad_mode=pad_mode),
            *[ResidualUnit(256, 256, residual_conv_kernel_size) for _ in range(4)],
            CausalConv3d(256, 256, (3, 3, 3), stride=(2, 2, 2), pad_mode=pad_mode),
            ResidualUnit(256, 512, residual_conv_kernel_size),
            *[ResidualUnit(512, 512, residual_conv_kernel_size) for _ in range(3)],
            *[ResidualUnit(512, 512, residual_conv_kernel_size) for _ in range(4)],
            GroupNorm3D(32, 512),
            Swish(512),
            nn.Conv3d(512, embed_dim, (1, 1, 1))
        ])
        self.decoder_layers = ModuleList([
            CausalConv3d(embed_dim, 512, (3, 3, 3), stride=(1, 1, 1), pad_mode=pad_mode),
            *[ResidualUnit(512, 512, residual_conv_kernel_size) for _ in range(4)],
            # todo Adaptive GroupNorm
            *[ResidualUnit(512, 512, residual_conv_kernel_size) for _ in range(4)],
            CausalConv3d(512, 4096, (3, 3, 3), stride=(1, 1, 1), pad_mode=pad_mode),
            Depth2Space3D((2, 2, 2)),
            # todo Adaptive GroupNorm
            ResidualUnit(512, 256, residual_conv_kernel_size),
            *[ResidualUnit(256, 256, residual_conv_kernel_size) for _ in range(3)],
            CausalConv3d(256, 2048, (3, 3, 3), stride=(1, 1, 1), pad_mode=pad_mode),
            Depth2Space3D((2, 2, 2)),
            # todo Adaptive GroupNorm
            *[ResidualUnit(256, 256, residual_conv_kernel_size) for _ in range(4)],
            CausalConv3d(256, 1024, (3, 3, 3), stride=(1, 1, 1), pad_mode=pad_mode),
            Depth2Space3D((1, 2, 2)),
            # todo Adaptive GroupNorm
            ResidualUnit(256, 128, residual_conv_kernel_size),
            *[ResidualUnit(128, 128, residual_conv_kernel_size) for _ in range(3)],
            GroupNorm3D(32, 128),
            Swish(128),
            CausalConv3d(128, channels, (3, 3, 3), stride=(1, 1, 1), pad_mode=pad_mode),  # nn.Conv3d(128, channels, (3, 3, 3)) todo
        ])

        layer_fmap_size = image_size // 8
        time_downsample_factor = 1
        has_cond_across_layers = []
        has_cond = False

        self.time_downsample_factor = time_downsample_factor
        self.time_padding = time_downsample_factor - 1

        self.fmap_size = layer_fmap_size

        # use a MLP stem for conditioning, if needed

        self.has_cond_across_layers = has_cond_across_layers
        self.has_cond = any(has_cond_across_layers)

        self.encoder_cond_in = nn.Identity()
        self.decoder_cond_in = nn.Identity()

        if has_cond:
            self.dim_cond = dim_cond

            self.encoder_cond_in = Sequential(
                nn.Linear(dim_cond, int(dim_cond * dim_cond_expansion_factor)),
                nn.SiLU()
            )

            self.decoder_cond_in = Sequential(
                nn.Linear(dim_cond, int(dim_cond * dim_cond_expansion_factor)),
                nn.SiLU()
            )

        # quantizer related

        self.use_fsq = use_fsq

        if not use_fsq:
            assert exists(codebook_size), 'if use_fsq is set to False, `codebook_size` must be set'

            # lookup free quantizer(s) - multiple codebooks is possible
            # each codebook will get its own entropy regularization

            # self.quantizers = LFQ(
            #     dim=embed_dim,
            #     codebook_size=codebook_size,
            #     num_codebooks=num_codebooks,
            #     entropy_loss_weight=lfq_entropy_loss_weight,
            #     commitment_loss_weight=lfq_commitment_loss_weight,
            #     diversity_gamma=lfq_diversity_gamma
            # )
            self.quantizers = LFQ(
                dim=embed_dim,
                codebook_size=codebook_size,
                num_codebooks=num_codebooks,
                sample_minimization_weight=1.0,
                batch_maximization_weight=1.0,
                token_factorization=False
            )

        else:
            assert exists(fsq_levels), 'if use_fsq is set to True, `fsq_levels` must be set. the effective codebook size is the cumulative product of all the FSQ levels'

            self.quantizers = FSQ(
                fsq_levels,
                dim=embed_dim,
                num_codebooks=num_codebooks
            )

        self.quantizer_aux_loss_weight = quantizer_aux_loss_weight

        # dummy loss

        self.register_buffer('zero', torch.tensor(0.), persistent=False)

    @property
    def device(self):
        return self.zero.device

    @classmethod
    def init_and_load_from(cls, path, strict=True):
        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path), map_location='cpu')

        assert 'config' in pkg, 'model configs were not found in this saved checkpoint'

        config = pickle.loads(pkg['config'])
        tokenizer = cls(**config)
        tokenizer.load(path, strict=strict)
        return tokenizer

    def parameters(self):
        return [
            *self.encoder_layers.parameters(),
            *self.decoder_layers.parameters(),
            *self.encoder_cond_in.parameters(),
            *self.decoder_cond_in.parameters(),
            *self.quantizers.parameters()
        ]


    def copy_for_eval(self):
        device = self.device
        vae_copy = copy.deepcopy(self.cpu())

        maybe_del_attr_(vae_copy, 'discr')
        maybe_del_attr_(vae_copy, 'vgg')
        maybe_del_attr_(vae_copy, 'multiscale_discrs')

        vae_copy.eval()
        return vae_copy.to(device)

    @remove_vgg
    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)

    @remove_vgg
    def load_state_dict(self, *args, **kwargs):
        return super().load_state_dict(*args, **kwargs)

    def save(self, path, overwrite=True):
        path = Path(path)
        assert overwrite or not path.exists(), f'{str(path)} already exists'

        pkg = dict(
            model_state_dict=self.state_dict(),
            version=__version__,
            config=self._configs
        )

        torch.save(pkg, str(path))

    def load(self, path, strict=True):
        path = Path(path)
        assert path.exists()

        pkg = torch.load(str(path))
        state_dict = pkg.get('model_state_dict')
        version = pkg.get('version')

        assert exists(state_dict)

        if exists(version):
            print(f'loading checkpointed tokenizer from version {version}')

        self.load_state_dict(state_dict, strict=strict)

    @beartype
    def encode(
            self,
            video: Tensor,
            quantize=False,
            cond: Optional[Tensor] = None,
    ):
        # whether to pad video or not

        # conditioning, if needed

        assert (not self.has_cond) or exists(
            cond), '`cond` must be passed into tokenizer forward method since conditionable layers were specified'

        if exists(cond):
            assert cond.shape == (video.shape[0], self.dim_cond)

            cond = self.encoder_cond_in(cond)
            cond_kwargs = dict(cond=cond)

        # encoder layers

        for fn in self.encoder_layers:
            layer_kwargs = dict()
            video = fn(video, **layer_kwargs)  # final: [bsz, embed_dim, (T+self.time_padding)/2/2, H/2/2/2, W/2/2/2]

        maybe_quantize = identity if not quantize else self.quantizers

        return maybe_quantize(video)

    @beartype
    def decode_from_code_indices(
            self,
            codes: Tensor,
            cond: Optional[Tensor] = None,
    ):
        assert codes.dtype in (torch.long, torch.int32)

        if codes.ndim == 2:
            video_code_len = codes.shape[-1]
            assert divisible_by(video_code_len,
                                self.fmap_size ** 2), f'flattened video ids must have a length ({video_code_len}) that is divisible by the fmap size ({self.fmap_size}) squared ({self.fmap_size ** 2})'

            codes = rearrange(codes, 'b (f h w) -> b f h w', h=self.fmap_size, w=self.fmap_size)

        quantized = self.quantizers.indices_to_codes(codes)

        return self.decode(quantized, cond=cond)

    @beartype
    def decode(
            self,
            quantized: Tensor,
            cond: Optional[Tensor] = None,
    ):
        batch = quantized.shape[0]

        # conditioning, if needed

        assert (not self.has_cond) or exists(
            cond), '`cond` must be passed into tokenizer forward method since conditionable layers were specified'

        if exists(cond):
            assert cond.shape == (batch, self.dim_cond)

            cond = self.decoder_cond_in(cond)
            cond_kwargs = dict(cond=cond)

        # decoder layers

        x = quantized

        for fn in self.decoder_layers:
            layer_kwargs = dict()
            x = fn(x, **layer_kwargs)

        video = x
        return video

    @torch.no_grad()
    def tokenize(self, video):
        self.eval()
        return self.forward(video, return_codes=True)

    @beartype
    def forward(
            self,
            video_or_images: Tensor,
            cond: Optional[Tensor] = None,
            return_id: bool = False,
    ):
        assert video_or_images.ndim in {4, 5}

        assert video_or_images.shape[-2:] == (self.image_size, self.image_size)

        # accept images for image pretraining (curriculum learning from images to video)

        is_image = video_or_images.ndim == 4

        if is_image:
            video = rearrange(video_or_images, 'b c ... -> b c 1 ...')
            video_contains_first_frame = True
        else:
            video = video_or_images

        batch, channels, frames = video.shape[:3]

        # encoder

        x = self.encode(video, cond=cond)  # [bsz, max_dim, (T-1)/time_downsample_factor+1, image_size/8, image_size/8]

        # lookup free quantization

        if self.use_fsq:
            quantized, codes = self.quantizers(x)

            aux_losses = self.zero
            quantizer_loss_breakdown = None
        else:
            # (quantized, codes, aux_losses), quantizer_loss_breakdown = self.quantizers(x,
            #                                                                            return_loss_breakdown=True)  # codes: [bsz, (T-1)/time_downsample_factor+1, image_size/8, image_size/8]
            (quantized, aux_losses, codes), quantizer_loss_breakdown = self.quantizers(x, return_loss_breakdown=True)
        # decoder

        recon_video = self.decode(quantized, cond=cond)

        if return_id:
            return recon_video, aux_losses, codes
        else:
            return recon_video, aux_losses, quantizer_loss_breakdown

    def get_last_layer(self):
        return self.decoder_layers[-1].conv.weight

# main class

class MagViT2(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x