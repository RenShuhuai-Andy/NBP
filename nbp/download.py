import requests
from tqdm import tqdm
import os
import torch

from .lm_transformer import Net2NetTransformer, HybridNet2NetTransformer

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 8192

    pbar = tqdm(total=0, unit='iB', unit_scale=True)
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    pbar.close()


def download(id, fname, root=os.path.expanduser('./ckpts')):
    os.makedirs(root, exist_ok=True)
    destination = os.path.join(root, fname)

    if os.path.exists(destination):
        return destination

    URL = 'https://drive.google.com/uc?export=download'
    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    save_response_content(response, destination)
    return destination


def load_transformer(args, device=torch.device('cpu')):
    gpt_ckpt, tokenizer_ckpt, llama_tokenizer, hybrid = args.gpt_ckpt, args.tokenizer_ckpt, args.llama_tokenizer, args.hybrid
    if hybrid:
        gpt = HybridNet2NetTransformer.load_from_checkpoint(gpt_ckpt, ckpt_path=gpt_ckpt, tokenizer_path=tokenizer_ckpt, llama_tokenizer=llama_tokenizer, strict=False, map_location='cpu').to(device)
    else:
        gpt = Net2NetTransformer.load_from_checkpoint(gpt_ckpt, ckpt_path=gpt_ckpt, tokenizer_path=tokenizer_ckpt, llama_tokenizer=llama_tokenizer, strict=False, map_location='cpu').to(device)
    ckpt = torch.load(gpt_ckpt, map_location='cpu')
    total_params = sum(p.numel() for p in ckpt['state_dict'].values())
    print(f"total_params: {total_params}")
    epoch = ckpt['epoch']
    global_step = ckpt['global_step']
    print(f"Load Transformer weights from {gpt_ckpt} @ ep {epoch}, global step {global_step}.")
    del ckpt
    gpt.eval()

    return gpt, epoch, global_step


_I3D_PRETRAINED_ID = '1mQK8KD8G6UWRa5t87SRMm5PVXtlpneJT'

def load_i3d_pretrained(device=torch.device('cpu')):
    from .fvd.pytorch_i3d import InceptionI3d
    i3d = InceptionI3d(400, in_channels=3).to(device)
    filepath = download(_I3D_PRETRAINED_ID, 'i3d_pretrained_400.pt')
    i3d.load_state_dict(torch.load(filepath, map_location=device))
    i3d.eval()
    return i3d

def load_magvit2(magvit2_ckpt, resolution=128, device=torch.device('cpu')):
    print(f"Load magvit2 weights from {magvit2_ckpt}.")
    from .magvit2 import VideoTokenizer
    from collections import OrderedDict
    video_tokenizer = VideoTokenizer(
        num_codebooks=1,
        image_size=resolution,
        embed_dim=6,
        codebook_size=64000,
        fsq_levels=[8, 8, 8, 5, 5, 5],
        use_fsq=True,
        flash_attn=True,
    )
    state_dict = torch.load(magvit2_ckpt, map_location='cpu')['model_state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    # load params
    video_tokenizer.load_state_dict(new_state_dict)
    video_tokenizer.eval()
    return video_tokenizer.to(device)

