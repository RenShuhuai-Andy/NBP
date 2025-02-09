"""
taken from: https://github.com/karpathy/minGPT/
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F
# from transformers import top_k_top_p_filtering

def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)


class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.semi_ar = True if hasattr(config, "semi_ar") else False
        self.n_head = config.n_head

        self.p_attn_drop = config.attn_pdrop

    def forward(self, x, attn_mask, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        present = torch.stack((k, v))
        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)
            T_q = T
            T_v = v.shape[-2]
            attn_mask = torch.ones((1, 1, T_q, T_v)).to(x.device)
        else:
            attn_mask = attn_mask[:, :, :T, :T]
        
        if hasattr(F, "scaled_dot_product_attention") and torch.__version__ >= "2.1.0":
            # query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None
            is_causal = (layer_past is None) and not self.semi_ar
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.p_attn_drop, is_causal=is_causal, attn_mask=attn_mask.bool())
        else:
            # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if layer_past is None:
                att = att.masked_fill(attn_mask == 0, float('-inf'))

            att = F.softmax(att, dim=-1)
            att = self.attn_drop(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, present   # TODO: check that this does not break anything


class Block(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),  # nice
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, attn_mask, layer_past=None, return_present=False):
        # TODO: check that training still works
        if return_present: assert not self.training
        # layer past: tuple of length two with B, nh, T, hs
        attn, present = self.attn(self.ln1(x), attn_mask, layer_past=layer_past)

        x = x + attn
        x = x + self.mlp(self.ln2(x))
        if layer_past is not None or return_present:
            return x, present
        return x


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """
    def __init__(self, args, vocab_size, block_size, n_layer=12, n_head=8, n_embd=256,
                 embd_pdrop=0., resid_pdrop=0., attn_pdrop=0., n_unmasked=0, vtokens_pos=False, semi_ar=False, token_number_per_step=1, cond_token_num=0):
        super().__init__()
        config = GPTConfig(vocab_size=vocab_size, block_size=block_size,
                           embd_pdrop=embd_pdrop, resid_pdrop=resid_pdrop, attn_pdrop=attn_pdrop,
                           n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                           n_unmasked=n_unmasked, semi_ar=semi_ar, token_number_per_step=token_number_per_step)
        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.vtokens_pos = vtokens_pos
        if self.vtokens_pos:
            self.vtokens_pos_emb = nn.Parameter(torch.zeros(1, args.sequence_length, args.resolution, args.resolution, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        self.register_buffer("uncond_embedding", nn.Parameter(torch.randn(cond_token_num, config.n_embd) / config.n_embd ** 0.5))
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.block_size = config.block_size
        self.apply(self._init_weights)
        self.config = config

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, attn_mask, embeddings=None, targets=None, cbox=None, tbox=None):
        # forward the GPT model
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector

        if embeddings is not None: # prepend explicit embeddings
            token_embeddings = torch.cat((embeddings, token_embeddings), dim=1)

        t = token_embeddings.shape[1]
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        # import pdb;pdb.set_trace()
        if self.vtokens_pos:
            if tbox:
                vtokens_position_embeddings = torch.cat([self.vtokens_pos_emb[:, tpos[0]:tpos[1], pos[0]:pos[1], pos[2]:pos[3], :].reshape(1, -1, self.vtokens_pos_emb.shape[-1]) for pos, tpos in zip(cbox, tbox)], 0)
            else:
                vtokens_position_embeddings = torch.cat([self.vtokens_pos_emb[:, :, pos[0]:pos[1], pos[2]:pos[3], :].reshape(1, -1, self.vtokens_pos_emb.shape[-1]) for pos in cbox], 0)
            position_embeddings = position_embeddings + vtokens_position_embeddings
        assert token_embeddings.shape[1:] == position_embeddings.shape[1:], f'token_embeddings.shape: {token_embeddings.shape}, position_embeddings.shape: {position_embeddings.shape}'
        x = self.drop(token_embeddings + position_embeddings)

        for i, block in enumerate(self.blocks):
            x = block(x, attn_mask)

        x = self.ln_f(x)
        logits = self.head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def forward_with_past(self, idx, attn_mask, embeddings=None, targets=None, past=None, past_length=None, cbox=None):
        # inference only
        assert not self.training
        token_embeddings = self.tok_emb(idx)    # each index maps to a (learnable) vector
        if embeddings is not None:              # prepend explicit embeddings
            token_embeddings = torch.cat((embeddings, token_embeddings), dim=1)

        if past is not None:
            assert past_length is not None
            past = torch.cat(past, dim=-2)   # n_layer, 2, b, nh, len_past, dim_head
            past_shape = list(past.shape)
            expected_shape = [self.config.n_layer, 2, idx.shape[0], self.config.n_head, past_length, self.config.n_embd//self.config.n_head]
            assert past_shape == expected_shape, f"{past_shape} =/= {expected_shape}"
            seq_len = token_embeddings.shape[1]
            position_embeddings = self.pos_emb[:, past_length:past_length+seq_len, :]
            if self.vtokens_pos:
                vtokens_position_embeddings = torch.cat([self.vtokens_pos_emb[:, :, pos[0]:pos[1], pos[2]:pos[3], :].reshape(1, -1, self.vtokens_pos_emb.shape[-1]) for pos in cbox], 0)
                vtokens_position_embeddings = vtokens_position_embeddings[:, past_length, :]
                position_embeddings = position_embeddings + vtokens_position_embeddings
        else:
            position_embeddings = self.pos_emb[:, :token_embeddings.shape[1], :]
            if self.vtokens_pos:
                vtokens_position_embeddings = torch.cat([self.vtokens_pos_emb[:, :, pos[0]:pos[1], pos[2]:pos[3], :].reshape(1, -1, self.vtokens_pos_emb.shape[-1]) for pos in cbox], 0)
                vtokens_position_embeddings = vtokens_position_embeddings[:, :token_embeddings.shape[1], :]
                position_embeddings = position_embeddings + vtokens_position_embeddings

        assert token_embeddings.shape[1:] == position_embeddings.shape[1:], f'token_embeddings.shape: {token_embeddings.shape}, position_embeddings.shape: {position_embeddings.shape}'
        x = self.drop(token_embeddings + position_embeddings)
        presents = []  # accumulate over layers
        for i, block in enumerate(self.blocks):
            x, present = block(x, attn_mask, layer_past=past[i, ...] if past is not None else None, return_present=True)
            presents.append(present)

        x = self.ln_f(x)
        logits = self.head(x)
        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss, torch.stack(presents)  # _, _, n_layer, 2, b, nh, 1, dim_head


    @torch.no_grad()
    def forward_with_past_and_future(self, idx, attn_mask, idx_future=None, embeddings=None, targets=None, past=None, past_length=None, future_length=None):
        # inference only
        assert not self.training
        if past is None:
            token_embeddings_past = self.tok_emb(idx)
            token_embeddings_future = self.tok_emb(idx_future)
            token_embeddings = torch.cat([token_embeddings_past, token_embeddings_future], dim=1)
        else:
            token_embeddings = self.tok_emb(idx)    # each index maps to a (learnable) vector

        if embeddings is not None:              # prepend explicit embeddings
            token_embeddings = torch.cat((embeddings, token_embeddings), dim=1)

        if past is not None:
            assert past_length is not None and future_length is not None
            past = torch.cat(past, dim=-2)   # n_layer, 2, b, nh, len_past, dim_head
            past_shape = list(past.shape)
            expected_shape = [self.config.n_layer, 2, idx.shape[0], self.config.n_head, past_length+future_length, self.config.n_embd//self.config.n_head]
            assert past_shape == expected_shape, f"{past_shape} =/= {expected_shape}"
            position_embeddings = self.pos_emb[:, past_length, :]  # each position maps to a (learnable) vector
        else:
            position_embeddings_past = self.pos_emb[:, :token_embeddings_past.shape[1], :]
            position_embeddings_future = self.pos_emb[:, -token_embeddings_future.shape[1]:, :]
            position_embeddings = torch.cat([position_embeddings_past, position_embeddings_future], dim=1)

        x = self.drop(token_embeddings + position_embeddings)
        presents = []  # accumulate over layers
        for i, block in enumerate(self.blocks):
            x, present = block(x, attn_mask, layer_past=past[i, ...] if past is not None else None, return_present=True)
            presents.append(present)

        x = self.ln_f(x)
        logits = self.head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss, torch.stack(presents)  # _, _, n_layer, 2, b, nh, 1, dim_head


def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out


@torch.no_grad()
def sample_with_past(x, attn_mask, model, steps, temperature=1., sample_logits=True,
                     top_k=None, top_p=None, callback=None, cbox=None, pred_token_num_per_step=1):
    # x is conditioning
    sample = x
    cond_bov_len = x.shape[1]
    past = None

    for n in range(steps):
        # print(n, x.shape, cond_len)
        if callback is not None:
            callback(n)
        past_length = n*pred_token_num_per_step+cond_bov_len-pred_token_num_per_step
        if cbox is None:
            logits, _, present = model.forward_with_past(x, attn_mask, past=past, past_length=past_length)
        else:
            logits, _, present = model.forward_with_past(x, attn_mask, past=past, past_length=past_length, cbox=cbox)
        if past is None:
            past = [present]
        else:
            past.append(present)
        if pred_token_num_per_step > 1:  # semi-ar
            logits = logits[:, -pred_token_num_per_step:, :] / temperature
        else:
            logits = logits[:, -1, :] / temperature

        if top_k is not None:
            if pred_token_num_per_step > 1:
                for i in range(pred_token_num_per_step):
                    logits[:, i] = top_k_top_p_filtering(logits[:, i], top_k=top_k, top_p=top_p)
            else:
                logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
        
        probs = F.softmax(logits, dim=-1)
        if not sample_logits:
            _, x = torch.topk(probs, k=1, dim=-1)
        else:
            if pred_token_num_per_step > 1:
                x = torch.zeros(x.shape[0], pred_token_num_per_step, dtype=x.dtype, device=x.device)
                for i in range(pred_token_num_per_step):
                    x[:, i] = torch.multinomial(probs[:, i], num_samples=1)
            else:
                x = torch.multinomial(probs, num_samples=1)
        if x.ndim == 3:
            x = x.squeeze(-1)  # [n, seq_len]
        # append to the sequence and continue
        sample = torch.cat((sample, x), dim=1)
    del past
    sample = sample[:, cond_bov_len:]  # cut conditioning off
    return sample


@torch.no_grad()
def hybrid_sample_with_past(x, attn_mask, model, steps, temperature=1., sample_logits=True,
                     top_k=None, top_p=None, callback=None, cbox=None, pred_token_num_per_step=1):
    # x is conditioning
    sample = x
    cond_bov_len = x.shape[1]
    past = None
    past_length = cond_bov_len - 1

    for n in range(steps):
        if n < pred_token_num_per_step:
            cur_pred_token_num_per_step = 1  # use ntp for the first pred_token_num_per_step
        else:
            cur_pred_token_num_per_step = pred_token_num_per_step  # use nbp after pred_token_num_per_step

        # print(n, x.shape, cond_len)
        if callback is not None:
            callback(n)

        if cbox is None:
            logits, _, present = model.forward_with_past(x, attn_mask, past=past, past_length=past_length)
        else:
            logits, _, present = model.forward_with_past(x, attn_mask, past=past, past_length=past_length, cbox=cbox)
        if past is None:
            past = [present]
        else:
            past.append(present)

        logits = logits[:, -cur_pred_token_num_per_step:, :] / temperature

        if top_k is not None:
            if pred_token_num_per_step > 1:
                for i in range(cur_pred_token_num_per_step):
                    logits[:, i] = top_k_top_p_filtering(logits[:, i], top_k=top_k, top_p=top_p)
            else:
                logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
        
        probs = F.softmax(logits, dim=-1)
        if not sample_logits:
            _, x = torch.topk(probs, k=1, dim=-1)
        else:
            x = torch.zeros(x.shape[0], cur_pred_token_num_per_step, dtype=x.dtype, device=x.device)
            for i in range(cur_pred_token_num_per_step):
                x[:, i] = torch.multinomial(probs[:, i], num_samples=1)
        if x.ndim == 3:
            x = x.squeeze(-1)  # [n, seq_len]

        if n == pred_token_num_per_step - 1:  # transfer to nbp
            middle_bos = torch.arange(1, pred_token_num_per_step, device=x.device).long().repeat(x.shape[0], 1)
            x = torch.cat([x, middle_bos], dim=-1)

        sample = torch.cat((sample, x), dim=1)
        past_length += cur_pred_token_num_per_step

    del past
    sample = sample[:, cond_bov_len:]  # cut conditioning off
    middle_bos_len = middle_bos.shape[1]
    sample = torch.cat([sample[:, :pred_token_num_per_step], sample[:, pred_token_num_per_step + middle_bos_len:]], dim=1)  # remove middle bos
    return sample


@torch.no_grad()
def hybrid_sample_with_past_cfg(x, model, steps, temperature=1., sample_logits=True,
                                top_k=None, top_p=None, callback=None, cbox=None, cfg_ratio=1.5, scale_cfg=False):

    pred_token_num_per_step = model.token_number_per_step

    _, bov = model.bov_provider.encode(x)
    x += pred_token_num_per_step  # shift token ids because of bov and middle_bos
    cond_len = x.shape[1]

    # x is conditioning
    sample = torch.cat((x, bov), dim=1)
    x = sample
    cond_bov_len = sample.shape[1]
    past = None

    sample_uncond = bov
    past_uncond = None
    x_uncond = sample_uncond

    attn_mask = model.get_attn_mask(sample, seq_len=cond_bov_len, prefix_len=cond_len)
    past_length = cond_len

    for n in range(steps):
        if n < pred_token_num_per_step:
            cur_pred_token_num_per_step = 1  # use ntp for the first pred_token_num_per_step
        else:
            cur_pred_token_num_per_step = pred_token_num_per_step  # use nbp after pred_token_num_per_step
        
        if n == 0:  # drop the condition, use uncond_embedding in gpt
            embeddings = model.transformer.uncond_embedding
            embeddings = embeddings.unsqueeze(0).repeat(x.shape[0], 1, 1)  # [bsz, cond_token_num, dim]
        else:
            embeddings = None

        ratio = n if scale_cfg else 1
        logits, _, present = model.transformer.forward_with_past(x, attn_mask, past=past, past_length=past_length)
        logits_uncond, _, present_uncond = model.transformer.forward_with_past(x_uncond, attn_mask, embeddings=embeddings, past=past_uncond, past_length=past_length) 

        if past is None:
            past = [present]
            past_uncond = [present_uncond]
        else:
            past.append(present)
            past_uncond.append(present_uncond)

        logits = logits[:, -cur_pred_token_num_per_step:, :] / temperature
        logits_uncond = logits_uncond[:, -cur_pred_token_num_per_step:, :] / temperature

        t = cfg_ratio * ratio
        logits_blend = (1 + t) * logits - t * logits_uncond

        if top_k is not None:
            for i in range(cur_pred_token_num_per_step):
                logits_blend[:, i] = top_k_top_p_filtering(logits_blend[:, i], top_k=top_k, top_p=top_p)
            # logits_blend = top_k_top_p_filtering(logits_blend, top_k=top_k, top_p=top_p)

        probs = F.softmax(logits_blend, dim=-1)
        if not sample_logits:
            _, x = torch.topk(probs, k=1, dim=-1)
        else:
            x = torch.zeros(x.shape[0], cur_pred_token_num_per_step, dtype=x.dtype, device=x.device)
            for i in range(cur_pred_token_num_per_step):
                x[:, i] = torch.multinomial(probs[:, i], num_samples=1)
        if x.ndim == 3:
            x = x.squeeze(-1)  # [n, seq_len]

        if n == pred_token_num_per_step - 1:  # transfer to nbp
            middle_bos = torch.arange(1, pred_token_num_per_step, device=x.device).long().repeat(x.shape[0], 1)
            x = torch.cat([x, middle_bos], dim=-1)

        x_uncond = x

        # append to the sequence and continue
        sample = torch.cat((sample, x), dim=1)

        past_length += cur_pred_token_num_per_step

    del past
    sample = sample[:, cond_bov_len:]  # cut conditioning off
    middle_bos_len = middle_bos.shape[1]
    sample = torch.cat([sample[:, :pred_token_num_per_step], sample[:, pred_token_num_per_step + middle_bos_len:]], dim=1)  # remove middle bos

    return sample
