import argparse
import random

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from timm.scheduler.cosine_lr import CosineLRScheduler
from transformers import LlamaTokenizer
from .utils import shift_dim, accuracy, comp_getattr, ForkedPdb
from .modules.gpt import GPT
from .modules.encoders import Labelator, BOVProvider, Identity

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class Net2NetTransformer(pl.LightningModule):
    def __init__(self,
                 args,
                 ckpt_path=None,
                 ignore_keys=['mask'],  # reset attention mask
                 first_stage_key="video",
                 cond_stage_key="label",
                 pkeep=1.0,
                 bov_token=0,
                 tokenizer_path=None,
                 llama_tokenizer=None
                 ):
        super().__init__()
        self.args = args
        self.class_cond_dim = args.class_cond_dim
        self.be_unconditional = args.unconditional
        self.bov_token = bov_token  # special token: begin_of_video

        self.first_stage_key = first_stage_key
        self.cond_stage_key = cond_stage_key
        self.vtokens = args.vtokens

        if tokenizer_path is not None:  # may need to reset tokenizer_path in inf
            args.tokenizer_path = tokenizer_path
        
        if llama_tokenizer is not None:  # may need to reset llama_tokenizer in inf
            args.llama_tokenizer = llama_tokenizer

        self.init_first_stage_from_ckpt(args)
        self.init_cond_stage_from_ckpt(args)

        if not hasattr(args, "p_drop_cond"):
            args.p_drop_cond = None

        if not hasattr(args, "semi_ar"):
            args.semi_ar = False

        if not hasattr(args, "token_number_per_step"):
            args.token_number_per_step = 1

        if not hasattr(args, "cond_token_num"):
            args.cond_token_num = 0

        if not hasattr(args, "hybrid"):
            args.hybrid = None

        if not hasattr(args, "starts_with_bov"):
            args.starts_with_bov = False

        self.bov_provider = BOVProvider(self.bov_token)
        self.p_drop_cond = args.p_drop_cond

        gpt_vocab_size = self.first_stage_vocab_size + self.cond_stage_vocab_size
        if self.be_unconditional:
            n_spe_tokens = 0
            assert args.starts_with_bov == False, "for be_unconditional, args.starts_with_bov must be False"
            assert args.hybrid == False, "for be_unconditional, args.hybrid must be False"
        else:
            if args.hybrid:
                n_spe_tokens = args.token_number_per_step
                assert args.starts_with_bov == True, "for hybrid, starts_with_bov must be True"
            else:
                n_spe_tokens = 1
        gpt_vocab_size += n_spe_tokens

        if not hasattr(args, "transformer_dropout"):
            args.transformer_dropout = 0.

        print(f"args in Net2NetTransformer: {args}")

        self.transformer = GPT(args, gpt_vocab_size, args.block_size, n_layer=args.n_layer, n_head=args.n_head,
                               n_embd=args.n_embd, vtokens_pos=args.vtokens_pos, n_unmasked=args.n_unmasked,
                               embd_pdrop=args.transformer_dropout, resid_pdrop=args.transformer_dropout, attn_pdrop=0., 
                               semi_ar=args.semi_ar,
                               token_number_per_step=args.token_number_per_step,
                               cond_token_num=args.cond_token_num)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.pkeep = pkeep
        self.save_hyperparameters()
        
        self.automatic_optimization = False
        self.grad_accumulates = args.grad_accumulates
        self.grad_clip_val = args.grad_clip_val

        self.semi_ar = args.semi_ar
        self.token_number_per_step = args.token_number_per_step
        self.cond_token_num = args.cond_token_num

    def init_from_ckpt(self, path, ignore_keys=list()):
        ckpt = torch.load(path, map_location="cpu")
        sd = ckpt["state_dict"]
        sd = {(k if not k.startswith('module.') else k.replace("module.", "")): v for k, v in sd.items()}
        print(f'len(sd): {len(sd)}')
        keys_to_delete = [k for k in sd.keys() for ik in ignore_keys if ik in k]
        for k in keys_to_delete:
            print("Deleting key {} from state_dict.".format(k))
            del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def init_first_stage_from_ckpt(self, args):
        from .download import load_magvit2
        if not args.vtokens:
            if args.tokenizer == 'magvit2':
                print(f"Loading magvit2 from {args.tokenizer_path}")
                self.first_stage_model = load_magvit2(args.tokenizer_path, args.resolution)
                for p in self.first_stage_model.parameters():
                    p.requires_grad = False
                self.first_stage_model.eval()
                self.first_stage_vocab_size = 64000
            else:
                raise NotImplementedError()
        else:
            self.first_stage_model = None
            self.first_stage_vocab_size = 16384
            # self.first_stage_vocab_size = self.args.first_stage_vocab_size

    def init_cond_stage_from_ckpt(self, args):
        if self.cond_stage_key=='label' and not self.be_unconditional:
            model = Labelator(n_classes=args.class_cond_dim)
            model = model.eval()
            model.train = disabled_train
            self.cond_stage_model = model
            self.cond_stage_vocab_size = self.class_cond_dim
            assert args.cond_token_num == 1
        elif self.cond_stage_key=='text':
            self.cond_stage_model = LlamaTokenizer.from_pretrained(args.llama_tokenizer, use_fast=False)
            self.cond_stage_vocab_size = self.cond_stage_model.total_vocab_size
            self.cond_stage_model.pad_token = self.cond_stage_model.eos_token
            assert args.cond_token_num >= 8
        elif self.be_unconditional:
            print(f"Using no cond stage. Assuming the training is intended to be unconditional. "
                  f"Prepending {self.bov_token} as a bov token.")
            self.be_unconditional = True
            self.cond_stage_key = self.first_stage_key
            self.cond_stage_model = None
            self.cond_stage_vocab_size = 0
        else:
            ValueError('conditional model %s is not implementated'%self.cond_stage_key)

    def forward(self, x, c, cbox=None):
        # one step to produce the logits
        _, z_indices = self.encode_to_z(x)
        _, c_indices = self.encode_to_c(c)
        c_indices = c_indices.long().to(x.device)

        bov, c_indices, z_indices, a_indices = self.process_b_c_z_a(c_indices, z_indices)

        if self.p_drop_cond is not None:
            if random.random() > self.p_drop_cond:
                cz_indices = torch.cat((c_indices, bov, a_indices), dim=1)
                embeddings = None
            else:  # drop the condition, use uncond_embedding in gpt
                cz_indices = torch.cat((bov, a_indices), dim=1)
                embeddings = self.transformer.uncond_embedding
                embeddings = embeddings.unsqueeze(0).repeat(bov.shape[0], 1, 1)  # [bsz, cond_token_num, dim]
        else:
            cz_indices = torch.cat((c_indices, bov, a_indices), dim=1)
            embeddings = None
        
        cond_bov_len = c_indices.shape[1] + bov.shape[1]
        seq_len = cz_indices.shape[1] - self.args.token_number_per_step  # trimming tokens in the last block
        attn_mask = self.get_attn_mask(c_indices, seq_len)

        # target includes all sequence elements (no need to handle first one
        # differently because we are conditioning)
        target = z_indices if self.args.starts_with_bov else z_indices[:, self.token_number_per_step:]
        # make the prediction
        logits, _ = self.transformer(cz_indices[:, :-self.token_number_per_step], attn_mask, embeddings=embeddings, cbox=cbox)
        # cut off conditioning outputs - output i corresponds to p(z_i | z_{<i}, c)
        cut_len = c_indices.shape[1] if self.args.hybrid else cond_bov_len
        logits = logits[:, cut_len:]
        
        assert logits.shape[1] == target.shape[1], f"logits.shape[1]: {logits.shape[1]} != target.shape[1]: {target.shape[1]}"
        return logits, target

    def get_attn_mask(self, c_indices, seq_len):
        '''
        c_indices: condition. doesn't contain bov. may have been padded for text condition. size: [B, L]
        seq_len: sequence length for attn_mask. should >= input token sequence for training
        prefix_len: prefix length for attn_mask. doesn't contain bov.  # todo, equals to condition length?
        '''
        bsz = c_indices.shape[0]
        prefix_len = c_indices.shape[1]
        attn_mask = torch.zeros((bsz, seq_len, seq_len))
        if self.cond_stage_key=='text':
            attn_mask_for_c = attn_mask[:, :, :prefix_len]
            c_indices_for_c = c_indices[:, :prefix_len]  # [] if prefix_len is 0
            nopad_pos = c_indices_for_c != (self.cond_stage_model.pad_token_id + 1)  # +1 for token id shift
            attn_mask_for_c[nopad_pos.unsqueeze(1).expand(-1, seq_len, -1)] = 1  # all tokens can attend to text tokens, except for the pad tokens
            attn_mask[:, :, :prefix_len] = attn_mask_for_c
        else:
            attn_mask[:, :, :prefix_len] = 1  # all tokens can attend to condition tokens
        for i, j in zip(range(prefix_len, seq_len, self.token_number_per_step),
                        range(prefix_len, seq_len, self.token_number_per_step)):
            attn_mask[:, i:i + self.token_number_per_step, prefix_len:j + self.token_number_per_step] = 1  # video tokens can attend to all video tokens before them

        # # debug attn mask
        # from matplotlib import pyplot as plt
        # data_numpy = attn_mask[0].cpu().numpy()
        # plt.imshow(data_numpy, cmap='hot', interpolation='nearest')
        # plt.colorbar()
        # plt.savefig('attn_mask.png')

        attn_mask = attn_mask.unsqueeze(1).to(c_indices.device)  # [bsz, nh, seq_len, seq_len]
        return attn_mask

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float('Inf')
        return out

    @torch.no_grad()
    def encode_to_z(self, x):
        """
        x: [bsz, c, t, h, w]
        """
        if self.vtokens:
            targets = x.reshape(x.shape[0], -1)
        else:
            if self.args.tokenizer == 'magvit2':
                _, targets = self.first_stage_model.encode(x, quantize=True)
                targets = targets.reshape(targets.shape[0], -1).to(torch.int64)
            else:
                ValueError('tokenizer %s is not implementated' % self.args.tokenizer)
        return x, targets

    @torch.no_grad()
    def encode_to_c(self, c):
        if isinstance(self.cond_stage_model, Labelator) or isinstance(self.cond_stage_model, BOVProvider):
            quant_c, indices = self.cond_stage_model.encode(c)
        elif isinstance(self.cond_stage_model, LlamaTokenizer):
            text_inputs = self.cond_stage_model(c, return_tensors="pt", padding="max_length", 
                                                max_length=self.cond_token_num, truncation=True, add_special_tokens=True)
            quant_c = indices = text_inputs.input_ids
        else:  # self.cond_stage_model is None
            quant_c = indices = torch.empty(len(c), 0)
        
        if len(indices.shape) > 2:
            indices = indices.view(c.shape[0], -1)
        return quant_c, indices

    def get_input(self, key, batch):
        x = batch[key]
        # if x.dtype == torch.double:
            # x = x.float()
        return x

    def get_xc(self, batch, N=None):
        """x = self.get_input(self.first_stage_key, batch)
        c = self.get_input(self.cond_stage_key, batch)
        if N is not None:
            x = x[:N]
            c = c[:N]"""
        if isinstance(batch, dict):
            x = batch[self.first_stage_key]
            c = batch[self.cond_stage_key]
        
        else:
            assert isinstance(batch, list) and len(batch) == 1
            x = batch[0][self.first_stage_key]
            c = batch[0][self.cond_stage_key]

        if N is not None:
            x = x[:N]
            c = c[:N]
        
        return x, c

    def process_b_c_z_a(self, c_indices, z_indices):
        if self.args.starts_with_bov:
            _, bov = self.bov_provider.encode(c_indices)
            bov = bov.repeat(1, self.token_number_per_step)  # repeat bov for nbp
            n_bov_idx = 1
        else:
            bsz = c_indices.shape[0]
            bov = torch.empty(bsz, 0).long().to(c_indices.device)
            n_bov_idx = 0
        c_indices = c_indices + n_bov_idx  # shift token ids because of newly-added bov
        z_indices = z_indices + self.cond_stage_vocab_size + n_bov_idx  # shift token ids because of bov and cond
    
        if self.training and self.pkeep < 1.0:
            mask = torch.bernoulli(self.pkeep*torch.ones(z_indices.shape,
                                                         device=z_indices.device))
            mask = mask.round().to(dtype=torch.int64)
            r_indices = torch.randint_like(z_indices, self.transformer.config.vocab_size)
            a_indices = mask*z_indices+(1-mask)*r_indices
        else:
            a_indices = z_indices
        return bov, c_indices, z_indices, a_indices

    def shared_step(self, batch, batch_idx):
        if not self.vtokens:
            self.first_stage_model.eval()
        x, c = self.get_xc(batch)
        if self.args.vtokens_pos:
            cbox = batch['cbox']
        else:
            cbox = None
        
        logits, target = self(x, c, cbox)  # logits: [bsz, seq_len, vocab_size], target: [bsz, seq_len]
        # print(logits.shape, target.shape)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        acc1, acc5 = accuracy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), topk=(1, 5))
        return loss, acc1, acc5

    def training_step(self, batch, batch_idx):
        sch = self.lr_schedulers()
        opt = self.optimizers()

        loss, acc1, acc5 = self.shared_step(batch, batch_idx)
        # print(batch_idx, loss)

        self.manual_backward(loss)

        cur_global_step = self.global_step
        if (cur_global_step + 1) % self.grad_accumulates == 0:
            if self.grad_clip_val is not None:
                self.clip_gradients(opt, gradient_clip_val=self.grad_clip_val)
                
            opt.step()
            
            sch.step(cur_global_step)
            opt.zero_grad()

        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('train/acc1', acc1, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('train/acc5', acc5, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc1, acc5 = self.shared_step(batch, batch_idx)
        self.log("val/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('val/acc1', acc1, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('val/acc5', acc5, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        """
        Following minGPT:
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.transformer.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')
        if self.args.vtokens_pos:
            no_decay.add('vtokens_pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.transformer.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.learning_rate, betas=(0.9, 0.95))

        lr_min = self.args.lr_min
        train_iters = self.args.max_steps
        warmup_steps = self.args.warmup_steps
        warmup_lr_init = self.args.warmup_lr_init

       
        scheduler = CosineLRScheduler(
            optimizer,
            lr_min = lr_min,
            t_initial = train_iters,
            warmup_lr_init=warmup_lr_init,
            warmup_t=warmup_steps,
            cycle_mul = 1.,
            cycle_limit=1,
            t_in_epochs=True,
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
    

    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        log = dict()
        if isinstance(batch, list):
            batch = batch[0]

        x = batch[self.first_stage_key]
        c = batch[self.cond_stage_key]

        logits, _ = self(x, c)
        probs = F.softmax(logits, dim=-1)
        _, ix = torch.topk(probs, k=1, dim=-1)
        
        index = torch.clamp(ix-self.cond_stage_vocab_size, min=0, max=self.first_stage_vocab_size-1).squeeze(-1)
        if self.args.tokenizer == 'magvit2':
            x_rec = self.first_stage_model.decode_from_code_indices(index)
            x_rec = torch.clamp(x_rec, 0, 1)
        else:
            ValueError('tokenizer %s is not implementated' % self.args.tokenizer)

        log["inputs"] = x
        log["reconstructions"] = x_rec
        return log

    def log_videos(self, batch, **kwargs):
        log = dict()
        if isinstance(batch, list):
            batch = batch[0]
        
        x = batch[self.first_stage_key]
        c = batch[self.cond_stage_key]

        logits, _ = self(x, c)
        probs = F.softmax(logits, dim=-1)
        _, ix = torch.topk(probs, k=1, dim=-1)
        
        index = torch.clamp(ix-self.cond_stage_vocab_size, min=0, max=self.first_stage_vocab_size-1).squeeze(-1)

        if self.args.tokenizer == 'magvit2':
            x_rec = self.first_stage_model.decode_from_code_indices(index)
            x_rec = torch.clamp(x_rec, 0, 1)
        else:
            ValueError('tokenizer %s is not implementated' % self.args.tokenizer)


        log["inputs"] = x
        log["reconstructions"] = x_rec
        return log


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--tokenizer_path', type=str, help='path to tokenizer ckpt, or model name to download pretrained')
        parser.add_argument('--unconditional', action='store_true')
        parser.add_argument('--base_lr', type=float, default=4.5e-06)
        # VideoGPT hyperparmeters
        parser.add_argument('--vocab_size', type=int, default=16384)
        parser.add_argument('--first_stage_vocab_size', type=int, default=16384)

        parser.add_argument('--p_drop_cond', type=float, default=None)
        parser.add_argument('--block_size', type=int, default=256)
        parser.add_argument('--n_layer', type=int, default=48)
        parser.add_argument('--n_head', type=int, default=24)
        parser.add_argument('--n_embd', type=int, default=1536)
        parser.add_argument('--n_unmasked', type=int, default=0)
        parser.add_argument('--transformer_dropout', type=float, default=0.1)
        
        parser.add_argument('--first_stage_key', type=str, default='video', choices=['video'])
        parser.add_argument('--cond_stage_key', type=str, default='label', choices=['label', 'text', 'stft'])
        parser.add_argument('--llama_tokenizer', type=str, default='llama-2-7b-chat-hf')
        parser.add_argument('--cond_token_num', type=int, default=0)

        parser.add_argument('--lr_min', type=float, default=0.)
        parser.add_argument('--warmup_steps', type=int, default=0)
        parser.add_argument('--warmup_lr_init', type=float, default=0.)
        parser.add_argument('--grad_accumulates', type=int, default=1)
        parser.add_argument('--grad_clip_val', type=float, default=1.0)

        parser.add_argument('--semi_ar', action='store_true')
        parser.add_argument('--starts_with_bov', action='store_true')
        return parser


class HybridNet2NetTransformer(Net2NetTransformer):
    def __init__(self,
                 args,
                 ckpt_path=None,
                 ignore_keys=['mask'],  # reset attention mask
                 first_stage_key="video",
                 cond_stage_key="label",
                 pkeep=1.0,
                 bov_token=0,
                 tokenizer_path=None,
                 llama_tokenizer=None
                ):
        super().__init__(args, ckpt_path, ignore_keys, first_stage_key, cond_stage_key, pkeep, bov_token, tokenizer_path, llama_tokenizer)

    def process_b_c_z_a(self, c_indices, z_indices):
        _, bov = self.bov_provider.encode(c_indices)
        c_indices = c_indices + self.args.token_number_per_step  # shift token ids because of bov and middle_bos
        z_indices = z_indices + self.cond_stage_vocab_size + self.args.token_number_per_step  # shift token ids because of bov and cond

        if self.training and self.pkeep < 1.0:
            mask = torch.bernoulli(self.pkeep*torch.ones(z_indices.shape,
                                                         device=z_indices.device))
            mask = mask.round().to(dtype=torch.int64)
            r_indices = torch.randint_like(z_indices, self.transformer.config.vocab_size)
            a_indices = mask*z_indices+(1-mask)*r_indices
        else:
            a_indices = z_indices

        ntp_part = a_indices[:, :self.args.token_number_per_step]
        nbp_part = a_indices[:, self.args.token_number_per_step:]
        middle_bos = torch.arange(1, self.args.token_number_per_step, device=a_indices.device).long().repeat(a_indices.shape[0], 1)
        a_indices = torch.cat([ntp_part, middle_bos, nbp_part], dim=-1)
        return bov, c_indices, z_indices, a_indices

    def get_attn_mask(self, c_indices, seq_len):
        '''
        c_indices: condition. doesn't contain bov. may have been padded for text condition. size: [B, L]
        seq_len: sequence length for attn_mask. should >= input token sequence for training
        '''
        bsz = c_indices.shape[0]
        prefix_len = c_indices.shape[1]
        attn_mask = torch.tril(torch.ones(seq_len, seq_len)).repeat(bsz, 1, 1)
        if self.cond_stage_key=='text':
            attn_mask_for_c = attn_mask[:, :, :prefix_len]
            c_indices_for_c = c_indices[:, :prefix_len]  # [] if prefix_len is 0
            nopad_pos = c_indices_for_c != (self.cond_stage_model.pad_token_id + 1)  # +1 for token id shift (because of bov)
            attn_mask_for_c[nopad_pos.unsqueeze(1).expand(-1, seq_len, -1)] = 1  # all tokens can attend to text tokens, except for the pad tokens
            attn_mask[:, :, :prefix_len] = attn_mask_for_c
        else:
            attn_mask[:, :, :prefix_len] = 1  # all tokens can attend to condition tokens

        nbp_start = prefix_len + self.token_number_per_step  # between condition and nbp_start, we use causal mask
        for i, j in zip(range(nbp_start, seq_len, self.token_number_per_step),
                        range(nbp_start, seq_len, self.token_number_per_step)):
            attn_mask[:, i:i + self.token_number_per_step, nbp_start:j + self.token_number_per_step] = 1  # video tokens can attend to all video tokens before them

        # # debug attn mask
        # from matplotlib import pyplot as plt
        # data_numpy = attn_mask[0].cpu().numpy()
        # plt.imshow(data_numpy, cmap='hot', interpolation='nearest')
        # plt.colorbar()
        # plt.savefig('attn_mask.png')

        attn_mask = attn_mask.unsqueeze(1).to(c_indices.device)  # [bsz, nh, seq_len, seq_len]
        return attn_mask
