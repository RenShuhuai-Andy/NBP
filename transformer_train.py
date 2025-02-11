import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from nbp import Net2NetTransformer, HybridNet2NetTransformer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import wandb
from nbp import VideoData
from nbp.modules.callbacks import ImageLogger, VideoLogger


def main():
    pl.seed_everything(1234)

    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer', type=str, default='magvit2')
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--pretrained_ckpt', type=str, default=None)
    parser.add_argument('--token_number_per_step', type=int, default=1)
    parser.add_argument('--use_deepspeed', action='store_true')
    parser.add_argument('--save_every_n_train_steps', type=int, default=10000)
    parser.add_argument('--best_checkpoint_monitor', type=str, default='val/loss')
    parser.add_argument('--hybrid', action='store_true')
    parser.add_argument('--log_gen_res', action='store_true')
    parser = pl.Trainer.add_argparse_args(parser)
    parser = Net2NetTransformer.add_model_specific_args(parser)
    parser = VideoData.add_data_specific_args(parser)
    args = parser.parse_args()

    data = VideoData(args)
    # pre-make relevant cached files if necessary
    # data.train_dataloader()
    # data.test_dataloader()

    args.class_cond_dim = data.n_classes if not args.unconditional and args.cond_stage_key=='label' else None
    if args.hybrid:
        model = HybridNet2NetTransformer(args, ckpt_path=args.pretrained_ckpt, first_stage_key=args.first_stage_key, cond_stage_key=args.cond_stage_key)
    else:
        model = Net2NetTransformer(args, ckpt_path=args.pretrained_ckpt, first_stage_key=args.first_stage_key, cond_stage_key=args.cond_stage_key)
    # print(ModelSummary(model))

    # configure learning rate
    bs, base_lr = args.batch_size, args.base_lr
    ngpu = args.gpus
    model.learning_rate = args.base_lr
    

    callbacks = []
    callbacks.append(ModelCheckpoint(every_n_train_steps=args.save_every_n_train_steps, save_top_k=-1, filename='{epoch}-{step}-{train/loss:.2f}'))
    callbacks.append(ModelCheckpoint(monitor=args.best_checkpoint_monitor, mode='min', save_top_k=1, filename='best_checkpoint'))
    callbacks.append(LearningRateMonitor(logging_interval='step'))

    if args.log_gen_res:
        print("Log the reconstructed videos...")
        callbacks.append(VideoLogger(batch_frequency=args.save_every_n_train_steps, max_videos=4, clamp=True, tokenizer=args.tokenizer))

    kwargs = dict()
    if args.gpus > 1:
        # find_unused_parameters = False to support gradient checkpointing
        plugins = ["deepspeed_stage_1"] if args.use_deepspeed else [pl.plugins.DDPPlugin(find_unused_parameters=False)]
        kwargs = dict(gpus=args.gpus,
                      plugins=plugins)

    if args.bf16:
        kwargs["precision"] = "bf16"
    
    if args.fp16:
        kwargs["precision"] = 16

    # load the most recent checkpoint file
    base_dir = os.path.join(args.default_root_dir, 'nbp')
    version_id_used = 0

    if os.path.exists(base_dir):
        log_folder = ckpt_file = ''
        
        versions = os.listdir(base_dir)
        if len(versions) > 0:
            # versions = sorted(versions, key = lambda x : int(x.split('_')[1]))
            versions = sorted(versions)
            log_folder = versions[-1]
            version_id_used = log_folder

        if args.resume_from_checkpoint is not None:
            print('will start from the recent ckpt %s' % args.resume_from_checkpoint)
        elif len(log_folder) > 0:
            ckpts_folder = os.path.join(base_dir, log_folder, 'checkpoints')
            print(f"ckpts_folder: {ckpts_folder}")  # |-- nbp/*version—id*/checkpoints/
            ckpt_folders = []
            if len(ckpt_file) == 0 and len(os.listdir(ckpts_folder)) > 0:
                ckpt_folders = os.listdir(ckpts_folder)
                if args.use_deepspeed:
                    # |-- best_checkpoint*/
                    #       |-- global_step*/
                    #       |-- latest
                    #       |-- zero_to_fp32.py
                    ckpt_folders = [c for c in ckpt_folders if c.startswith("best_checkpoint")]  # only best_checkpoint* folders contain ckpt files
                    latest_idx = []
                    for c in ckpt_folders:
                        try:
                            assert any(['global' in f for f in os.listdir(os.path.join(ckpts_folder, c))])  # assert the global_step folder exists
                            with open(os.path.join(ckpts_folder, c, 'latest'), 'r') as f:
                                latest_idx.append(int(f.read().replace('global_step', '')))  # get the latest global_step
                        except:
                            latest_idx.append(0)
                    ckpt_folders = [(c, l) for c, l in zip(ckpt_folders, latest_idx)]
                    ckpt_folders = sorted(ckpt_folders, key = lambda x : x[1])
                    print(f"latest ckpt is global step {ckpt_folders[-1][1]} in {ckpt_folders[-1][0]}")
                    ckpt_folders = [c[0] for c in ckpt_folders]
                else:
                    # |-- best_checkpoint.ckpt
                    # |-- epoch=x-step=xxx-train/
                    #       |-- loss=xxx.ckpt
                    ckpt_folders = [c for c in ckpt_folders if c.startswith("epoch")]
                    ckpt_folders = sorted(ckpt_folders, key = lambda x : int(x.split("=")[2].split("-")[0]))
                # val_check_interval

            if len(ckpt_folders) > 0:
                have_read = False
                ckpt_folder_index = -1
                while (not have_read) and (ckpt_folder_index >= -len(ckpt_folders)):
                    ckpt_folder = os.path.join(ckpts_folder, ckpt_folders[ckpt_folder_index])
                    print(f"trying read ckpt file from: {ckpt_folder}")
                    if args.use_deepspeed:
                        have_read = True
                        args.resume_from_checkpoint = ckpt_folder
                        print('will start from the recent ckpt %s' % args.resume_from_checkpoint)
                    else:
                        ckpt_files = os.listdir(ckpt_folder)
                        if len(ckpt_files) > 0:
                            have_read = True
                            args.resume_from_checkpoint = os.path.join(ckpt_folder, ckpt_files[0])
                            print('will start from the recent ckpt %s' % args.resume_from_checkpoint)
                    ckpt_folder_index -= 1

    print(f"Train from the version: {version_id_used}.")

    os.makedirs(args.default_root_dir, exist_ok=True)
    wandb_logger = WandbLogger(project="nbp", name=os.path.basename(args.default_root_dir), save_dir=args.default_root_dir, config=args, version=version_id_used, offline=True)
    trainer = pl.Trainer.from_argparse_args(args, log_every_n_steps=49, logger=wandb_logger, callbacks=callbacks,
                                            max_steps=args.max_steps, **kwargs)


    trainer.fit(model, data)


if __name__ == '__main__':
    main()
