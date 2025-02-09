save_dir='/path/to/ckpt/OmniTokenizer/k600_ar_trans2_semi_ar/k600_fp_eval2048_val_ep82_step546120/topp0.90_topk2048'

python3 evaluation/fvd_external.py --dataset k600 --gen_dir ${save_dir}/recon/ --gt_dir ${save_dir}/input/ --resolution 64 --num_videos 2048