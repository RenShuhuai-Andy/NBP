python3 tokenizer_eval.py \
        --ckpt "/path/to/ckpt/NBP-tokenizer-k600/magvit2_k600.pt" \
        --tokenizer "magvit2" --batch_size 8 \
        --save "/path/to/ckpt/OmniTokenizer/recons/magvit2" \
        --data_path "/path/to/data/kinetics-dataset" --train_datalist "/path/to/annotations/k600_train.txt" --val_datalist "/path/to/annotations/k600_val.txt" \
        --resolution 128 --dataset "imagenet_k600" --loader_type "joint" --sequence_length 17  # --save_videos --seed 9778

